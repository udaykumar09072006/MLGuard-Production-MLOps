"""Orchestration of ML model training, cross-validation, and pipeline serialization."""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from app.core.config import settings
from app.core.logging_config import logger
from app.ml.data_validation import validate_training_inputs
from app.ml.evaluation import evaluate_classification, evaluate_regression, pick_best_model
from app.ml.model_factory import get_model
from app.ml.preprocessing import build_preprocessor, identify_feature_types
from app.schemas.models import FeatureSchema
from app.schemas.training import ModelEvaluationResult, TrainingJobResponse, TrainingRequest
from app.utils.identifiers import generate_unique_id


def extract_feature_schemas(X: pd.DataFrame) -> list[FeatureSchema]:
    """Build schema metadata to validate future prediction requests."""
    schemas = []
    for col in X.columns:
        series = X[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        dtype_str = str(series.dtype)
        inferred = "numerical" if is_numeric else "categorical"

        min_val = float(series.min()) if is_numeric and len(series.dropna()) else None
        max_val = float(series.max()) if is_numeric and len(series.dropna()) else None
        allowed_cats = [str(c) for c in series.dropna().unique().tolist()] if not is_numeric else None
        default_val = float(series.median()) if is_numeric else (series.mode().iloc[0] if len(series.dropna()) else "")

        schemas.append(
            FeatureSchema(
                name=col,
                dtype=dtype_str,
                inferred_type=inferred,
                required=True,
                allowed_categories=allowed_cats,
                min_val=min_val,
                max_val=max_val,
                default_val=default_val,
            )
        )
    return schemas


def train_models_pipeline(df: pd.DataFrame, request: TrainingRequest, job_id: str) -> TrainingJobResponse:
    """Execute complete end-to-end training pipeline."""
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    logger.info(f"Starting training job {job_id} on dataset {request.dataset_id} for target '{request.target_column}'")

    X, y = validate_training_inputs(df, request.target_column, request.task_type, request.test_size)
    numeric_features, categorical_features = identify_feature_types(X)
    feature_schemas = extract_feature_schemas(X)

    # Train / Test split without data leakage
    stratify = y if request.task_type in ("binary_classification", "multiclass_classification") else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=request.test_size,
            random_state=request.random_state,
            stratify=stratify,
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed ({e}), falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=request.test_size,
            random_state=request.random_state,
            stratify=None,
        )

    # Save reference baseline training data for drift detection
    ref_filename = f"ref_{job_id}.parquet"
    ref_path = os.path.join(settings.REFERENCE_DATA_DIR, ref_filename)
    X_train.to_parquet(ref_path)

    results: list[ModelEvaluationResult] = []

    for algo in request.models:
        t0 = time.time()
        estimator = get_model(algo, request.task_type, random_state=request.random_state)

        # Assemble full pipeline: Preprocessing -> Estimator
        preprocessor = build_preprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            numeric_imputer_strategy=request.numeric_imputer,
            categorical_imputer_strategy=request.categorical_imputer,
            scaling=request.scaling,
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

        # Cross validation on training set
        cv = (
            StratifiedKFold(n_splits=request.cv_folds, shuffle=True, random_state=request.random_state)
            if request.task_type in ("binary_classification", "multiclass_classification")
            else KFold(n_splits=request.cv_folds, shuffle=True, random_state=request.random_state)
        )
        scoring = "r2" if request.task_type == "regression" else "f1_weighted"
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
            cv_scores_list = [round(float(s), 4) for s in cv_scores]
        except Exception as cv_err:
            logger.warning(f"Cross-validation warning for {algo}: {cv_err}")
            cv_scores_list = None
            cv_mean = None
            cv_std = None

        # Fit complete pipeline strictly on X_train
        pipeline.fit(X_train, y_train)

        # Evaluate strictly on held-out test set
        y_pred = pipeline.predict(X_test)
        duration = round(time.time() - t0, 3)

        if request.task_type == "regression":
            metrics = evaluate_regression(y_test.to_numpy(), y_pred)
        else:
            y_proba = None
            if hasattr(pipeline, "predict_proba"):
                try:
                    y_proba = pipeline.predict_proba(X_test)
                except Exception:
                    y_proba = None
            metrics = evaluate_classification(y_test.to_numpy(), y_pred, y_proba, request.task_type)

        run_id = generate_unique_id("run")
        model_name = f"{algo}_{run_id}"

        # Serialize complete pipeline artifact
        artifact_filename = f"{model_name}.joblib"
        artifact_path = os.path.join(settings.MODEL_STORAGE_DIR, artifact_filename)
        joblib.dump(
            {
                "pipeline": pipeline,
                "feature_names": list(X.columns),
                "feature_schemas": [s.model_dump() for s in feature_schemas],
                "target_column": request.target_column,
                "task_type": request.task_type,
                "algorithm": algo,
                "metrics": metrics,
                "reference_data_path": ref_path,
                "trained_at": time.time(),
            },
            artifact_path,
        )

        results.append(
            ModelEvaluationResult(
                model_name=model_name,
                algorithm=algo,
                run_id=run_id,
                metrics=metrics,
                cv_scores=cv_scores_list,
                cv_mean=round(cv_mean, 4) if cv_mean is not None else None,
                cv_std=round(cv_std, 4) if cv_std is not None else None,
                duration_seconds=duration,
                feature_names=list(X.columns),
                artifact_uri=artifact_path,
            )
        )

    # Pick best model
    best_name, best_val = pick_best_model(
        [r.model_dump() for r in results], request.task_type, request.metric_objective
    )

    completed_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    return TrainingJobResponse(
        job_id=job_id,
        status="completed",
        task_type=request.task_type,
        target_column=request.target_column,
        dataset_id=request.dataset_id,
        results=results,
        best_model_name=best_name,
        best_metric_value=round(best_val, 4) if best_val is not None else None,
        created_at=start_time_iso,
        completed_at=completed_time_iso,
    )
