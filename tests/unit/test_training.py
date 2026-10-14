"""Unit tests for training algorithms, estimators, and serialization."""

import os

import joblib
import pandas as pd

from app.ml.model_factory import get_model
from app.ml.training import train_models_pipeline
from app.schemas.training import TrainingRequest


def test_get_model_factory():
    reg = get_model("random_forest", task_type="regression", random_state=42)
    assert hasattr(reg, "fit")
    clf = get_model("logistic_regression", task_type="binary_classification", random_state=42)
    assert hasattr(clf, "predict")


def test_train_regression_pipeline_execution():
    df = pd.DataFrame({
        "size": [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500],
        "rooms": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        "type": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "price": [150, 220, 280, 340, 410, 480, 560, 620, 710, 790],
    })
    req = TrainingRequest(
        dataset_id="test_reg_ds",
        target_column="price",
        task_type="regression",
        models=["linear_regression", "random_forest"],
        test_size=0.2,
        cv_folds=2,
        random_state=42,
    )
    job_resp = train_models_pipeline(df, req, "test_job_1")
    assert job_resp.status == "completed"
    assert len(job_resp.results) == 2
    assert job_resp.best_model_name is not None

    # Verify saved artifact
    artifact_path = job_resp.results[0].artifact_uri
    assert os.path.exists(artifact_path)
    loaded = joblib.load(artifact_path)
    assert "pipeline" in loaded
    assert hasattr(loaded["pipeline"], "predict")


def test_train_classification_pipeline_execution():
    df = pd.DataFrame({
        "feature_x": [1.0, 1.2, 5.0, 5.2, 1.1, 5.1, 1.3, 5.3, 1.0, 5.4],
        "category": ["cat", "cat", "dog", "dog", "cat", "dog", "cat", "dog", "cat", "dog"],
        "label": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    })
    req = TrainingRequest(
        dataset_id="test_clf_ds",
        target_column="label",
        task_type="binary_classification",
        models=["logistic_regression"],
        test_size=0.2,
        cv_folds=2,
        random_state=42,
    )
    job_resp = train_models_pipeline(df, req, "test_job_2")
    assert job_resp.status == "completed"
    assert len(job_resp.results) == 1
    assert "accuracy" in job_resp.results[0].metrics
    assert "f1" in job_resp.results[0].metrics
