"""Inference service executing predictions against active model and logging metrics."""

import time

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import ModelNotActiveError, ModelNotFoundError, PredictionInputError
from app.core.logging_config import logger
from app.db.repositories import MonitoringAlertRepository, PredictionLogRepository
from app.schemas.predictions import PredictionItemResult, PredictionRequest, PredictionResponse
from app.services.model_registry_service import ModelRegistryService
from app.utils.identifiers import generate_unique_id


class PredictionService:
    def __init__(self, db: Session):
        self.db = db
        self.registry_service = ModelRegistryService(db)
        self.log_repo = PredictionLogRepository(db)
        self.alert_repo = MonitoringAlertRepository(db)
        self._cache = {}  # In-memory artifact cache {artifact_path: artifact_obj}

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        t0 = time.time()
        request_id = generate_unique_id("req")

        # 1. Resolve model
        if request.model_id:
            model = self.registry_service.get_model_by_id(request.model_id)
        else:
            model = self.registry_service.get_active_model()
            if not model:
                raise ModelNotActiveError("No active model is deployed for serving.")

        # 2. Validate input feature records against schema
        if not request.features:
            raise PredictionInputError("Input feature list cannot be empty.")

        input_df = pd.DataFrame(request.features)
        expected_schema = {f.name: f for f in model.feature_schema}

        missing_features = [name for name in expected_schema if name not in input_df.columns]
        if missing_features:
            raise PredictionInputError(f"Missing required feature(s): {missing_features}")

        # Ensure correct column ordering
        ordered_features = [f.name for f in model.feature_schema]
        input_df = input_df[ordered_features]

        # 3. Load model artifact
        try:
            if model.artifact_path not in self._cache:
                self._cache[model.artifact_path] = joblib.load(model.artifact_path)
            artifact = self._cache[model.artifact_path]
            pipeline = artifact["pipeline"]
        except Exception as e:
            logger.error(f"Failed to load model artifact at {model.artifact_path}: {e}")
            raise ModelNotFoundError(f"Model artifact at {model.artifact_path} could not be loaded: {e}")

        # 4. Generate predictions
        try:
            raw_preds = pipeline.predict(input_df)

            probabilities = None
            if model.task_type in ("binary_classification", "multiclass_classification"):
                if hasattr(pipeline, "predict_proba"):
                    try:
                        proba_matrix = pipeline.predict_proba(input_df)
                        classes = getattr(pipeline.named_steps["model"], "classes_", range(proba_matrix.shape[1]))
                        probabilities = [
                            {str(cls): round(float(p), 4) for cls, p in zip(classes, row)}
                            for row in proba_matrix
                        ]
                    except Exception:
                        probabilities = None

            results: list[PredictionItemResult] = []
            for i, p in enumerate(raw_preds):
                pred_val = int(p) if isinstance(p, (np.integer, int)) else (
                    round(float(p), 2) if isinstance(p, (np.floating, float)) else str(p)
                )
                probs = probabilities[i] if probabilities else None
                results.append(PredictionItemResult(prediction=pred_val, probabilities=probs))

            latency_ms = round((time.time() - t0) * 1000, 2)

            # 5. Log prediction to database
            self.log_repo.log_prediction(
                request_id=request_id,
                model_id=model.model_id,
                model_version=model.version,
                latency_ms=latency_ms,
                inputs=request.features,
                predictions=[r.prediction for r in results],
                status="success",
            )

            # Check latency alert threshold
            if latency_ms > settings.LATENCY_ALERT_THRESHOLD_MS:
                self.alert_repo.create_alert(
                    alert_id=generate_unique_id("alt"),
                    model_id=model.model_id,
                    alert_type="HIGH_LATENCY",
                    severity="WARNING",
                    message=f"Inference latency {latency_ms}ms exceeded alert threshold of {settings.LATENCY_ALERT_THRESHOLD_MS}ms",
                    details={"latency_ms": latency_ms, "threshold_ms": settings.LATENCY_ALERT_THRESHOLD_MS, "request_id": request_id},
                )

            return PredictionResponse(
                request_id=request_id,
                model_id=model.model_id,
                model_version=model.version,
                task_type=model.task_type,
                predictions=results,
                latency_ms=latency_ms,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

        except Exception as e:
            latency_ms = round((time.time() - t0) * 1000, 2)
            self.log_repo.log_prediction(
                request_id=request_id,
                model_id=model.model_id,
                model_version=model.version,
                latency_ms=latency_ms,
                inputs=request.features,
                predictions=[],
                status="error",
                error_message=str(e),
            )
            raise PredictionInputError(f"Prediction computation failed: {str(e)}")
