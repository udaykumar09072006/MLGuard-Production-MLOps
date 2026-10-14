"""Data drift service executing statistical comparison between reference and production data."""

import os
import time

import joblib
import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import ModelNotActiveError, MonitoringDriftError
from app.db.repositories import MonitoringAlertRepository, PredictionLogRepository
from app.ml.drift_detection import compute_dataset_drift
from app.schemas.monitoring import DriftSummaryResponse
from app.services.model_registry_service import ModelRegistryService
from app.utils.identifiers import generate_unique_id


class DriftService:
    def __init__(self, db: Session):
        self.db = db
        self.registry_service = ModelRegistryService(db)
        self.log_repo = PredictionLogRepository(db)
        self.alert_repo = MonitoringAlertRepository(db)

    def calculate_drift(self, model_id: str | None = None) -> DriftSummaryResponse:
        """Run statistical drift detection on production feature requests vs baseline reference."""
        # 1. Resolve model
        if model_id:
            model = self.registry_service.get_model_by_id(model_id)
        else:
            model = self.registry_service.get_active_model()
            if not model:
                raise ModelNotActiveError("No active model deployed for drift monitoring.")

        # 2. Load reference baseline training data
        if not os.path.exists(model.artifact_path):
            raise MonitoringDriftError(f"Model artifact not found at {model.artifact_path}")

        artifact = joblib.load(model.artifact_path)
        ref_path = artifact.get("reference_data_path")
        if not ref_path or not os.path.exists(ref_path):
            raise MonitoringDriftError("Reference training dataset for this model is unavailable.")

        ref_df = pd.read_parquet(ref_path)

        # 3. Collect recent prediction traffic
        recent_inputs = self.log_repo.get_recent_inputs(model_id=model.model_id, limit=300)
        if len(recent_inputs) < settings.MIN_DRIFT_SAMPLE_SIZE:
            # Not enough production observations yet: return clean baseline comparison
            feature_names = [f.name for f in model.feature_schema]
            return DriftSummaryResponse(
                model_id=model.model_id,
                evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                sample_size=len(recent_inputs),
                features_analyzed=len(feature_names),
                drifted_features_count=0,
                overall_drift_detected=False,
                feature_reports=[],
            )

        curr_df = pd.DataFrame(recent_inputs)
        feature_names = [f.name for f in model.feature_schema if f.name in curr_df.columns]

        # 4. Compute drift across all features
        reports = compute_dataset_drift(
            ref_df=ref_df,
            curr_df=curr_df,
            features=feature_names,
            p_value_threshold=settings.DRIFT_P_VALUE_THRESHOLD,
        )

        drifted_count = sum(1 for r in reports if r.drift_detected)
        overall_drift = drifted_count > 0

        # 5. Generate persistent alert if drift detected
        if overall_drift:
            drifted_names = [r.feature_name for r in reports if r.drift_detected]
            existing_active_alerts = self.alert_repo.list_alerts(model_id=model.model_id, status="ACTIVE")
            drift_alerts = [a for a in existing_active_alerts if a.alert_type == "DATA_DRIFT"]

            if not drift_alerts:
                self.alert_repo.create_alert(
                    alert_id=generate_unique_id("alt"),
                    model_id=model.model_id,
                    alert_type="DATA_DRIFT",
                    severity="WARNING",
                    message=f"Statistical data drift detected in {drifted_count} feature(s): {', '.join(drifted_names)}",
                    details={
                        "drifted_features": drifted_names,
                        "sample_size": len(recent_inputs),
                        "threshold": settings.DRIFT_P_VALUE_THRESHOLD,
                    },
                )

        return DriftSummaryResponse(
            model_id=model.model_id,
            evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            sample_size=len(recent_inputs),
            features_analyzed=len(feature_names),
            drifted_features_count=drifted_count,
            overall_drift_detected=overall_drift,
            feature_reports=reports,
        )
