"""Monitoring summary and ground truth evaluation service."""

import json
import time
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.db.models import PredictionLogRecord
from app.db.repositories import MonitoringAlertRepository, PredictionLogRepository
from app.ml.evaluation import evaluate_classification, evaluate_regression
from app.schemas.monitoring import AlertItem, DelayedEvaluationMetric, LatencyStats
from app.services.model_registry_service import ModelRegistryService


class MonitoringService:
    def __init__(self, db: Session):
        self.db = db
        self.log_repo = PredictionLogRepository(db)
        self.alert_repo = MonitoringAlertRepository(db)
        self.registry_service = ModelRegistryService(db)

    def get_latency_summary(self, model_id: str | None = None) -> LatencyStats:
        return self.log_repo.get_latency_stats(model_id)

    def list_alerts(self, model_id: str | None = None, status: str | None = None) -> list[AlertItem]:
        records = self.alert_repo.list_alerts(model_id, status)
        items = []
        for r in records:
            items.append(
                AlertItem(
                    alert_id=r.id,
                    model_id=r.model_id,
                    alert_type=r.alert_type,
                    severity=r.severity,
                    message=r.message,
                    details=json.loads(r.details_json) if r.details_json else {},
                    status=r.status,
                    created_at=r.created_at.isoformat() if r.created_at else "",
                    resolved_at=r.resolved_at.isoformat() if r.resolved_at else None,
                )
            )
        return items

    def update_alert_status(self, alert_id: str, new_status: str) -> AlertItem | None:
        rec = self.alert_repo.update_alert_status(alert_id, new_status)
        if not rec:
            return None
        return AlertItem(
            alert_id=rec.id,
            model_id=rec.model_id,
            alert_type=rec.alert_type,
            severity=rec.severity,
            message=rec.message,
            details=json.loads(rec.details_json) if rec.details_json else {},
            status=rec.status,
            created_at=rec.created_at.isoformat() if rec.created_at else "",
            resolved_at=rec.resolved_at.isoformat() if rec.resolved_at else None,
        )

    def attach_ground_truth(self, request_id: str, actual_label: Any) -> bool:
        record = self.log_repo.attach_ground_truth(request_id, actual_label)
        return bool(record is not None)

    def compute_ground_truth_metrics(self, model_id: str | None = None) -> DelayedEvaluationMetric | None:
        """Compute realized production metrics for predictions that received actual labels."""
        model = self.registry_service.get_model_by_id(model_id) if model_id else self.registry_service.get_active_model()
        if not model:
            return None

        query = self.db.query(PredictionLogRecord).filter(
            PredictionLogRecord.model_id == model.model_id,
            PredictionLogRecord.actual_label.isnot(None),
            PredictionLogRecord.status == "success",
        )
        records = query.all()

        if len(records) < 5:
            return None

        y_true = []
        y_pred = []
        for r in records:
            try:
                preds = json.loads(r.predictions_json)
                if preds:
                    y_pred.append(preds[0])
                    y_true.append(float(r.actual_label) if model.task_type == "regression" else r.actual_label)
            except Exception:
                pass

        if len(y_true) < 5:
            return None

        if model.task_type == "regression":
            metrics = evaluate_regression(np.array(y_true, dtype=float), np.array(y_pred, dtype=float))
        else:
            metrics = evaluate_classification(np.array(y_true), np.array(y_pred), task_type=model.task_type)

        return DelayedEvaluationMetric(
            model_id=model.model_id,
            task_type=model.task_type,
            sample_count=len(y_true),
            metrics=metrics,
            evaluated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
