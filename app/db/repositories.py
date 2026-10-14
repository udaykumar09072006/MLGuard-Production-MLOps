"""Database repository layer providing atomic queries and operations."""

import json
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.db.models import (
    DatasetRecord,
    ModelRegistryRecord,
    MonitoringAlertRecord,
    PredictionLogRecord,
    TrainingJobRecord,
)
from app.schemas.datasets import DatasetSummary
from app.schemas.monitoring import LatencyStats
from app.schemas.training import TrainingJobResponse


class DatasetRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_dataset(self, summary: DatasetSummary) -> DatasetRecord:
        record = DatasetRecord(
            id=summary.dataset_id,
            filename=summary.filename,
            file_path=summary.file_path,
            row_count=summary.row_count,
            column_count=summary.column_count,
            file_size_bytes=summary.file_size_bytes,
            fingerprint_sha256=summary.fingerprint_sha256,
            suggested_target=summary.suggested_target,
            suggested_task=summary.suggested_task,
            columns_json=json.dumps([c.model_dump() for c in summary.columns]),
            warnings_json=json.dumps([w.model_dump() for w in summary.warnings]),
        )
        self.db.merge(record)
        self.db.commit()
        return record

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        return self.db.query(DatasetRecord).filter(DatasetRecord.id == dataset_id).first()

    def list_datasets(self) -> list[DatasetRecord]:
        return self.db.query(DatasetRecord).order_by(DatasetRecord.created_at.desc()).all()


class TrainingJobRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_job(self, job_id: str, dataset_id: str, target: str, task: str) -> TrainingJobRecord:
        record = TrainingJobRecord(
            id=job_id,
            dataset_id=dataset_id,
            target_column=target,
            task_type=task,
            status="pending",
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def update_job_completed(self, response: TrainingJobResponse) -> TrainingJobRecord:
        record = self.db.query(TrainingJobRecord).filter(TrainingJobRecord.id == response.job_id).first()
        if record:
            record.status = response.status
            record.best_model_name = response.best_model_name
            record.best_metric_value = response.best_metric_value
            record.results_json = json.dumps([r.model_dump() for r in response.results])
            record.completed_at = datetime.utcnow()
            self.db.commit()
        return record

    def get_job(self, job_id: str) -> TrainingJobRecord | None:
        return self.db.query(TrainingJobRecord).filter(TrainingJobRecord.id == job_id).first()

    def list_jobs(self) -> list[TrainingJobRecord]:
        return self.db.query(TrainingJobRecord).order_by(TrainingJobRecord.created_at.desc()).all()


class ModelRegistryRepository:
    def __init__(self, db: Session):
        self.db = db

    def register_model(
        self,
        model_id: str,
        name: str,
        algorithm: str,
        task_type: str,
        target_column: str,
        run_id: str,
        metrics: dict[str, Any],
        feature_schemas: list[dict[str, Any]],
        artifact_path: str,
    ) -> ModelRegistryRecord:
        # Calculate next version for this model family
        existing = (
            self.db.query(ModelRegistryRecord)
            .filter(ModelRegistryRecord.name == name)
            .count()
        )
        version = existing + 1

        record = ModelRegistryRecord(
            id=model_id,
            version=version,
            name=name,
            algorithm=algorithm,
            task_type=task_type,
            target_column=target_column,
            run_id=run_id,
            metrics_json=json.dumps(metrics),
            feature_schema_json=json.dumps(feature_schemas),
            is_active=False,
            validation_status="validated",
            artifact_path=artifact_path,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_model(self, model_id: str) -> ModelRegistryRecord | None:
        return self.db.query(ModelRegistryRecord).filter(ModelRegistryRecord.id == model_id).first()

    def get_active_model(self) -> ModelRegistryRecord | None:
        return (
            self.db.query(ModelRegistryRecord)
            .filter(ModelRegistryRecord.is_active.is_(True))
            .order_by(ModelRegistryRecord.activated_at.desc())
            .first()
        )

    def activate_model(self, model_id: str, activated_by: str = "system") -> ModelRegistryRecord:
        # Demote all current active models
        self.db.query(ModelRegistryRecord).update({ModelRegistryRecord.is_active: False})

        target = self.db.query(ModelRegistryRecord).filter(ModelRegistryRecord.id == model_id).first()
        if target:
            target.is_active = True
            target.activated_at = datetime.utcnow()
            target.activated_by = activated_by
            self.db.commit()
            self.db.refresh(target)
        return target

    def list_models(self) -> list[ModelRegistryRecord]:
        return self.db.query(ModelRegistryRecord).order_by(ModelRegistryRecord.created_at.desc()).all()


class PredictionLogRepository:
    def __init__(self, db: Session):
        self.db = db

    def log_prediction(
        self,
        request_id: str,
        model_id: str,
        model_version: int,
        latency_ms: float,
        inputs: list[dict[str, Any]],
        predictions: list[Any],
        status: str = "success",
        error_message: str | None = None,
    ) -> PredictionLogRecord:
        record = PredictionLogRecord(
            id=request_id,
            model_id=model_id,
            model_version=model_version,
            latency_ms=latency_ms,
            status=status,
            input_features_json=json.dumps(inputs),
            predictions_json=json.dumps(predictions),
            error_message=error_message,
        )
        self.db.add(record)
        self.db.commit()
        return record

    def get_recent_inputs(self, model_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        query = self.db.query(PredictionLogRecord).filter(PredictionLogRecord.status == "success")
        if model_id:
            query = query.filter(PredictionLogRecord.model_id == model_id)
        records = query.order_by(PredictionLogRecord.created_at.desc()).limit(limit).all()

        flattened = []
        for r in records:
            try:
                items = json.loads(r.input_features_json)
                flattened.extend(items)
            except Exception:
                pass
        return flattened

    def get_latency_stats(self, model_id: str | None = None) -> LatencyStats:
        query = self.db.query(PredictionLogRecord)
        if model_id:
            query = query.filter(PredictionLogRecord.model_id == model_id)
        records = query.all()

        total = len(records)
        if total == 0:
            return LatencyStats(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                error_rate=0.0,
            )

        successes = [r.latency_ms for r in records if r.status == "success"]
        failures = total - len(successes)
        latencies = np.array(successes) if successes else np.array([0.0])

        return LatencyStats(
            total_requests=total,
            successful_requests=len(successes),
            failed_requests=failures,
            avg_latency_ms=round(float(np.mean(latencies)), 2),
            p50_latency_ms=round(float(np.percentile(latencies, 50)), 2),
            p95_latency_ms=round(float(np.percentile(latencies, 95)), 2),
            p99_latency_ms=round(float(np.percentile(latencies, 99)), 2),
            error_rate=round(failures / total, 4),
        )

    def attach_ground_truth(self, request_id: str, actual_label: Any) -> PredictionLogRecord | None:
        record = self.db.query(PredictionLogRecord).filter(PredictionLogRecord.id == request_id).first()
        if record:
            record.actual_label = str(actual_label)
            self.db.commit()
        return record


class MonitoringAlertRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_alert(
        self,
        alert_id: str,
        model_id: str,
        alert_type: str,
        severity: str,
        message: str,
        details: dict[str, Any],
    ) -> MonitoringAlertRecord:
        record = MonitoringAlertRecord(
            id=alert_id,
            model_id=model_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            details_json=json.dumps(details),
            status="ACTIVE",
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def list_alerts(self, model_id: str | None = None, status: str | None = None) -> list[MonitoringAlertRecord]:
        query = self.db.query(MonitoringAlertRecord)
        if model_id:
            query = query.filter(MonitoringAlertRecord.model_id == model_id)
        if status:
            query = query.filter(MonitoringAlertRecord.status == status)
        return query.order_by(MonitoringAlertRecord.created_at.desc()).all()

    def update_alert_status(self, alert_id: str, new_status: str) -> MonitoringAlertRecord | None:
        record = self.db.query(MonitoringAlertRecord).filter(MonitoringAlertRecord.id == alert_id).first()
        if record:
            record.status = new_status
            if new_status == "RESOLVED":
                record.resolved_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(record)
        return record
