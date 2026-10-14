"""SQLAlchemy database tables for MLGuard entities."""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text

from app.db.database import Base


class DatasetRecord(Base):
    __tablename__ = "datasets"

    id = Column(String(64), primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    fingerprint_sha256 = Column(String(64), nullable=False)
    suggested_target = Column(String(128), nullable=True)
    suggested_task = Column(String(64), nullable=True)
    columns_json = Column(Text, nullable=False)
    warnings_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingJobRecord(Base):
    __tablename__ = "training_jobs"

    id = Column(String(64), primary_key=True, index=True)
    dataset_id = Column(String(64), nullable=False)
    target_column = Column(String(128), nullable=False)
    task_type = Column(String(64), nullable=False)
    status = Column(String(32), default="pending")  # pending, running, completed, failed
    best_model_name = Column(String(128), nullable=True)
    best_metric_value = Column(Float, nullable=True)
    results_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class ModelRegistryRecord(Base):
    __tablename__ = "model_registry"

    id = Column(String(64), primary_key=True, index=True)
    version = Column(Integer, nullable=False, default=1)
    name = Column(String(128), nullable=False)
    algorithm = Column(String(64), nullable=False)
    task_type = Column(String(64), nullable=False)
    target_column = Column(String(128), nullable=False)
    run_id = Column(String(64), nullable=False)
    metrics_json = Column(Text, nullable=False)
    feature_schema_json = Column(Text, nullable=False)
    is_active = Column(Boolean, default=False)
    validation_status = Column(String(32), default="validated")  # validated, rejected, provisional
    artifact_path = Column(String(512), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    activated_at = Column(DateTime, nullable=True)
    activated_by = Column(String(128), nullable=True)


class PredictionLogRecord(Base):
    __tablename__ = "prediction_logs"

    id = Column(String(64), primary_key=True, index=True)
    model_id = Column(String(64), nullable=False, index=True)
    model_version = Column(Integer, nullable=False)
    latency_ms = Column(Float, nullable=False)
    status = Column(String(32), default="success")  # success, error
    input_features_json = Column(Text, nullable=False)
    predictions_json = Column(Text, nullable=False)
    actual_label = Column(String(128), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class MonitoringAlertRecord(Base):
    __tablename__ = "monitoring_alerts"

    id = Column(String(64), primary_key=True, index=True)
    model_id = Column(String(64), nullable=False, index=True)
    alert_type = Column(String(64), nullable=False)  # DATA_DRIFT, HIGH_LATENCY, HIGH_ERROR_RATE
    severity = Column(String(32), default="WARNING")  # INFO, WARNING, CRITICAL
    message = Column(Text, nullable=False)
    details_json = Column(Text, nullable=False)
    status = Column(String(32), default="ACTIVE")    # ACTIVE, ACKNOWLEDGED, RESOLVED
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    resolved_at = Column(DateTime, nullable=True)
