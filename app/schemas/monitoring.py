"""Schemas for latency, data drift, alerts, and delayed ground truth."""

from typing import Any

from pydantic import BaseModel


class LatencyStats(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float


class FeatureDriftReport(BaseModel):
    feature_name: str
    feature_type: str  # "numerical", "categorical"
    test_method: str   # "Kolmogorov-Smirnov", "Chi-Square"
    statistic: float
    p_value: float
    threshold: float
    drift_detected: bool
    reference_summary: dict[str, Any]
    current_summary: dict[str, Any]


class DriftSummaryResponse(BaseModel):
    model_id: str
    evaluated_at: str
    sample_size: int
    features_analyzed: int
    drifted_features_count: int
    overall_drift_detected: bool
    feature_reports: list[FeatureDriftReport]


class AlertItem(BaseModel):
    alert_id: str
    model_id: str
    alert_type: str  # "DATA_DRIFT", "HIGH_LATENCY", "HIGH_ERROR_RATE"
    severity: str    # "WARNING", "CRITICAL", "INFO"
    message: str
    details: dict[str, Any]
    status: str      # "ACTIVE", "ACKNOWLEDGED", "RESOLVED"
    created_at: str
    resolved_at: str | None = None


class GroundTruthSubmission(BaseModel):
    request_id: str
    prediction_index: int = 0
    actual_label: Any


class DelayedEvaluationMetric(BaseModel):
    model_id: str
    task_type: str
    sample_count: int
    metrics: dict[str, float]
    evaluated_at: str
