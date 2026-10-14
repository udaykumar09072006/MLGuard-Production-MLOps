"""Model monitoring, data drift, and alerting endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_drift_service, get_monitoring_service
from app.core.exceptions import ModelNotActiveError, MonitoringDriftError
from app.schemas.monitoring import (
    AlertItem,
    DelayedEvaluationMetric,
    DriftSummaryResponse,
    GroundTruthSubmission,
    LatencyStats,
)
from app.services.drift_service import DriftService
from app.services.monitoring_service import MonitoringService

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/summary", response_model=LatencyStats)
def get_monitoring_summary(
    model_id: str | None = Query(None),
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
):
    """Retrieve latency percentiles (p50, p95, p99), throughput, and error rates."""
    return monitoring_service.get_latency_summary(model_id)


@router.get("/drift", response_model=DriftSummaryResponse)
def get_data_drift_report(
    model_id: str | None = Query(None),
    drift_service: DriftService = Depends(get_drift_service),
):
    """Run two-sample Kolmogorov-Smirnov and Chi-square tests comparing live traffic to baseline."""
    try:
        return drift_service.calculate_drift(model_id)
    except ModelNotActiveError as e:
        raise HTTPException(status_code=409, detail=e.message)
    except MonitoringDriftError as e:
        raise HTTPException(status_code=400, detail=e.message)


@router.get("/alerts", response_model=list[AlertItem])
def list_monitoring_alerts(
    model_id: str | None = Query(None),
    status_filter: str | None = Query(None, alias="status"),
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
):
    """List monitoring alerts filtered by model ID or status (ACTIVE, ACKNOWLEDGED, RESOLVED)."""
    return monitoring_service.list_alerts(model_id, status_filter)


@router.post("/alerts/{alert_id}/status", response_model=AlertItem)
def update_alert_status(
    alert_id: str,
    new_status: str = Query(..., pattern="^(ACKNOWLEDGED|RESOLVED|ACTIVE)$"),
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
):
    """Acknowledge or resolve an active monitoring alert."""
    updated = monitoring_service.update_alert_status(alert_id, new_status)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return updated


@router.post("/ground-truth", status_code=status.HTTP_200_OK)
def submit_ground_truth(
    submission: GroundTruthSubmission,
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
):
    """Attach delayed ground truth label to prior prediction for realized accuracy tracking."""
    success = monitoring_service.attach_ground_truth(submission.request_id, submission.actual_label)
    if not success:
        raise HTTPException(status_code=404, detail=f"Prediction request '{submission.request_id}' not found.")
    return {"status": "success", "message": "Ground truth label recorded."}


@router.get("/delayed-evaluation", response_model=Optional[DelayedEvaluationMetric])
def get_delayed_evaluation(
    model_id: str | None = Query(None),
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
):
    """Calculate realized production metrics against ground truth labels."""
    metrics = monitoring_service.compute_ground_truth_metrics(model_id)
    if not metrics:
        return None
    return metrics
