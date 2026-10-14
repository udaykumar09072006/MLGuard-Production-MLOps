"""Experiment tracking inspection endpoints."""


from fastapi import APIRouter, Depends

from app.api.dependencies import get_experiment_service
from app.services.experiment_service import ExperimentService

router = APIRouter(prefix="/experiments", tags=["Experiments"])


@router.get("", response_model=list[dict])
def list_experiments(
    experiment_service: ExperimentService = Depends(get_experiment_service),
):
    """Retrieve logged MLflow experiment runs and holdout metrics."""
    return experiment_service.list_recent_runs()
