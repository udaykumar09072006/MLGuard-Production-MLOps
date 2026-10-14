"""Training job submission and status inspection endpoints."""

import json

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_training_service
from app.core.exceptions import DatasetValidationError, ModelTrainingError
from app.schemas.training import TrainingJobResponse, TrainingRequest
from app.services.training_service import TrainingService

router = APIRouter(prefix="/training", tags=["Training"])


@router.post("/jobs", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
def submit_training_job(
    request: TrainingRequest,
    training_service: TrainingService = Depends(get_training_service),
):
    """Start model training across configured algorithms with cross-validation and MLflow logging."""
    try:
        response = training_service.run_training_job(request)
        return response
    except (DatasetValidationError, ModelTrainingError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal training error: {str(e)}")


@router.get("/jobs/{job_id}", response_model=dict)
def get_training_job_status(
    job_id: str,
    training_service: TrainingService = Depends(get_training_service),
):
    """Inspect status and metrics of a training job."""
    job = training_service.job_repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found.")

    return {
        "job_id": job.id,
        "dataset_id": job.dataset_id,
        "target_column": job.target_column,
        "task_type": job.task_type,
        "status": job.status,
        "best_model_name": job.best_model_name,
        "best_metric_value": job.best_metric_value,
        "results": json.loads(job.results_json) if job.results_json else [],
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else "",
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@router.get("/jobs", response_model=list[dict])
def list_training_jobs(training_service: TrainingService = Depends(get_training_service)):
    """List all recent training jobs."""
    jobs = training_service.job_repo.list_jobs()
    return [
        {
            "job_id": j.id,
            "dataset_id": j.dataset_id,
            "target_column": j.target_column,
            "task_type": j.task_type,
            "status": j.status,
            "best_model_name": j.best_model_name,
            "best_metric_value": j.best_metric_value,
            "created_at": j.created_at.isoformat() if j.created_at else "",
        }
        for j in jobs
    ]
