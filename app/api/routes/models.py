"""Model Registry and serving activation endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_model_registry_service
from app.core.exceptions import ModelNotFoundError, ModelTrainingError
from app.schemas.models import ModelActivationRequest, ModelVersionSchema
from app.services.model_registry_service import ModelRegistryService

router = APIRouter(prefix="/models", tags=["Model Registry"])


@router.get("", response_model=list[ModelVersionSchema])
def list_registered_models(
    registry_service: ModelRegistryService = Depends(get_model_registry_service),
):
    """List all registered candidate models, their metrics, and deployment statuses."""
    return registry_service.list_all_models()


@router.get("/active", response_model=Optional[ModelVersionSchema])
def get_active_model(
    registry_service: ModelRegistryService = Depends(get_model_registry_service),
):
    """Retrieve the currently deployed active model version serving inference traffic."""
    active = registry_service.get_active_model()
    if not active:
        raise HTTPException(status_code=404, detail="No active model is currently deployed.")
    return active


@router.get("/{model_id}", response_model=ModelVersionSchema)
def get_model_details(
    model_id: str,
    registry_service: ModelRegistryService = Depends(get_model_registry_service),
):
    """Inspect model version metadata, holdout metrics, and input feature requirements."""
    try:
        return registry_service.get_model_by_id(model_id)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)


@router.post("/{model_id}/activate", response_model=ModelVersionSchema, status_code=status.HTTP_200_OK)
def activate_model_version(
    model_id: str,
    body: ModelActivationRequest = ModelActivationRequest(),
    registry_service: ModelRegistryService = Depends(get_model_registry_service),
):
    """Promote an eligible, validated model version to production serving with audit log."""
    try:
        return registry_service.activate_model_version(model_id, operator=body.activated_by or "system")
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
    except ModelTrainingError as e:
        raise HTTPException(status_code=400, detail=e.message)
