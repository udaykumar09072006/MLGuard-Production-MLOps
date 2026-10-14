"""Online prediction serving endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_prediction_service
from app.core.exceptions import ModelNotActiveError, ModelNotFoundError, PredictionInputError
from app.schemas.predictions import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter(tags=["Predictions"])


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
):
    """Serve predictions using the active model with strict schema validation and latency logging."""
    try:
        return prediction_service.predict(request)
    except ModelNotActiveError as e:
        raise HTTPException(status_code=409, detail=e.message)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message)
    except PredictionInputError as e:
        raise HTTPException(status_code=422, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
