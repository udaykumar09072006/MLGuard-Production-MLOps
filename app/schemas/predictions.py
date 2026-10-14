"""Schemas for online model inference."""

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    model_id: str | None = None  # None uses currently active model
    features: list[dict[str, Any]] = Field(..., description="List of feature records")


class PredictionItemResult(BaseModel):
    prediction: Any
    probabilities: dict[str, float] | None = None


class PredictionResponse(BaseModel):
    request_id: str
    model_id: str
    model_version: int
    task_type: str
    predictions: list[PredictionItemResult]
    latency_ms: float
    timestamp: str
