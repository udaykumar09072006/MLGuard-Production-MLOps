"""Schemas for model registry and version management."""

from typing import Any

from pydantic import BaseModel, Field


class FeatureSchema(BaseModel):
    name: str
    dtype: str
    inferred_type: str  # "numerical", "categorical"
    required: bool = True
    allowed_categories: list[str] | None = None
    min_val: float | None = None
    max_val: float | None = None
    default_val: Any | None = None


class ModelVersionSchema(BaseModel):
    model_id: str
    version: int
    name: str
    task_type: str
    target_column: str
    algorithm: str
    run_id: str
    metrics: dict[str, Any]
    feature_schema: list[FeatureSchema]
    is_active: bool = False
    validation_status: str = "validated"  # "validated", "rejected", "provisional"
    created_at: str
    activated_at: str | None = None
    activated_by: str | None = None
    artifact_path: str


class ModelActivationRequest(BaseModel):
    activated_by: str | None = Field(default="system_operator")
    reason: str | None = Field(default="Manual promotion to active serving")
