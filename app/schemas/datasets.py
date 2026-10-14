"""Pydantic schemas for datasets and validation."""

from typing import Any

from pydantic import BaseModel, Field


class ColumnSummary(BaseModel):
    name: str
    dtype: str
    inferred_type: str  # "numerical", "categorical", "id", "datetime"
    missing_count: int
    missing_pct: float
    unique_count: int
    sample_values: list[Any] = Field(default_factory=list)
    stats: dict[str, float] | None = None


class DatasetValidationWarning(BaseModel):
    code: str
    message: str
    column: str | None = None
    severity: str = "warning"  # "info", "warning", "error"


class DatasetSummary(BaseModel):
    dataset_id: str
    filename: str
    file_path: str
    row_count: int
    column_count: int
    file_size_bytes: int
    fingerprint_sha256: str
    columns: list[ColumnSummary]
    warnings: list[DatasetValidationWarning]
    suggested_target: str | None = None
    suggested_task: str | None = None


class DatasetInspectRequest(BaseModel):
    dataset_id: str
    target_column: str | None = None
    task_type: str | None = None  # "regression", "binary_classification", "multiclass_classification"
