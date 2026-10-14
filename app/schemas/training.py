"""Schemas for model training configuration and job status."""

from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    algorithm: str  # "linear_regression", "random_forest", "gradient_boosting", "logistic_regression"
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingRequest(BaseModel):
    dataset_id: str
    target_column: str
    task_type: str  # "regression", "binary_classification", "multiclass_classification"
    models: list[str] = Field(default=["random_forest", "gradient_boosting"])
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    cv_folds: int = Field(default=3, ge=2, le=10)
    random_state: int = Field(default=42)
    numeric_imputer: str = Field(default="median")  # "median", "mean"
    categorical_imputer: str = Field(default="most_frequent")
    scaling: str = Field(default="standard")  # "standard", "robust", "none"
    metric_objective: str | None = None  # e.g., "rmse", "r2", "f1", "accuracy", "roc_auc"


class ModelEvaluationResult(BaseModel):
    model_name: str
    algorithm: str
    run_id: str
    metrics: dict[str, Any]
    confusion_matrix: list[list[int]] | None = None
    cv_scores: list[float] | None = None
    cv_mean: float | None = None
    cv_std: float | None = None
    duration_seconds: float
    feature_names: list[str]
    artifact_uri: str


class TrainingJobResponse(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    task_type: str
    target_column: str
    dataset_id: str
    results: list[ModelEvaluationResult] = Field(default_factory=list)
    best_model_name: str | None = None
    best_metric_value: float | None = None
    error_message: str | None = None
    created_at: str
    completed_at: str | None = None
