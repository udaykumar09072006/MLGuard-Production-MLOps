"""Application configuration settings for MLGuard using Pydantic Settings."""

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    APP_PORT: int = 8000
    DASHBOARD_PORT: int = 8501

    # Storage Paths
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_STORAGE_DIR: str = os.path.join(BASE_DIR, "data", "uploads")
    MODEL_STORAGE_DIR: str = os.path.join(BASE_DIR, "artifacts", "models")
    REFERENCE_DATA_DIR: str = os.path.join(BASE_DIR, "artifacts", "reference")
    SAMPLE_DATA_DIR: str = os.path.join(BASE_DIR, "data", "sample")

    # Database
    DATABASE_URL: str = "sqlite:///./artifacts/mlguard.db"

    # MLflow Tracking
    MLFLOW_TRACKING_URI: str = "file:./artifacts/mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "MLGuard_Experiments"

    # Monitoring Thresholds
    DRIFT_P_VALUE_THRESHOLD: float = 0.05
    LATENCY_ALERT_THRESHOLD_MS: float = 250.0
    ERROR_RATE_ALERT_THRESHOLD: float = 0.10
    MIN_DRIFT_SAMPLE_SIZE: int = 15

    # File Upload limits
    MAX_UPLOAD_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: list[str] = [".csv"]

    # API
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8501", "*"]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure directories exist
for path in [
    settings.DATA_STORAGE_DIR,
    settings.MODEL_STORAGE_DIR,
    settings.REFERENCE_DATA_DIR,
    settings.SAMPLE_DATA_DIR,
    os.path.join(settings.BASE_DIR, "artifacts"),
]:
    os.makedirs(path, exist_ok=True)
