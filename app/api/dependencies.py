"""FastAPI dependency injection providers."""

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.dataset_service import DatasetService
from app.services.drift_service import DriftService
from app.services.experiment_service import ExperimentService
from app.services.model_registry_service import ModelRegistryService
from app.services.monitoring_service import MonitoringService
from app.services.prediction_service import PredictionService
from app.services.training_service import TrainingService


def get_dataset_service(db: Session = Depends(get_db)) -> DatasetService:
    return DatasetService(db)


def get_training_service(db: Session = Depends(get_db)) -> TrainingService:
    return TrainingService(db)


def get_experiment_service() -> ExperimentService:
    return ExperimentService()


def get_model_registry_service(db: Session = Depends(get_db)) -> ModelRegistryService:
    return ModelRegistryService(db)


def get_prediction_service(db: Session = Depends(get_db)) -> PredictionService:
    return PredictionService(db)


def get_drift_service(db: Session = Depends(get_db)) -> DriftService:
    return DriftService(db)


def get_monitoring_service(db: Session = Depends(get_db)) -> MonitoringService:
    return MonitoringService(db)
