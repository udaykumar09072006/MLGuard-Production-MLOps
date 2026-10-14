"""Model registry service managing versions, eligibility checks, and active deployments."""

import json

from sqlalchemy.orm import Session

from app.core.exceptions import ModelNotFoundError, ModelTrainingError
from app.db.models import ModelRegistryRecord
from app.db.repositories import ModelRegistryRepository
from app.schemas.models import FeatureSchema, ModelVersionSchema


class ModelRegistryService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = ModelRegistryRepository(db)

    def register_candidate_model(
        self,
        name: str,
        algorithm: str,
        task_type: str,
        target_column: str,
        run_id: str,
        metrics: dict,
        feature_schemas: list,
        artifact_path: str,
    ) -> ModelRegistryRecord:
        """Register trained candidate model with version bump."""
        from app.utils.identifiers import generate_unique_id
        model_id = generate_unique_id("mod")

        return self.repo.register_model(
            model_id=model_id,
            name=name,
            algorithm=algorithm,
            task_type=task_type,
            target_column=target_column,
            run_id=run_id,
            metrics=metrics,
            feature_schemas=feature_schemas,
            artifact_path=artifact_path,
        )

    def activate_model_version(self, model_id: str, operator: str = "system") -> ModelVersionSchema:
        """Activate a model version for production inference serving."""
        record = self.repo.get_model(model_id)
        if not record:
            raise ModelNotFoundError(f"Model ID '{model_id}' does not exist in registry.")

        if record.validation_status != "validated":
            raise ModelTrainingError(
                f"Model '{model_id}' status is '{record.validation_status}'. Only 'validated' models can be activated."
            )

        updated = self.repo.activate_model(model_id, activated_by=operator)
        return self._to_schema(updated)

    def get_active_model(self) -> ModelVersionSchema | None:
        record = self.repo.get_active_model()
        return self._to_schema(record) if record else None

    def get_model_by_id(self, model_id: str) -> ModelVersionSchema:
        record = self.repo.get_model(model_id)
        if not record:
            raise ModelNotFoundError(f"Model '{model_id}' not found.")
        return self._to_schema(record)

    def list_all_models(self) -> list[ModelVersionSchema]:
        records = self.repo.list_models()
        return [self._to_schema(r) for r in records]

    def _to_schema(self, record: ModelRegistryRecord) -> ModelVersionSchema:
        metrics_dict = json.loads(record.metrics_json) if record.metrics_json else {}
        feats = json.loads(record.feature_schema_json) if record.feature_schema_json else []
        feature_schemas = [FeatureSchema(**f) for f in feats]

        return ModelVersionSchema(
            model_id=record.id,
            version=record.version,
            name=record.name,
            task_type=record.task_type,
            target_column=record.target_column,
            algorithm=record.algorithm,
            run_id=record.run_id,
            metrics=metrics_dict,
            feature_schema=feature_schemas,
            is_active=record.is_active,
            validation_status=record.validation_status,
            created_at=record.created_at.isoformat() if record.created_at else "",
            activated_at=record.activated_at.isoformat() if record.activated_at else None,
            activated_by=record.activated_by,
            artifact_path=record.artifact_path,
        )
