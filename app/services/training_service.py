"""Training orchestration service tying dataset, ML pipeline, MLflow, and DB."""

from sqlalchemy.orm import Session

from app.core.logging_config import logger
from app.db.repositories import TrainingJobRepository
from app.ml.training import train_models_pipeline
from app.schemas.training import TrainingJobResponse, TrainingRequest
from app.services.dataset_service import DatasetService
from app.services.experiment_service import ExperimentService
from app.services.model_registry_service import ModelRegistryService
from app.utils.identifiers import generate_unique_id


class TrainingService:
    def __init__(self, db: Session):
        self.db = db
        self.dataset_service = DatasetService(db)
        self.job_repo = TrainingJobRepository(db)
        self.registry_service = ModelRegistryService(db)
        self.experiment_service = ExperimentService()

    def run_training_job(self, request: TrainingRequest) -> TrainingJobResponse:
        job_id = generate_unique_id("job")
        self.job_repo.create_job(
            job_id=job_id,
            dataset_id=request.dataset_id,
            target=request.target_column,
            task=request.task_type,
        )

        try:
            df = self.dataset_service.load_dataset_dataframe(request.dataset_id)
            response = train_models_pipeline(df, request, job_id)

            # Persist evaluation results and best model to DB
            self.job_repo.update_job_completed(response)

            # Register models and log to MLflow
            for res in response.results:
                # Log to MLflow
                self.experiment_service.log_run(
                    run_name=res.model_name,
                    params={
                        "algorithm": res.algorithm,
                        "task_type": request.task_type,
                        "target_column": request.target_column,
                        "test_size": request.test_size,
                        "cv_folds": request.cv_folds,
                        "scaling": request.scaling,
                    },
                    metrics=res.metrics,
                    tags={"dataset_id": request.dataset_id, "job_id": job_id},
                    artifact_path=res.artifact_uri,
                )

                # Register each trained model candidate into Model Registry
                # Load feature schemas saved in the artifact
                import joblib
                artifact_content = joblib.load(res.artifact_uri)
                feature_schemas = artifact_content.get("feature_schemas", [])

                reg_record = self.registry_service.register_candidate_model(
                    name=res.algorithm,
                    algorithm=res.algorithm,
                    task_type=request.task_type,
                    target_column=request.target_column,
                    run_id=res.run_id,
                    metrics=res.metrics,
                    feature_schemas=feature_schemas,
                    artifact_path=res.artifact_uri,
                )

                # If this model was picked as the best model, activate it automatically for serving
                if res.model_name == response.best_model_name:
                    self.registry_service.activate_model_version(reg_record.id, operator="auto_train_job")
                    logger.info(f"Auto-activated best model: {reg_record.id} ({res.model_name})")

            return response

        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
            failed_resp = TrainingJobResponse(
                job_id=job_id,
                status="failed",
                task_type=request.task_type,
                target_column=request.target_column,
                dataset_id=request.dataset_id,
                error_message=str(e),
                created_at="",
            )
            self.job_repo.update_job_completed(failed_resp)
            raise
