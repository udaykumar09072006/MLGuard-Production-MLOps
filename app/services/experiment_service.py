"""MLflow tracking integration service."""

import os
from typing import Any

import mlflow

from app.core.config import settings
from app.core.logging_config import logger


class ExperimentService:
    def __init__(self):
        self.tracking_uri = settings.MLFLOW_TRACKING_URI
        self.experiment_name = settings.MLFLOW_EXPERIMENT_NAME
        self._init_mlflow()

    def _init_mlflow(self):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow initialized with tracking URI: {self.tracking_uri}")
        except Exception as e:
            logger.warning(f"Could not connect to MLflow backend ({e}). Local fallback mode active.")

    def log_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str],
        artifact_path: str | None = None,
    ) -> str | None:
        """Log parameters, metrics, tags, and pipeline artifact to MLflow."""
        try:
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id

                # Log params (flattened)
                for k, v in params.items():
                    mlflow.log_param(k, str(v)[:250])

                # Log scalar metrics
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and v is not None:
                        mlflow.log_metric(k, float(v))

                # Log tags
                for k, v in tags.items():
                    mlflow.set_tag(k, str(v))

                # Log model artifact if present
                if artifact_path and os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path, artifact_path="model")

                logger.info(f"MLflow run {run_id} logged successfully for {run_name}")
                return run_id
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
            return None

    def list_recent_runs(self, max_results: int = 20) -> list:
        try:
            exp = mlflow.get_experiment_by_name(self.experiment_name)
            if not exp:
                return []
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=max_results)
            return runs.to_dict(orient="records") if not runs.empty else []
        except Exception as e:
            logger.warning(f"Failed to fetch runs from MLflow: {e}")
            return []
