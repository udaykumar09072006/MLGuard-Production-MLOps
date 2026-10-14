"""Model factory instantiating regression and classification estimators with bounded hyperparams."""

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

from app.core.exceptions import ModelTrainingError

REGRESSION_MODELS = {
    "linear_regression": LinearRegression,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}

CLASSIFICATION_MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def get_model(algorithm: str, task_type: str, random_state: int = 42, custom_params: dict[str, Any] = None) -> BaseEstimator:
    """Instantiate a validated estimator configured with deterministic seeds."""
    params = custom_params or {}
    algo_key = algorithm.lower().strip()

    if task_type == "regression":
        if algo_key not in REGRESSION_MODELS:
            raise ModelTrainingError(f"Unsupported regression algorithm '{algo_key}'. Options: {list(REGRESSION_MODELS.keys())}")

        if algo_key == "linear_regression":
            return LinearRegression(**params)
        elif algo_key == "random_forest":
            defaults = {"n_estimators": 100, "max_depth": 10, "random_state": random_state}
            defaults.update(params)
            return RandomForestRegressor(**defaults)
        elif algo_key == "gradient_boosting":
            defaults = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "random_state": random_state}
            defaults.update(params)
            return GradientBoostingRegressor(**defaults)

    elif task_type in ("binary_classification", "multiclass_classification"):
        if algo_key not in CLASSIFICATION_MODELS:
            raise ModelTrainingError(f"Unsupported classification algorithm '{algo_key}'. Options: {list(CLASSIFICATION_MODELS.keys())}")

        if algo_key == "logistic_regression":
            defaults = {"max_iter": 1000, "random_state": random_state}
            defaults.update(params)
            return LogisticRegression(**defaults)
        elif algo_key == "random_forest":
            defaults = {"n_estimators": 100, "max_depth": 10, "random_state": random_state}
            defaults.update(params)
            return RandomForestClassifier(**defaults)
        elif algo_key == "gradient_boosting":
            defaults = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "random_state": random_state}
            defaults.update(params)
            return GradientBoostingClassifier(**defaults)

    else:
        raise ModelTrainingError(f"Invalid task_type: '{task_type}'.")
