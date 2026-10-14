"""Comprehensive evaluation metrics for regression and classification."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics on test holdout set."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    task_type: str = "binary_classification",
) -> dict[str, Any]:
    """Compute classification metrics, confusion matrix, ROC-AUC and PR-AUC.

    Uses weighted averaging for multiclass to account for class imbalance fairly.
    """
    average_mode = "binary" if task_type == "binary_classification" else "weighted"

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average=average_mode, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average=average_mode, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=average_mode, zero_division=0))

    metrics: dict[str, Any] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # ROC-AUC & PR-AUC calculation where probabilities exist
    if y_proba is not None:
        try:
            if task_type == "binary_classification":
                # y_proba shape is (N, 2), take positive class probability
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    p_pos = y_proba[:, 1]
                    metrics["roc_auc"] = round(float(roc_auc_score(y_true, p_pos)), 4)
                    metrics["pr_auc"] = round(float(average_precision_score(y_true, p_pos)), 4)
                elif y_proba.ndim == 1:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
                    metrics["pr_auc"] = round(float(average_precision_score(y_true, y_proba)), 4)
            elif task_type == "multiclass_classification":
                metrics["roc_auc"] = round(
                    float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")), 4
                )
        except Exception:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def pick_best_model(
    results: list[dict[str, Any]], task_type: str, metric_objective: str | None = None
) -> tuple[str | None, float | None]:
    """Select best performing model based on configured objective metric."""
    if not results:
        return None, None

    if task_type == "regression":
        objective = (metric_objective or "rmse").lower()
        # For rmse and mae, lower is better. For r2, higher is better.
        if objective in ("rmse", "mae"):
            best = min(results, key=lambda r: r["metrics"].get(objective, float("inf")))
        else:
            best = max(results, key=lambda r: r["metrics"].get(objective, float("-inf")))
    else:
        # Default classification objective is f1 score
        objective = (metric_objective or "f1").lower()
        best = max(results, key=lambda r: r["metrics"].get(objective, float("-inf")))

    return best["model_name"], best["metrics"].get(objective)
