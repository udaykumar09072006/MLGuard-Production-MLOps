"""Unit tests for evaluation metrics calculations."""

import numpy as np

from app.ml.evaluation import evaluate_classification, evaluate_regression, pick_best_model


def test_evaluate_regression_metrics():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([105.0, 195.0, 305.0, 395.0])
    metrics = evaluate_regression(y_true, y_pred)
    assert metrics["mae"] == 5.0
    assert metrics["rmse"] == 5.0
    assert metrics["r2"] > 0.95


def test_evaluate_classification_metrics():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 0])
    y_proba = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.85, 0.15],
        [0.1, 0.9],
        [0.6, 0.4],
        [0.75, 0.25],
    ])
    metrics = evaluate_classification(y_true, y_pred, y_proba, task_type="binary_classification")
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] is not None


def test_pick_best_model_selection():
    results = [
        {"model_name": "model_A", "metrics": {"rmse": 45.0, "r2": 0.85}},
        {"model_name": "model_B", "metrics": {"rmse": 20.0, "r2": 0.95}},
    ]
    best_name, best_val = pick_best_model(results, task_type="regression", metric_objective="rmse")
    assert best_name == "model_B"
    assert best_val == 20.0
