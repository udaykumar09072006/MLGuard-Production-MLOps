"""Unit tests for Kolmogorov-Smirnov and Chi-Square drift algorithms."""

import numpy as np
import pandas as pd

from app.ml.drift_detection import compute_dataset_drift, evaluate_numerical_drift


def test_evaluate_numerical_drift_identical_distributions():
    rng = np.random.default_rng(42)
    data = rng.normal(100, 15, 200)
    ref = pd.Series(data[:100])
    curr = pd.Series(data[100:])

    report = evaluate_numerical_drift(ref, curr, "feature_1", threshold=0.05)
    assert not report.drift_detected
    assert report.p_value > 0.05


def test_evaluate_numerical_drift_shifted_distribution():
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(50, 10, 150))
    curr = pd.Series(rng.normal(90, 10, 150))  # clear mean shift

    report = evaluate_numerical_drift(ref, curr, "feature_shifted", threshold=0.05)
    assert report.drift_detected
    assert report.p_value < 0.05


def test_compute_dataset_drift_mixed_columns():
    ref_df = pd.DataFrame({
        "num": [10.0, 11.0, 12.0, 10.5, 11.5, 12.5, 10.2, 11.8, 12.1, 10.9] * 5,
        "cat": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"] * 5,
    })
    curr_df = pd.DataFrame({
        "num": [50.0, 55.0, 60.0, 52.0, 58.0, 62.0, 51.0, 59.0, 61.0, 55.0] * 5,  # drifted
        "cat": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"] * 5,
    })
    reports = compute_dataset_drift(ref_df, curr_df, features=["num", "cat"], p_value_threshold=0.05)
    assert len(reports) == 2
    num_report = next(r for r in reports if r.feature_name == "num")
    assert num_report.drift_detected
