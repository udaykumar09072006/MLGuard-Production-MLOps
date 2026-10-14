"""Unit tests for dataset inspection and validation rules."""

import pandas as pd
import pytest

from app.core.exceptions import DatasetValidationError
from app.ml.data_validation import inspect_dataset, validate_training_inputs


def test_inspect_dataset_success():
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "category": ["A", "B", "A", "B", "A", "B"],
        "target": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    summary = inspect_dataset(df, "ds_1", "test.csv", "/tmp/test.csv", 100, "hash123")
    assert summary.row_count == 6
    assert summary.column_count == 3
    assert summary.suggested_task == "regression"
    assert summary.suggested_target == "target"


def test_inspect_dataset_empty_fails():
    empty_df = pd.DataFrame()
    with pytest.raises(DatasetValidationError, match="Dataset is empty"):
        inspect_dataset(empty_df, "ds_0", "empty.csv", "/tmp/empty.csv", 0, "hash0")


def test_inspect_dataset_single_column_fails():
    single_df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
    with pytest.raises(DatasetValidationError, match="At least 1 feature and 1 target"):
        inspect_dataset(single_df, "ds_0", "single.csv", "/tmp/single.csv", 10, "hash1")


def test_inspect_dataset_detects_constant_and_id_warnings():
    df = pd.DataFrame({
        "id_col": [f"user_{i}" for i in range(10)],
        "constant_col": [42] * 10,
        "feature": [i * 1.5 for i in range(10)],
        "target": [0, 1] * 5,
    })
    summary = inspect_dataset(df, "ds_warn", "test.csv", "/tmp/test.csv", 150, "hash")
    warning_codes = [w.code for w in summary.warnings]
    assert "HIGH_CARDINALITY_ID" in warning_codes
    assert "CONSTANT_COLUMN" in warning_codes


def test_validate_training_inputs_regression_checks():
    df = pd.DataFrame({
        "sqft": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "price": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    })
    X, y = validate_training_inputs(df, target_column="price", task_type="regression", test_size=0.2)
    assert len(X) == 10
    assert "sqft" in X.columns
    assert "price" not in X.columns


def test_validate_training_inputs_binary_classification_invalid_classes():
    df = pd.DataFrame({
        "f1": range(12),
        "target": [0, 1, 2] * 4,  # 3 classes passed to binary classification
    })
    with pytest.raises(DatasetValidationError, match="Binary classification requires exactly 2 distinct classes"):
        validate_training_inputs(df, target_column="target", task_type="binary_classification", test_size=0.2)
