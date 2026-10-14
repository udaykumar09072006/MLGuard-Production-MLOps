"""Unit tests for preprocessing transformers and pipeline isolation."""

import numpy as np
import pandas as pd

from app.ml.preprocessing import build_preprocessor, identify_feature_types


def test_identify_feature_types():
    df = pd.DataFrame({
        "num1": [1.0, 2.5, 3.8],
        "num2": [10, 20, 30],
        "cat1": ["A", "B", "C"],
        "cat2": ["yes", "no", "yes"],
    })
    nums, cats = identify_feature_types(df)
    assert set(nums) == {"num1", "num2"}
    assert set(cats) == {"cat1", "cat2"}


def test_preprocessor_handles_missing_values_and_unseen_categories():
    train_df = pd.DataFrame({
        "num": [10.0, np.nan, 30.0, 40.0],
        "cat": ["low", "med", "high", np.nan],
    })
    test_df = pd.DataFrame({
        "num": [25.0, np.nan],
        "cat": ["unseen_category", "low"],  # test unseen categorical handling
    })

    preprocessor = build_preprocessor(
        numeric_features=["num"],
        categorical_features=["cat"],
        numeric_imputer_strategy="median",
        categorical_imputer_strategy="most_frequent",
        scaling="standard",
    )

    # Fit strictly on train
    X_train_trans = preprocessor.fit_transform(train_df)
    assert X_train_trans.shape[0] == 4
    assert not np.isnan(X_train_trans).any()

    # Transform test without refitting
    X_test_trans = preprocessor.transform(test_df)
    assert X_test_trans.shape[0] == 2
    assert not np.isnan(X_test_trans).any()
