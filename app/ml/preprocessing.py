"""Scikit-learn preprocessing pipeline builder using ColumnTransformer."""


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    numeric_imputer_strategy: str = "median",
    categorical_imputer_strategy: str = "most_frequent",
    scaling: str = "standard",
) -> ColumnTransformer:
    """Build a robust scikit-learn ColumnTransformer that handles numeric & categorical features

    without data leakage.
    """
    transformers = []

    if numeric_features:
        num_steps = [
            ("imputer", SimpleImputer(strategy=numeric_imputer_strategy))
        ]
        if scaling == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif scaling == "robust":
            num_steps.append(("scaler", RobustScaler()))
        # if 'none', no scaler is added

        num_pipeline = Pipeline(steps=num_steps)
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=categorical_imputer_strategy, fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def identify_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition features cleanly into numerical and categorical lists."""
    numeric_cols = []
    categorical_cols = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols
