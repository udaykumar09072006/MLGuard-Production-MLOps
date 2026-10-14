"""Dataset validation and schema inspection engine."""


import pandas as pd

from app.core.exceptions import DatasetValidationError
from app.schemas.datasets import ColumnSummary, DatasetSummary, DatasetValidationWarning


def inspect_dataset(df: pd.DataFrame, dataset_id: str, filename: str, file_path: str, file_size: int, fingerprint: str) -> DatasetSummary:
    """Perform comprehensive data inspection, schema identification, and safety checks."""
    if df.empty:
        raise DatasetValidationError("Dataset is empty. At least 5 rows and 2 columns are required.")

    row_count, col_count = df.shape
    if row_count < 5:
        raise DatasetValidationError(f"Dataset has only {row_count} rows. Minimum 5 rows required for validation and splitting.")
    if col_count < 2:
        raise DatasetValidationError(f"Dataset has only {col_count} column. At least 1 feature and 1 target column required.")

    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        raise DatasetValidationError("Dataset contains duplicate column names.")

    columns_summary: list[ColumnSummary] = []
    warnings: list[DatasetValidationWarning] = []

    # Check duplicate rows
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        pct = (duplicate_rows / row_count) * 100
        warnings.append(DatasetValidationWarning(
            code="DUPLICATE_ROWS",
            message=f"Dataset contains {duplicate_rows} duplicate rows ({pct:.1f}%).",
            severity="warning" if pct < 20 else "error"
        ))

    numerical_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = round((missing_count / row_count) * 100, 2)
        unique_count = int(series.nunique(dropna=True))

        # Infer type
        is_numeric = pd.api.types.is_numeric_dtype(series)
        inferred_type = "numerical" if is_numeric else "categorical"

        # Check if ID column (all unique strings/numbers, or named like id)
        col_lower = str(col).lower()
        if unique_count == row_count and ("id" in col_lower or "guid" in col_lower or "index" in col_lower):
            inferred_type = "id"
            warnings.append(DatasetValidationWarning(
                code="HIGH_CARDINALITY_ID",
                message=f"Column '{col}' appears to be a unique identifier. Consider excluding it from features.",
                column=col,
                severity="warning"
            ))

        # Check constant columns
        if unique_count <= 1:
            warnings.append(DatasetValidationWarning(
                code="CONSTANT_COLUMN",
                message=f"Column '{col}' has {unique_count} unique value. It provides zero variance.",
                column=col,
                severity="warning"
            ))

        # Check missing percentage warning
        if missing_pct > 30.0:
            warnings.append(DatasetValidationWarning(
                code="HIGH_MISSING_VALUES",
                message=f"Column '{col}' has {missing_pct}% missing values.",
                column=col,
                severity="warning" if missing_pct < 60.0 else "error"
            ))

        stats: dict[str, float] | None = None
        if is_numeric and unique_count > 0:
            valid_nums = series.dropna()
            stats = {
                "mean": float(valid_nums.mean()),
                "std": float(valid_nums.std() if len(valid_nums) > 1 else 0.0),
                "min": float(valid_nums.min()),
                "p25": float(valid_nums.quantile(0.25)),
                "p50": float(valid_nums.median()),
                "p75": float(valid_nums.quantile(0.75)),
                "max": float(valid_nums.max()),
            }
            if inferred_type != "id":
                numerical_cols.append(col)
        else:
            if inferred_type != "id":
                categorical_cols.append(col)

        sample_vals = [str(x) for x in series.dropna().iloc[:5].tolist()]

        columns_summary.append(ColumnSummary(
            name=str(col),
            dtype=str(series.dtype),
            inferred_type=inferred_type,
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            sample_values=sample_vals,
            stats=stats
        ))

    # Task and Target Suggestion Heuristic
    suggested_target = None
    suggested_task = None
    last_col = df.columns[-1]

    # Target candidate priority
    for col in reversed(df.columns):
        c_lower = str(col).lower()
        if any(keyword in c_lower for keyword in ["target", "label", "price", "churn", "status", "class", "outcome"]):
            suggested_target = col
            break
    if not suggested_target:
        suggested_target = last_col

    if suggested_target:
        target_series = df[suggested_target].dropna()
        n_unique_target = target_series.nunique()
        if pd.api.types.is_float_dtype(target_series) or (pd.api.types.is_numeric_dtype(target_series) and n_unique_target > 5):
            suggested_task = "regression"
        elif n_unique_target == 2:
            suggested_task = "binary_classification"
        else:
            suggested_task = "multiclass_classification"

    return DatasetSummary(
        dataset_id=dataset_id,
        filename=filename,
        file_path=file_path,
        row_count=row_count,
        column_count=col_count,
        file_size_bytes=file_size,
        fingerprint_sha256=fingerprint,
        columns=columns_summary,
        warnings=warnings,
        suggested_target=suggested_target,
        suggested_task=suggested_task,
    )


def validate_training_inputs(
    df: pd.DataFrame, target_column: str, task_type: str, test_size: float
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate dataset split feasibility, drop unlabelled target rows, and ensure clean input."""
    if target_column not in df.columns:
        raise DatasetValidationError(f"Target column '{target_column}' does not exist in dataset.")

    # Drop rows where target is NaN
    clean_df = df.dropna(subset=[target_column]).copy()
    if len(clean_df) < 10:
        raise DatasetValidationError(f"Insufficient samples ({len(clean_df)}) after dropping missing target values.")

    y = clean_df[target_column]
    X = clean_df.drop(columns=[target_column])

    if X.shape[1] < 1:
        raise DatasetValidationError("Dataset has no predictor features remaining after removing target.")

    # Task type validations
    if task_type == "regression":
        if not pd.api.types.is_numeric_dtype(y):
            raise DatasetValidationError(f"Regression target '{target_column}' must be numeric, got {y.dtype}.")
        if y.nunique() < 3:
            raise DatasetValidationError(f"Regression target has only {y.nunique()} unique values. Choose classification instead.")
    elif task_type in ("binary_classification", "multiclass_classification"):
        unique_classes = y.nunique()
        if task_type == "binary_classification" and unique_classes != 2:
            raise DatasetValidationError(f"Binary classification requires exactly 2 distinct classes, found {unique_classes}.")
        if task_type == "multiclass_classification" and unique_classes < 3:
            raise DatasetValidationError(f"Multiclass classification requires at least 3 distinct classes, found {unique_classes}.")

        # Check min class count for stratification
        class_counts = y.value_counts()
        min_count = class_counts.min()
        if min_count < 2:
            raise DatasetValidationError(f"Class '{class_counts.idxmin()}' only has {min_count} instance. Cannot stratify.")

    return X, y
