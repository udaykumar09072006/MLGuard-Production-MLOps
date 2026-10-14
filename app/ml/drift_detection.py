"""Data drift detection engine comparing incoming production features against training baseline.

Implements:
1. Kolmogorov-Smirnov (KS) two-sample test for continuous numerical features.
   Limitation: High sensitivity on huge sample sizes (tiny differences become p < 0.05).
2. Chi-Square goodness-of-fit test / frequency distribution distance for categorical features.
   Limitation: Requires expected counts >= 5 per category, fallback to Total Variation Distance if violated.
"""

import numpy as np
import pandas as pd
from scipy import stats

from app.schemas.monitoring import FeatureDriftReport


def evaluate_numerical_drift(
    ref_series: pd.Series,
    curr_series: pd.Series,
    feature_name: str,
    threshold: float = 0.05,
) -> FeatureDriftReport:
    """Compute 2-sample Kolmogorov-Smirnov test on continuous feature."""
    ref_clean = ref_series.dropna().to_numpy()
    curr_clean = curr_series.dropna().to_numpy()

    if len(ref_clean) < 5 or len(curr_clean) < 5:
        return FeatureDriftReport(
            feature_name=feature_name,
            feature_type="numerical",
            test_method="Kolmogorov-Smirnov",
            statistic=0.0,
            p_value=1.0,
            threshold=threshold,
            drift_detected=False,
            reference_summary={"count": len(ref_clean), "mean": float(np.mean(ref_clean)) if len(ref_clean) else 0.0},
            current_summary={"count": len(curr_clean), "mean": float(np.mean(curr_clean)) if len(curr_clean) else 0.0},
        )

    ks_res = stats.ks_2samp(ref_clean, curr_clean)
    statistic = float(ks_res.statistic)
    p_value = float(ks_res.pvalue)
    drift_detected = bool(p_value < threshold)

    return FeatureDriftReport(
        feature_name=feature_name,
        feature_type="numerical",
        test_method="Kolmogorov-Smirnov",
        statistic=round(statistic, 4),
        p_value=round(p_value, 6),
        threshold=threshold,
        drift_detected=drift_detected,
        reference_summary={
            "count": len(ref_clean),
            "mean": round(float(np.mean(ref_clean)), 2),
            "std": round(float(np.std(ref_clean)), 2),
            "median": round(float(np.median(ref_clean)), 2),
            "min": round(float(np.min(ref_clean)), 2),
            "max": round(float(np.max(ref_clean)), 2),
        },
        current_summary={
            "count": len(curr_clean),
            "mean": round(float(np.mean(curr_clean)), 2),
            "std": round(float(np.std(curr_clean)), 2),
            "median": round(float(np.median(curr_clean)), 2),
            "min": round(float(np.min(curr_clean)), 2),
            "max": round(float(np.max(curr_clean)), 2),
        },
    )


def evaluate_categorical_drift(
    ref_series: pd.Series,
    curr_series: pd.Series,
    feature_name: str,
    threshold: float = 0.05,
) -> FeatureDriftReport:
    """Compute Chi-Square or Total Variation Distance (TVD) for categorical distributions."""
    ref_counts = ref_series.dropna().value_counts(normalize=True)
    curr_counts = curr_series.dropna().value_counts(normalize=True)

    all_categories = sorted(list(set(ref_counts.index).union(set(curr_counts.index))))
    ref_probs = np.array([ref_counts.get(cat, 1e-5) for cat in all_categories])
    curr_probs = np.array([curr_counts.get(cat, 1e-5) for cat in all_categories])

    # Total Variation Distance: TVD = 0.5 * sum(|p - q|)
    tvd = float(0.5 * np.sum(np.abs(ref_probs - curr_probs)))

    # Scaled Chi-square test on expected frequency counts
    n_curr = len(curr_series.dropna())
    observed = np.array([curr_series.dropna().tolist().count(cat) for cat in all_categories])
    expected = ref_probs * n_curr

    # If expected counts are sufficient, run chisquare
    if np.all(expected >= 1.0) and len(all_categories) > 1:
        try:
            res = stats.chisquare(f_obs=observed, f_exp=expected)
            statistic = float(res.statistic)
            p_value = float(res.pvalue)
            drift_detected = bool(p_value < threshold)
            test_method = "Chi-Square"
        except Exception:
            test_method = "Total Variation Distance"
            statistic = tvd
            p_value = 1.0 - tvd
            drift_detected = bool(tvd > 0.20)
    else:
        test_method = "Total Variation Distance"
        statistic = tvd
        p_value = round(max(0.0, 1.0 - tvd), 4)
        drift_detected = bool(tvd > 0.20)

    return FeatureDriftReport(
        feature_name=feature_name,
        feature_type="categorical",
        test_method=test_method,
        statistic=round(statistic, 4),
        p_value=round(p_value, 6),
        threshold=threshold,
        drift_detected=drift_detected,
        reference_summary={cat: round(float(p), 3) for cat, p in zip(all_categories, ref_probs)},
        current_summary={cat: round(float(p), 3) for cat, p in zip(all_categories, curr_probs)},
    )


def compute_dataset_drift(
    ref_df: pd.DataFrame,
    curr_df: pd.DataFrame,
    features: list[str],
    p_value_threshold: float = 0.05,
) -> list[FeatureDriftReport]:
    """Execute drift detection across all tracked features."""
    reports: list[FeatureDriftReport] = []

    for feat in features:
        if feat not in ref_df.columns or feat not in curr_df.columns:
            continue

        ref_col = ref_df[feat]
        curr_col = curr_df[feat]

        if pd.api.types.is_numeric_dtype(ref_col):
            report = evaluate_numerical_drift(ref_col, curr_col, feat, threshold=p_value_threshold)
        else:
            report = evaluate_categorical_drift(ref_col, curr_col, feat, threshold=p_value_threshold)

        reports.append(report)

    return reports
