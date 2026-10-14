"""Test data fixtures providing small synthetic dataframes."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame({
        "sqft": rng.normal(2000, 400, n).clip(1000, 3500),
        "beds": rng.integers(1, 5, size=n),
        "city": rng.choice(["East", "West", "North"], size=n),
        "price": rng.normal(300000, 50000, n).clip(150000, 500000),
    })


@pytest.fixture
def sample_classification_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 80
    return pd.DataFrame({
        "age": rng.integers(20, 70, size=n),
        "balance": rng.normal(5000, 1500, n).clip(500, 15000),
        "tier": rng.choice(["Bronze", "Silver", "Gold"], size=n),
        "target": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
    })
