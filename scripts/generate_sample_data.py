"""Generate reproducible synthetic sample datasets for MLGuard.

Generates:
1. housing_prices.csv - Tabular regression problem with numeric & categorical features
2. customer_churn.csv - Tabular binary classification problem with realistic class imbalance
"""

import os

import numpy as np
import pandas as pd


def generate_housing_data(n_samples: int = 500, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    square_feet = rng.normal(2100, 600, n_samples).clip(750, 4800).round()
    bedrooms = rng.integers(1, 6, size=n_samples)
    bathrooms = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], size=n_samples)
    year_built = rng.integers(1960, 2024, size=n_samples)
    neighborhoods = ["Downtown", "Suburbs", "Uptown", "Lakeside"]
    neighborhood = rng.choice(neighborhoods, size=n_samples, p=[0.25, 0.35, 0.20, 0.20])
    property_types = ["Single Family", "Condo", "Townhouse"]
    property_type = rng.choice(property_types, size=n_samples, p=[0.6, 0.25, 0.15])
    garage_spaces = rng.integers(0, 4, size=n_samples)

    # Base price calculation with noise
    neighborhood_multiplier = {
        "Downtown": 1.35,
        "Suburbs": 1.0,
        "Uptown": 1.25,
        "Lakeside": 1.45,
    }
    type_multiplier = {
        "Single Family": 1.1,
        "Condo": 0.9,
        "Townhouse": 1.0,
    }

    base_price = (
        120000
        + square_feet * 135
        + bedrooms * 14000
        + bathrooms * 18000
        + (year_built - 1960) * 1100
        + garage_spaces * 12500
    )

    multipliers = np.array([neighborhood_multiplier[n] * type_multiplier[t] for n, t in zip(neighborhood, property_type)])
    noise = rng.normal(0, 25000, n_samples)
    sale_price = (base_price * multipliers + noise).clip(110000, 950000).round(-2)

    df = pd.DataFrame({
        "square_feet": square_feet,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "neighborhood": neighborhood,
        "property_type": property_type,
        "garage_spaces": garage_spaces,
        "sale_price": sale_price.astype(int),
    })

    # Introduce realistic missing values (< 3%) to test imputation pipeline
    missing_indices = rng.choice(n_samples, size=int(n_samples * 0.025), replace=False)
    df.loc[missing_indices, "garage_spaces"] = np.nan

    return df


def generate_churn_data(n_samples: int = 600, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    tenure_months = rng.integers(1, 72, size=n_samples)
    contract_types = ["Month-to-month", "One year", "Two year"]
    contract_type = rng.choice(contract_types, size=n_samples, p=[0.55, 0.25, 0.20])
    internet_services = ["DSL", "Fiber optic", "No"]
    internet_service = rng.choice(internet_services, size=n_samples, p=[0.35, 0.45, 0.20])
    payment_methods = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    payment_method = rng.choice(payment_methods, size=n_samples, p=[0.35, 0.20, 0.25, 0.20])
    support_calls = rng.integers(0, 7, size=n_samples)

    # Monthly charges based on service type
    base_charge = np.where(internet_service == "Fiber optic", 85.0,
                  np.where(internet_service == "DSL", 50.0, 25.0))
    monthly_charges = (base_charge + rng.normal(0, 10, n_samples)).clip(20.0, 120.0).round(2)
    total_charges = (monthly_charges * tenure_months + rng.normal(0, 50, n_samples)).clip(20.0, 8500.0).round(2)

    # Churn probability logit
    contract_weight = {"Month-to-month": 1.2, "One year": -0.5, "Two year": -1.4}
    c_weights = np.array([contract_weight[c] for c in contract_type])

    z = (
        -1.5
        - 0.04 * tenure_months
        + 0.015 * monthly_charges
        + 0.45 * support_calls
        + c_weights
        + (1.0 if internet_service == "Fiber optic" else 0.0)
    )
    p_churn = 1 / (1 + np.exp(-z))
    churned = (rng.uniform(0, 1, n_samples) < p_churn).astype(int)

    df = pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "support_calls": support_calls,
        "churned": churned,
    })

    # Add realistic minor missing values
    missing_indices = rng.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    df.loc[missing_indices, "total_charges"] = np.nan

    return df


def main():
    target_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
    os.makedirs(target_dir, exist_ok=True)

    housing_path = os.path.join(target_dir, "housing_prices.csv")
    churn_path = os.path.join(target_dir, "customer_churn.csv")

    housing_df = generate_housing_data()
    housing_df.to_csv(housing_path, index=False)
    print(f"Generated housing regression dataset: {housing_path} ({len(housing_df)} rows)")

    churn_df = generate_churn_data()
    churn_df.to_csv(churn_path, index=False)
    print(f"Generated customer churn dataset: {churn_path} ({len(churn_df)} rows, churn rate: {churn_df['churned'].mean():.2%})")


if __name__ == "__main__":
    main()
