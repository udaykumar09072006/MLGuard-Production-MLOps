"""Seed demo script demonstrating full end-to-end MLGuard workflows:

1. Loads offline sample datasets (housing prices & customer churn).
2. Trains regression algorithms (Linear Regression, Random Forest, Gradient Boosting).
3. Trains classification algorithms (Logistic Regression, Random Forest, Gradient Boosting).
4. Registers and validates models into Model Registry.
5. Activates top model for inference serving.
6. Runs real test predictions through PredictionService.
7. Simulates production traffic with slight drift to demonstrate drift detection.
"""

import os
import sys

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.db.database import Base, SessionLocal, engine
from app.schemas.predictions import PredictionRequest
from app.schemas.training import TrainingRequest
from app.services.dataset_service import DatasetService
from app.services.drift_service import DriftService
from app.services.model_registry_service import ModelRegistryService
from app.services.monitoring_service import MonitoringService
from app.services.prediction_service import PredictionService
from app.services.training_service import TrainingService


def seed_demo():
    print("=" * 70)
    print("MLGuard Demo Seeding: Initializing Database & Pipelines")
    print("=" * 70)

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    dataset_service = DatasetService(db)
    training_service = TrainingService(db)
    registry_service = ModelRegistryService(db)
    prediction_service = PredictionService(db)
    drift_service = DriftService(db)
    monitoring_service = MonitoringService(db)

    # 1. Ingest Housing Prices (Regression)
    housing_csv = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "housing_prices.csv")
    with open(housing_csv, "rb") as f:
        housing_summary = dataset_service.process_and_save_csv("housing_prices.csv", f.read())
    print(f"[OK] Ingested Housing Dataset: {housing_summary.dataset_id} ({housing_summary.row_count} rows)")

    # 2. Ingest Customer Churn (Classification)
    churn_csv = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "customer_churn.csv")
    with open(churn_csv, "rb") as f:
        churn_summary = dataset_service.process_and_save_csv("customer_churn.csv", f.read())
    print(f"[OK] Ingested Churn Dataset: {churn_summary.dataset_id} ({churn_summary.row_count} rows)")

    # 3. Train Regression Models
    print("\n--- Training Regression Models ---")
    reg_req = TrainingRequest(
        dataset_id=housing_summary.dataset_id,
        target_column="sale_price",
        task_type="regression",
        models=["linear_regression", "random_forest", "gradient_boosting"],
        test_size=0.2,
        cv_folds=3,
        random_state=42,
    )
    reg_job = training_service.run_training_job(reg_req)
    print(f"[OK] Regression Job Finished: Best Model = {reg_job.best_model_name}")
    for res in reg_job.results:
        print(f"     -> {res.algorithm.upper()}: RMSE={res.metrics.get('rmse')}, R2={res.metrics.get('r2')}")

    # 4. Train Classification Models
    print("\n--- Training Classification Models ---")
    clf_req = TrainingRequest(
        dataset_id=churn_summary.dataset_id,
        target_column="churned",
        task_type="binary_classification",
        models=["logistic_regression", "random_forest", "gradient_boosting"],
        test_size=0.2,
        cv_folds=3,
        random_state=42,
    )
    clf_job = training_service.run_training_job(clf_req)
    print(f"[OK] Classification Job Finished: Best Model = {clf_job.best_model_name}")
    for res in clf_job.results:
        print(f"     -> {res.algorithm.upper()}: Accuracy={res.metrics.get('accuracy')}, F1={res.metrics.get('f1')}")

    # 5. Verify Active Model
    active_model = registry_service.get_active_model()
    print(f"\n[OK] Currently Active Serving Model: {active_model.name} (v{active_model.version}) [{active_model.task_type}]")

    # 6. Run Sample Predictions
    print("\n--- Executing Inference Predictions ---")
    sample_records = [
        {
            "tenure_months": 3,
            "monthly_charges": 89.5,
            "total_charges": 268.5,
            "contract_type": "Month-to-month",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
            "support_calls": 4,
        },
        {
            "tenure_months": 48,
            "monthly_charges": 25.0,
            "total_charges": 1200.0,
            "contract_type": "Two year",
            "internet_service": "No",
            "payment_method": "Credit card",
            "support_calls": 0,
        },
    ]
    pred_res = prediction_service.predict(PredictionRequest(features=sample_records))
    print(f"[OK] Prediction Output for {len(pred_res.predictions)} items (Latency: {pred_res.latency_ms}ms):")
    for i, p in enumerate(pred_res.predictions):
        print(f"     Record {i+1}: Prediction={p.prediction}, Probabilities={p.probabilities}")

    # 7. Generate Simulated Production Traffic to Test Drift Detection
    print("\n--- Generating Simulated Production Traffic for Drift Monitoring ---")
    # Generate 25 records with slight shifted distribution (e.g. higher charges, more support calls)
    import random
    traffic = []
    for _ in range(25):
        traffic.append({
            "tenure_months": random.randint(1, 15),
            "monthly_charges": random.uniform(95.0, 130.0),  # shifted higher
            "total_charges": random.uniform(200.0, 1500.0),
            "contract_type": "Month-to-month",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
            "support_calls": random.randint(3, 7),           # shifted higher
        })
    prediction_service.predict(PredictionRequest(features=traffic))

    # Run Drift Analysis
    drift_report = drift_service.calculate_drift(active_model.model_id)
    print(f"[OK] Drift Analysis Complete: Evaluated {drift_report.features_analyzed} features on {drift_report.sample_size} production records.")
    print(f"     Overall Drift Detected: {drift_report.overall_drift_detected} (Drifted features: {drift_report.drifted_features_count})")
    for r in drift_report.feature_reports:
        flag = "DRIFT" if r.drift_detected else "STABLE"
        print(f"     [{flag}] {r.feature_name} (p-val: {r.p_value:.4f}, stat: {r.statistic})")

    # 8. Check Alerts
    alerts = monitoring_service.list_alerts()
    print(f"\n[OK] Monitoring Alerts Count: {len(alerts)}")
    for a in alerts:
        print(f"     Alert [{a.severity}]: {a.message} (Status: {a.status})")

    db.close()
    print("\n" + "=" * 70)
    print("MLGuard Demo Seeding Complete! All pipelines verified.")
    print("=" * 70)


if __name__ == "__main__":
    seed_demo()
