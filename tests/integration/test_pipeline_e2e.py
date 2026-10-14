"""End-to-end integration test: Upload -> Train -> Register -> Predict -> Drift -> Monitor."""

import io

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_complete_mlguard_lifecycle_e2e():
    # 1. Ingest dataset
    csv_rows = ["x1,x2,category,label"]
    for i in range(40):
        c = "A" if i % 2 == 0 else "B"
        label = 1 if i > 20 else 0
        csv_rows.append(f"{i * 1.5},{i * 2.0},{c},{label}")
    csv_bytes = "\n".join(csv_rows).encode("utf-8")

    files = {"file": ("e2e_data.csv", io.BytesIO(csv_bytes), "text/csv")}
    upload_res = client.post("/api/v1/datasets/upload", files=files)
    assert upload_res.status_code == 201
    dataset_id = upload_res.json()["dataset_id"]

    # 2. Train models
    train_payload = {
        "dataset_id": dataset_id,
        "target_column": "label",
        "task_type": "binary_classification",
        "models": ["logistic_regression"],
        "test_size": 0.2,
        "cv_folds": 2,
        "random_state": 42,
    }
    train_res = client.post("/api/v1/training/jobs", json=train_payload)
    assert train_res.status_code == 201
    job_data = train_res.json()
    assert job_data["status"] == "completed"
    assert len(job_data["results"]) >= 1

    # 3. Model Registry - check active model
    active_res = client.get("/api/v1/models/active")
    assert active_res.status_code == 200
    active_model = active_res.json()
    assert active_model["is_active"] is True
    model_id = active_model["model_id"]

    # 4. Predict
    pred_payload = {
        "model_id": model_id,
        "features": [
            {"x1": 5.0, "x2": 8.0, "category": "A"},
            {"x1": 45.0, "x2": 60.0, "category": "B"},
        ],
    }
    pred_res = client.post("/api/v1/predict", json=pred_payload)
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert len(pred_data["predictions"]) == 2
    assert pred_data["latency_ms"] > 0

    # 5. Monitoring Summary
    mon_res = client.get("/api/v1/monitoring/summary")
    assert mon_res.status_code == 200
    assert mon_res.json()["total_requests"] >= 1
