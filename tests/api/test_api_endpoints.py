"""API integration tests using FastAPI TestClient."""

import io

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health_and_readiness_endpoints():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json()["status"] in ("ready", "degraded")


def test_upload_dataset_endpoint():
    csv_content = b"feature1,feature2,target\n1,2,10\n3,4,20\n5,6,30\n7,8,40\n9,10,50\n11,12,60\n"
    files = {"file": ("test_upload.csv", io.BytesIO(csv_content), "text/csv")}
    res = client.post("/api/v1/datasets/upload", files=files)
    assert res.status_code == 201
    data = res.json()
    assert data["row_count"] == 6
    assert data["column_count"] == 3
    assert "dataset_id" in data


def test_monitoring_summary_endpoint():
    res = client.get("/api/v1/monitoring/summary")
    assert res.status_code == 200
    data = res.json()
    assert "total_requests" in data
    assert "avg_latency_ms" in data


def test_monitoring_alerts_endpoint():
    res = client.get("/api/v1/monitoring/alerts")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_predict_endpoint_missing_model_returns_conflict():
    # If no model active or invalid request, ensure clean error code
    payload = {"features": [{"f1": 10}]}
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code in (404, 409, 422)
