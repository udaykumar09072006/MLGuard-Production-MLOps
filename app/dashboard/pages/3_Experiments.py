"""Streamlit Page 3: Experiments & MLflow Tracking."""

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Experiments - MLGuard", page_icon="🧪", layout="wide")
st.title("🧪 Experiment Tracking & Comparison")
st.write("Compare runs logged across experiments with metrics, hyperparameters, and artifacts.")

try:
    models_resp = requests.get(f"{API_BASE_URL}/api/v1/models")
    models = models_resp.json() if models_resp.status_code == 200 else []
except Exception as e:
    models = []
    st.error(f"Cannot connect to API: {e}")

if not models:
    st.info("No training runs recorded yet. Train models in the 'Train Models' tab or run `python scripts/seed_demo.py`.")
else:
    # Build runs table
    table_data = []
    for m in models:
        row = {
            "Model ID": m["model_id"],
            "Algorithm": m["algorithm"],
            "Task": m["task_type"],
            "Target": m["target_column"],
            "Version": f"v{m['version']}",
            "Active Serving": "🟢 YES" if m["is_active"] else "⚪ No",
            "Status": m["validation_status"],
            "Created": m["created_at"][:19],
        }
        for k, v in m.get("metrics", {}).items():
            if isinstance(v, (int, float)):
                row[k.upper()] = round(v, 4)
        table_data.append(row)

    df_runs = pd.DataFrame(table_data)
    st.subheader("Leaderboard & Experiment Comparison")
    st.dataframe(df_runs, use_container_width=True)

    # Detailed Run Inspection
    st.subheader("Run Inspector")
    model_ids = [m["model_id"] for m in models]
    chosen_id = st.selectbox("Inspect Model Run", model_ids)
    chosen_model = next(m for m in models if m["model_id"] == chosen_id)

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Model Metadata")
        st.json({
            "model_id": chosen_model["model_id"],
            "algorithm": chosen_model["algorithm"],
            "task_type": chosen_model["task_type"],
            "target_column": chosen_model["target_column"],
            "run_id": chosen_model["run_id"],
            "artifact_path": chosen_model["artifact_path"],
            "is_active": chosen_model["is_active"],
            "validation_status": chosen_model["validation_status"],
        })

    with col2:
        st.write("### Holdout Evaluation Metrics")
        st.json(chosen_model["metrics"])
