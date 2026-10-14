"""Streamlit Page 2: Train Models."""

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Train Models - MLGuard", page_icon="⚡", layout="wide")
st.title("⚡ Model Training Engine")
st.write("Configure and train multiple scikit-learn models with strict train/test isolation and cross-validation.")

# Fetch datasets
try:
    ds_resp = requests.get(f"{API_BASE_URL}/api/v1/datasets")
    datasets = ds_resp.json() if ds_resp.status_code == 200 else []
except Exception:
    datasets = []

if not datasets:
    st.warning("⚠️ No datasets found. Please upload a dataset or execute demo seeding first.")
    st.stop()

ds_options = {f"{d['filename']} ({d['dataset_id']})": d for d in datasets}
selected_ds_label = st.selectbox("Select Training Dataset", list(ds_options.keys()))
selected_ds = ds_options[selected_ds_label]

# Fetch full dataset metadata to inspect columns
try:
    meta_resp = requests.get(f"{API_BASE_URL}/api/v1/datasets/{selected_ds['dataset_id']}")
    meta_data = meta_resp.json() if meta_resp.status_code == 200 else {}
    available_cols = [c["name"] for c in meta_data.get("columns", [])]
except Exception:
    available_cols = []

col_t1, col_t2 = st.columns(2)
with col_t1:
    target_col = st.selectbox("Target Column", available_cols, index=len(available_cols)-1 if available_cols else 0)

with col_t2:
    default_task = selected_ds.get("suggested_task") or "regression"
    task_idx = 0 if default_task == "regression" else (1 if default_task == "binary_classification" else 2)
    task_type = st.selectbox(
        "Prediction Task Type",
        ["regression", "binary_classification", "multiclass_classification"],
        index=task_idx,
    )

st.subheader("Algorithm Selection & Hyperparameters")
col_alg1, col_alg2 = st.columns(2)

with col_alg1:
    if task_type == "regression":
        available_algos = ["linear_regression", "random_forest", "gradient_boosting"]
    else:
        available_algos = ["logistic_regression", "random_forest", "gradient_boosting"]

    selected_algos = st.multiselect("Select Algorithms to Train & Compare", available_algos, default=available_algos)

with col_alg2:
    test_size = st.slider("Held-out Test Split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    cv_folds = st.slider("Cross-Validation Folds", min_value=2, max_value=10, value=3)

with st.expander("Advanced Preprocessing Settings"):
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        num_imp = st.selectbox("Numeric Imputation", ["median", "mean"])
    with col_p2:
        scaling = st.selectbox("Feature Scaling", ["standard", "robust", "none"])
    with col_p3:
        random_seed = st.number_input("Deterministic Random Seed", value=42)

if st.button("🚀 Start Training Pipeline", type="primary"):
    if not selected_algos:
        st.error("Please select at least one algorithm.")
    else:
        payload = {
            "dataset_id": selected_ds["dataset_id"],
            "target_column": target_col,
            "task_type": task_type,
            "models": selected_algos,
            "test_size": test_size,
            "cv_folds": cv_folds,
            "random_state": random_seed,
            "numeric_imputer": num_imp,
            "scaling": scaling,
        }
        with st.spinner("Executing data preprocessing, model fitting, cross-validation, and artifact logging..."):
            try:
                res = requests.post(f"{API_BASE_URL}/api/v1/training/jobs", json=payload)
                if res.status_code == 201:
                    job = res.json()
                    st.success(f"Training Complete! Job ID: `{job['job_id']}`")
                    st.balloons()

                    st.markdown(f"### 🏆 Best Model: **{job['best_model_name']}**")
                    if job.get("best_metric_value") is not None:
                        st.write(f"Objective Metric Score: `{job['best_metric_value']}`")

                    # Display comparison table
                    results_rows = []
                    for r in job.get("results", []):
                        row = {
                            "Algorithm": r["algorithm"],
                            "Model Name": r["model_name"],
                            "Duration (s)": r["duration_seconds"],
                            "CV Mean Score": r.get("cv_mean"),
                        }
                        row.update(r["metrics"])
                        results_rows.append(row)

                    st.subheader("Model Evaluation Summary (Held-out Test Data)")
                    st.dataframe(pd.DataFrame(results_rows), use_container_width=True)
                else:
                    st.error(f"Training failed: {res.text}")
            except Exception as e:
                st.error(f"Connection failure: {str(e)}")
