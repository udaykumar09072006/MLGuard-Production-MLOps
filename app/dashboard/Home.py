"""MLGuard Streamlit Dashboard - Home Page."""

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="MLGuard - Production ML Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ MLGuard: Machine Learning Platform")
st.markdown(
    "**Automated ML Training, Deployment, Model Registry & Real-Time Monitoring**\n\n"
    "MLGuard orchestrates tabular ML pipelines, tracks experiments with MLflow, registers models, "
    "serves low-latency predictions via FastAPI, and monitors input distributions for statistical data drift."
)

st.divider()

# System Health Probe
col_h1, col_h2, col_h3, col_h4 = st.columns(4)

try:
    health_resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
    is_healthy = health_resp.status_code == 200
except Exception:
    is_healthy = False

try:
    ready_resp = requests.get(f"{API_BASE_URL}/ready", timeout=3)
    is_ready = ready_resp.status_code == 200 and ready_resp.json().get("status") == "ready"
except Exception:
    is_ready = False

with col_h1:
    st.metric("API Service Status", "🟢 Online" if is_healthy else "🔴 Offline")

with col_h2:
    st.metric("Readiness Probe", "🟢 Ready" if is_ready else "🟡 Degraded")

# Fetch Active Model
try:
    active_resp = requests.get(f"{API_BASE_URL}/api/v1/models/active", timeout=3)
    active_model = active_resp.json() if active_resp.status_code == 200 else None
except Exception:
    active_model = None

with col_h3:
    if active_model:
        st.metric("Active Model", f"{active_model['name']} (v{active_model['version']})")
    else:
        st.metric("Active Model", "None Deployed")

# Fetch Alerts Count
try:
    alerts_resp = requests.get(f"{API_BASE_URL}/api/v1/monitoring/alerts?status=ACTIVE", timeout=3)
    active_alerts = len(alerts_resp.json()) if alerts_resp.status_code == 200 else 0
except Exception:
    active_alerts = 0

with col_h4:
    st.metric("Active Drift/Latency Alerts", active_alerts, delta="Needs Review" if active_alerts > 0 else "Normal")

st.divider()

# Navigation Quick Links
st.subheader("Platform Quick Navigation")
nav1, nav2, nav3 = st.columns(3)

with nav1:
    st.markdown("### 1. 📂 Dataset Explorer")
    st.write("Upload CSV files, inspect schema types, detect data leakage, check missingness, and explore distributions.")

with nav2:
    st.markdown("### 2. ⚡ Train Models")
    st.write("Configure Linear/Logistic Regression, Random Forest, and Gradient Boosting with K-Fold cross-validation.")

with nav3:
    st.markdown("### 3. 🧪 Experiments")
    st.write("Compare runs across MAE, RMSE, R², Accuracy, F1-Score, ROC-AUC, and inspect MLflow parameters.")

nav4, nav5, nav6 = st.columns(3)

with nav4:
    st.markdown("### 4. 🏛️ Model Registry")
    st.write("Inspect immutable versioned artifacts, validation qualification status, and activate serving models.")

with nav5:
    st.markdown("### 5. 🔮 Prediction Playground")
    st.write("Serve real-time online predictions via dynamic feature forms with measured millisecond latency.")

with nav6:
    st.markdown("### 6. 📊 Monitoring & Drift")
    st.write("Track Kolmogorov-Smirnov drift tests, latency percentiles (p50/p95/p99), and manage alerts.")
