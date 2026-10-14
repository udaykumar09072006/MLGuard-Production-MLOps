"""Streamlit Page 6: Model Monitoring & Data Drift Reports."""

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Monitoring & Drift - MLGuard", page_icon="📊", layout="wide")
st.title("📊 Model Monitoring, Latency & Data Drift")
st.write("Real-time telemetry, statistical hypothesis testing against baseline training data, and operational alerts.")

# 1. Latency & Throughput Metrics
st.subheader("1. Inference Telemetry & Latency Metrics")
try:
    summary_res = requests.get(f"{API_BASE_URL}/api/v1/monitoring/summary")
    summary = summary_res.json() if summary_res.status_code == 200 else {}
except Exception:
    summary = {}

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Inferences", summary.get("total_requests", 0))
m2.metric("Avg Latency", f"{summary.get('avg_latency_ms', 0)} ms")
m3.metric("p50 Latency", f"{summary.get('p50_latency_ms', 0)} ms")
m4.metric("p95 Latency", f"{summary.get('p95_latency_ms', 0)} ms")
m5.metric("Error Rate", f"{summary.get('error_rate', 0):.2%}")

st.divider()

# 2. Data Drift Detection
st.subheader("2. Statistical Data Drift Detection (KS & Chi-Square)")
st.write("Compares recent production inputs against baseline training reference data using 2-sample Kolmogorov-Smirnov test (numerical) and Chi-Square test (categorical).")

if st.button("🔄 Run Data Drift Analysis Now", type="primary"):
    with st.spinner("Executing statistical drift tests..."):
        try:
            drift_res = requests.get(f"{API_BASE_URL}/api/v1/monitoring/drift")
            if drift_res.status_code == 200:
                drift_data = drift_res.json()

                c_d1, c_d2, c_d3 = st.columns(3)
                c_d1.metric("Analyzed Features", drift_data["features_analyzed"])
                c_d2.metric("Production Observations", drift_data["sample_size"])
                c_d3.metric(
                    "Overall Drift Status",
                    "⚠️ DRIFT DETECTED" if drift_data["overall_drift_detected"] else "✅ STABLE",
                    delta=f"{drift_data['drifted_features_count']} drifted features",
                )

                if drift_data["feature_reports"]:
                    reports_table = []
                    for r in drift_data["feature_reports"]:
                        reports_table.append({
                            "Feature": r["feature_name"],
                            "Type": r["feature_type"],
                            "Method": r["test_method"],
                            "p-value": r["p_value"],
                            "Threshold": r["threshold"],
                            "Statistic": r["statistic"],
                            "Drift Detected": "🔴 YES" if r["drift_detected"] else "🟢 No",
                        })
                    st.dataframe(pd.DataFrame(reports_table), use_container_width=True)
                else:
                    st.info(f"Insufficient production samples (minimum {drift_data.get('sample_size')}/15 collected).")
            else:
                st.error(f"Drift analysis error: {drift_res.text}")
        except Exception as e:
            st.error(f"Connection failure: {str(e)}")

st.divider()

# 3. Alert Management
st.subheader("3. Monitoring Alerts & Incidents")
try:
    alerts_res = requests.get(f"{API_BASE_URL}/api/v1/monitoring/alerts")
    alerts = alerts_res.json() if alerts_res.status_code == 200 else []
except Exception:
    alerts = []

if not alerts:
    st.info("Zero active alerts. System operating within all drift and latency thresholds.")
else:
    for a in alerts:
        with st.container(border=True):
            col_a1, col_a2, col_a3 = st.columns([4, 2, 2])
            with col_a1:
                st.markdown(f"**[{a['severity']}] {a['alert_type']}** - {a['message']}")
                st.caption(f"Created: {a['created_at'][:19]} | Alert ID: `{a['alert_id']}`")
            with col_a2:
                st.write(f"Status: **{a['status']}**")
            with col_a3:
                if a["status"] != "RESOLVED":
                    if st.button("Mark Resolved", key=f"res_{a['alert_id']}"):
                        requests.post(f"{API_BASE_URL}/api/v1/monitoring/alerts/{a['alert_id']}/status?new_status=RESOLVED")
                        st.rerun()
                else:
                    st.write("Resolved ✅")
