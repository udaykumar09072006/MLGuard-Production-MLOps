"""Streamlit Page 5: Predictions Playground."""

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Predictions - MLGuard", page_icon="🔮", layout="wide")
st.title("🔮 Model Prediction Playground")
st.write("Serve real-time predictions through FastAPI against the deployed active model.")

try:
    active_resp = requests.get(f"{API_BASE_URL}/api/v1/models/active")
    active_model = active_resp.json() if active_resp.status_code == 200 else None
except Exception as e:
    active_model = None
    st.error(f"Cannot reach API: {e}")

if not active_model:
    st.warning("⚠️ No model is currently deployed. Please visit the Model Registry to activate a model.")
    st.stop()

st.info(
    f"Connected to Active Serving Model: **{active_model['name']} (v{active_model['version']})** | "
    f"Task: `{active_model['task_type']}` | Target: `{active_model['target_column']}`"
)

st.subheader("Dynamic Inference Form")
feature_inputs = {}

# Dynamically construct input fields based on feature_schema
cols = st.columns(2)
for idx, feat in enumerate(active_model["feature_schema"]):
    col = cols[idx % 2]
    with col:
        if feat["inferred_type"] == "numerical":
            val = col.number_input(
                label=f"{feat['name']} ({feat['dtype']})",
                value=float(feat.get("default_val") or 0.0),
                key=f"input_{feat['name']}",
            )
            feature_inputs[feat["name"]] = val
        else:
            options = feat.get("allowed_categories") or ["Category A", "Category B"]
            val = col.selectbox(
                label=f"{feat['name']} (categorical)",
                options=options,
                key=f"input_{feat['name']}",
            )
            feature_inputs[feat["name"]] = val

st.divider()

if st.button("⚡ Run Real-Time Prediction", type="primary"):
    payload = {
        "model_id": active_model["model_id"],
        "features": [feature_inputs],
    }
    with st.spinner("Submitting inference request to FastAPI..."):
        try:
            res = requests.post(f"{API_BASE_URL}/api/v1/predict", json=payload)
            if res.status_code == 200:
                pred_data = res.json()
                st.success("Prediction Succeeded!")

                col_res1, col_res2, col_res3 = st.columns(3)
                prediction_val = pred_data["predictions"][0]["prediction"]
                probabilities = pred_data["predictions"][0].get("probabilities")

                col_res1.metric("Predicted Output", str(prediction_val))
                col_res2.metric("Inference Latency", f"{pred_data['latency_ms']} ms")
                col_res3.metric("Model Version Used", f"v{pred_data['model_version']}")

                if probabilities:
                    st.write("### Class Probabilities")
                    st.json(probabilities)

                with st.expander("Inspect Raw JSON Response"):
                    st.json(pred_data)
            else:
                st.error(f"Prediction Error ({res.status_code}): {res.text}")
        except Exception as e:
            st.error(f"Inference request failed: {str(e)}")
