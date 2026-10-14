"""Streamlit Page 4: Model Registry."""

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Model Registry - MLGuard", page_icon="🏛️", layout="wide")
st.title("🏛️ Model Registry & Version Management")
st.write("Track versioned candidate models, review validation qualification statuses, and manage active deployments.")

try:
    resp = requests.get(f"{API_BASE_URL}/api/v1/models")
    models = resp.json() if resp.status_code == 200 else []
except Exception as e:
    models = []
    st.error(f"Cannot connect to API: {e}")

if not models:
    st.info("No registered models found. Train models or run `python scripts/seed_demo.py`.")
else:
    active_model = next((m for m in models if m["is_active"]), None)

    if active_model:
        st.success(
            f"🎯 Currently Deployed Active Model: **{active_model['name']} (Version {active_model['version']})** | "
            f"ID: `{active_model['model_id']}` | Task: `{active_model['task_type']}`"
        )
    else:
        st.warning("⚠️ No model is currently activated for production inference.")

    st.subheader("Candidate Model Versions")
    for m in models:
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            with col1:
                st.markdown(f"### {m['name']} (v{m['version']})")
                st.caption(f"ID: `{m['model_id']}` | Algo: `{m['algorithm']}` | Task: `{m['task_type']}`")
            with col2:
                st.write("**Key Metrics:**")
                st.write(m.get("metrics", {}))
            with col3:
                badge = "🟢 ACTIVE" if m["is_active"] else "⚪ Candidate"
                st.markdown(f"**Status:** {badge}")
                st.write(f"Validated: `{m['validation_status']}`")
            with col4:
                if not m["is_active"]:
                    if st.button("Promote to Active", key=f"act_{m['model_id']}", type="primary"):
                        act_resp = requests.post(
                            f"{API_BASE_URL}/api/v1/models/{m['model_id']}/activate",
                            json={"activated_by": "streamlit_user", "reason": "Dashboard operator promotion"},
                        )
                        if act_resp.status_code == 200:
                            st.success(f"Model {m['model_id']} promoted!")
                            st.rerun()
                        else:
                            st.error(f"Failed to activate: {act_resp.text}")
                else:
                    st.write("Active Serving Model")
