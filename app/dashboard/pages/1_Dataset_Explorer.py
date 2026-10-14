"""Streamlit Page 1: Dataset Explorer."""

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Dataset Explorer - MLGuard", page_icon="📂", layout="wide")
st.title("📂 Dataset Explorer & Validation Suite")
st.write("Upload a CSV file or inspect existing registered datasets. Validate schemas and identify leakage risks.")

tab_upload, tab_view = st.tabs(["Upload New Dataset", "Registered Datasets"])

with tab_upload:
    uploaded_file = st.file_uploader("Select CSV file", type=["csv"])
    if uploaded_file is not None:
        if st.button("Upload and Validate Dataset", type="primary"):
            with st.spinner("Uploading and validating schema..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                try:
                    resp = requests.post(f"{API_BASE_URL}/api/v1/datasets/upload", files=files)
                    if resp.status_code == 201:
                        data = resp.json()
                        st.success(f"Dataset successfully registered! ID: `{data['dataset_id']}`")

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Row Count", data["row_count"])
                        col2.metric("Column Count", data["column_count"])
                        col3.metric("Suggested Target", data.get("suggested_target", "None"))
                        col4.metric("Suggested Task", data.get("suggested_task", "None"))

                        # Display Warnings
                        if data.get("warnings"):
                            st.warning("⚠️ Validation Warnings Detected:")
                            for w in data["warnings"]:
                                st.write(f"- **[{w['severity'].upper()}]** {w['code']}: {w['message']}")
                        else:
                            st.info("✅ Zero data integrity warnings found.")

                        # Column details
                        st.subheader("Feature Schema & Statistics")
                        cols_df = pd.DataFrame(data["columns"])
                        st.dataframe(cols_df, use_container_width=True)
                    else:
                        st.error(f"Validation Error: {resp.text}")
                except Exception as e:
                    st.error(f"Connection failure: {str(e)}")

with tab_view:
    try:
        datasets_resp = requests.get(f"{API_BASE_URL}/api/v1/datasets")
        if datasets_resp.status_code == 200:
            datasets = datasets_resp.json()
            if datasets:
                st.dataframe(pd.DataFrame(datasets), use_container_width=True)
            else:
                st.info("No uploaded datasets yet. Upload a CSV or run `python scripts/seed_demo.py`.")
    except Exception as e:
        st.warning(f"Could not connect to FastAPI backend at {API_BASE_URL}: {e}")
