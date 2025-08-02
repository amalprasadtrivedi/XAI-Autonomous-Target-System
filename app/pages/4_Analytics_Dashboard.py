# app/pages/4_Analytics_Dashboard.py

import streamlit as st
import pandas as pd
from datetime import datetime

from backend.services.logger import get_logs
from app.utils.config import SESSION_LOG_FILE
from app.components.charts import (
    decision_pie_chart,
    error_bar_chart,
    decisions_over_time_chart,
    confidence_distribution_chart
)

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Analytics Dashboard", layout="wide", page_icon="📊")

# -------------------- Title & System Description --------------------
st.title("📈 XAI Analytics Dashboard")
st.markdown("""
Welcome to the **XAI Analytics & Insights Dashboard**, a critical module of the
**Series One Processor Hyper Intelligence Encryptor (S.O.P.H.I.E.)**.

This system leverages Explainable AI (XAI) to detect and classify military targets with human oversight.
Use this dashboard to explore:
- Trends in operator decisions
- Detection confidence distribution
- False positive and false negative statistics
- Temporal patterns and mission-level insights

> **Note:** Data is sourced from operator logs recorded during detection sessions.
""")

# -------------------- Sidebar Information --------------------
st.sidebar.title("XAI Target Recognition")
st.sidebar.info("""
Use filters below to drill into specific mission logs and decision patterns.

This system ensures accountability, traceability, and enhanced operational awareness
for mission-critical applications.
""")
st.sidebar.markdown("---")

# -------------------- Load Operator Logs --------------------
try:
    df = pd.DataFrame(get_logs(SESSION_LOG_FILE))  # Pull logs from CSV
    if df.empty:
        st.warning("⚠️ No logs available yet. Start detection sessions to populate logs.")
        st.stop()
    else:
        st.success(f"✅ Loaded {len(df)} logged operator decisions.")
except FileNotFoundError:
    st.error("❌ Log file not found. Ensure `session_logs.csv` exists in the reports directory.")
    st.stop()

# -------------------- Data Cleaning --------------------
# Format data for analytics
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
df["fp"] = df["fp"].astype(int)
df["fn"] = df["fn"].astype(int)
df = df.rename(columns={
    "operator_decision": "decision",
    "fp": "false_positive",
    "fn": "false_negative"
})

# -------------------- Sidebar Filtering --------------------
st.sidebar.header("🔍 Filter Data")

# Date filter
min_date = df["timestamp"].dt.date.min()
max_date = df["timestamp"].dt.date.max()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

# Label filter
label_filter = st.sidebar.multiselect(
    "Target Labels", options=df["label"].unique(), default=list(df["label"].unique())
)

# Decision filter
decision_filter = st.sidebar.multiselect(
    "Decisions", options=["Confirm", "Reject"], default=["Confirm", "Reject"]
)

# Apply filters
filtered_df = df[
    (df["label"].isin(label_filter)) &
    (df["decision"].isin(decision_filter)) &
    (df["timestamp"].dt.date >= date_range[0]) &
    (df["timestamp"].dt.date <= date_range[1])
]

# -------------------- Filtered Summary Section --------------------
st.markdown("## 📂 Filtered Decision Records")
st.markdown(f"Showing **{len(filtered_df)}** decision logs based on current filters.")

# Metric Blocks
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎯 Total Targets", len(filtered_df))
with col2:
    st.metric("✅ Confirmed", int((filtered_df["decision"] == "Confirm").sum()))
with col3:
    st.metric("❗ False Positives", int(filtered_df["false_positive"].sum()))
with col4:
    st.metric("🚫 False Negatives", int(filtered_df["false_negative"].sum()))

# Optional table view
with st.expander("🔍 View Filtered Data Table"):
    st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# -------------------- Analytics Section --------------------
st.markdown("---")
st.markdown("## 📊 Visual Analytics")

with st.spinner("🧠 Generating interactive charts..."):

    # Decision distribution
    st.subheader("1️⃣ Decision Breakdown by Operator")
    st.markdown("Pie chart summarizing confirmed vs rejected targets.")
    decision_pie_chart(filtered_df)

    # Error chart
    st.subheader("2️⃣ Error Distribution by Class")
    st.markdown("Visualize false positives and false negatives across target types.")
    error_bar_chart(filtered_df)

    # Time-series trend
    st.subheader("3️⃣ Decision Trends Over Time")
    st.markdown("Explore how decisions varied across missions and time.")
    decisions_over_time_chart(filtered_df)

    # Confidence plot
    st.subheader("4️⃣ Confidence Score Distribution")
    st.markdown("Histogram of model confidence for all reviewed targets.")
    confidence_distribution_chart(filtered_df)

# -------------------- Footer / Notes --------------------
st.markdown("---")
st.markdown("""
📌 **Note to Analysts & Engineers**  
All visualizations are dynamically filtered. Use them for:
- 📈 Post-mission reviews
- 🧪 Training data audits
- ⚙️ Model improvement pipelines

🔐 **Log File Location:** `reports/session_logs.csv`  
🛡️ **System Version:** `SOPHIE-XAI v1.0`
""")
