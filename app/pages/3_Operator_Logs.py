# app/pages/3_Operator_Logs.py

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from app.utils.config import SESSION_LOG_FILE
from backend.services.logger import get_logs

# -------------------- Page Setup --------------------
st.set_page_config(
    page_title="Operator Logs",
    layout="wide",
    page_icon="📋"
)

# -------------------- Title & Introduction --------------------
st.title("📋 Operator Decision Logs")

st.markdown("""
Welcome to the **Operator Decision Log Panel** of the **XAI Target Recognition System (S.O.P.H.I.E.)**.

This dashboard allows you to:
- 📖 Review historical decisions made by human operators
- 📅 Filter decisions by label, date, or type (False Positives / Negatives)
- 📈 View summary statistics on operator performance
- 💾 Export logs for audit or retraining datasets

> Built for post-mission analysis, training validation, and AI auditing.
""")

# --------------------- Sidebar ---------------------
st.sidebar.title("XAI Target Recognition")
st.sidebar.info("Part of the Series One Processor Hyper Intelligence Encryptor (S.O.P.H.I.E.)")
st.sidebar.markdown("---")

# -------------------- Load Data --------------------
try:
    df = pd.DataFrame(get_logs(SESSION_LOG_FILE))  # Load logs from CSV
    if df.empty:
        st.warning("⚠️ No logs available yet. Start by running detection or submitting decisions.")
        st.stop()
    else:
        st.success(f"✅ {len(df)} log entries successfully loaded.")
except FileNotFoundError:
    st.error("🚫 Log file not found. Please check `reports/session_logs.csv`.")
    st.stop()

# -------------------- Data Preprocessing --------------------
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["fp"] = df["fp"].astype(bool)
df["fn"] = df["fn"].astype(bool)

# -------------------- Sidebar Filtering Section --------------------
st.sidebar.header("🔍 Filter Logs")

# Filter by Date Range
min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

# Filter by Label
label_filter = st.sidebar.multiselect(
    "Target Labels",
    options=df["label"].unique(),
    default=list(df["label"].unique())
)

# Filter by Operator Decision
decision_filter = st.sidebar.multiselect(
    "Operator Decision",
    options=["Confirm", "Reject"],
    default=["Confirm", "Reject"]
)

# False Positive / Negative Filter
fp_filter = st.sidebar.checkbox("Show Only False Positives")
fn_filter = st.sidebar.checkbox("Show Only False Negatives")

# -------------------- Apply Filters --------------------
filtered_df = df.copy()

filtered_df = filtered_df[
    (filtered_df["label"].isin(label_filter)) &
    (filtered_df["operator_decision"].isin(decision_filter)) &
    (filtered_df["timestamp"].dt.date >= date_range[0]) &
    (filtered_df["timestamp"].dt.date <= date_range[1])
]

if fp_filter:
    filtered_df = filtered_df[filtered_df["fp"] == True]
if fn_filter:
    filtered_df = filtered_df[filtered_df["fn"] == True]

# -------------------- Filtered Table --------------------
st.markdown("## 🔍 Filtered Operator Logs")

st.markdown("""
Below is the table of all operator decisions that match your current filters.
You may use this table to analyze patterns, performance, and review human oversight in AI-assisted detection.
""")

st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# -------------------- Summary Statistics --------------------
st.markdown("---")
st.markdown("## 📊 Summary Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🧾 Total Decisions", len(df))

with col2:
    st.metric("❗ False Positives", int(df["fp"].sum()))

with col3:
    st.metric("🚫 False Negatives", int(df["fn"].sum()))

# -------------------- Export Section --------------------
st.markdown("---")
st.markdown("## 📥 Export Filtered Logs")

export_name = f"XAI_Operator_Logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Logs as CSV",
    data=csv_data,
    file_name=export_name,
    mime="text/csv"
)

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("""
✅ **XAI Log Review Completed**  
This tool is essential for maintaining oversight and compliance with ethical AI standards in military surveillance applications.

> Version: `SOPHIE-XAI v1.0`  
> Logged operator actions are stored in `/reports/session_logs.csv`
""")
