# app/pages/2_Explanation_Review.py

import streamlit as st
import pandas as pd
import os
from PIL import Image
from datetime import datetime

from backend.services.logger import update_remarks, get_logs
from app.utils.config import EXPLANATION_IMAGES_PATH, SESSION_LOG_FILE

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Explanation Review", layout="wide", page_icon="🧠")
st.title("🧠 Explanation Review Dashboard")
st.markdown("""
Welcome to the **Explanation Review Panel** of the **XAI Target Recognition System (S.O.P.H.I.E.)**.

This module allows human operators, analysts, or auditors to:
- 🔍 Revisit **detection events**
- 📊 View **explanation overlays** from Grad-CAM and SHAP
- 📝 **Update feedback** or operator remarks
- 🔄 Filter historical records based on label or decision

> Built for transparency, auditability, and responsible use of AI in military applications.
""")

# -------------------- Sidebar --------------------
st.sidebar.title("🧠 Explanation Review")
st.sidebar.info("Review logs, confirm explanations, and leave remarks for accountability.")
st.sidebar.markdown("---")

# -------------------- Load Session Logs --------------------
try:
    logs_list = get_logs(SESSION_LOG_FILE)
    logs_df = pd.DataFrame(logs_list)

    if logs_df.empty:
        st.warning("🚫 No logs available yet. Please run a detection session first.")
        st.stop()
    else:
        st.success(f"📁 Loaded {len(logs_df)} detection log entries.")
except FileNotFoundError:
    st.error("Log file not found. Please check `reports/session_logs.csv`.")
    st.stop()

# -------------------- Validate Columns --------------------
required_columns = {"target_id", "label", "decision", "confidence", "comments"}
missing = required_columns - set(logs_df.columns)
if missing:
    st.error(f"❌ Missing columns in logs: {missing}")
    st.stop()

# -------------------- Filtering Panel --------------------
st.sidebar.header("🔍 Filter Records")

label_filter = st.sidebar.multiselect(
    "Filter by Detected Label",
    options=logs_df["label"].unique(),
    default=logs_df["label"].unique()
)

decision_filter = st.sidebar.multiselect(
    "Filter by Operator Decision",
    options=["Confirm", "Reject"],
    default=["Confirm", "Reject"]
)

# Apply filtering
filtered_df = logs_df[
    (logs_df["label"].isin(label_filter)) &
    (logs_df["decision"].isin(decision_filter))
]

# -------------------- Display Logs Table --------------------
st.markdown("## 📊 Detection & Review Logs")

st.markdown(f"""
The table below summarizes the filtered detection events. You can use this view to:
- Inspect previous AI-based detections
- Validate operator decisions
- Ensure explainability and human-in-the-loop trust
""")

st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# -------------------- Detailed Review Section --------------------
st.markdown("---")
st.header("🔬 Explanation Analysis for Individual Detections")

for idx, row in filtered_df.iterrows():
    st.markdown(f"### 🎯 Target ID: `{row['target_id']}` | Label: `{row['label']}` | Confidence: `{round(float(row['confidence']) * 100, 2)}%`")

    # Define image paths
    gradcam_path = os.path.join(EXPLANATION_IMAGES_PATH, f"gradcam_target_{row['target_id']}.png")
    shap_path = os.path.join(EXPLANATION_IMAGES_PATH, f"shap_target_{row['target_id']}.png")

    # Split layout into two columns: explanation visuals + operator update
    col1, col2 = st.columns([1.2, 2])

    # -------------------- Left: Visual Explanation --------------------
    with col1:
        if os.path.exists(gradcam_path):
            st.image(gradcam_path, caption="🔥 Grad-CAM Explanation", use_column_width=True)
        else:
            st.warning("⚠️ Grad-CAM image not found.")

        if os.path.exists(shap_path):
            st.image(shap_path, caption="🧠 SHAP Insight", use_column_width=True)
        else:
            st.info("ℹ️ SHAP image not available.")

    # -------------------- Right: Operator Feedback --------------------
    with col2:
        st.markdown("#### 🧾 Operator Review and Update")
        st.markdown("This section allows analysts to review or update remarks provided earlier.")

        current_remarks = row.get("comments", "")
        new_remarks = st.text_area(
            f"✏️ Update Remarks for Target `{row['target_id']}`",
            value=current_remarks,
            height=100,
            key=f"remarks_input_{idx}"
        )

        if st.button("💾 Save Remarks", key=f"save_btn_{idx}"):
            success = update_remarks(
                log_file=SESSION_LOG_FILE,
                target_id=row["target_id"],
                new_remarks=new_remarks
            )
            if success:
                st.success("✅ Remarks updated successfully.")
            else:
                st.error("❌ Failed to update remarks.")

    st.markdown("---")

# -------------------- System Summary --------------------
st.markdown("""
## 🧩 System Insight

The XAI Target Recognition System integrates Explainable AI into object detection pipelines for military reconnaissance and autonomous systems. This review module:

- Stores detection records with associated metadata
- Uses **Grad-CAM** to highlight key regions used by the AI model
- Applies **SHAP (SHapley Additive exPlanations)** to quantify feature contributions
- Enables human feedback for **accountability** and **model refinement**

### Why This Matters
- ✅ Builds operator trust in AI decisions
- ✅ Identifies potential false positives
- ✅ Helps retrain AI on edge cases

> A vital tool for ensuring ethical and explainable defense AI systems.
""")
