# app/main.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tempfile
from PIL import Image
from datetime import datetime

# Backend modules
from backend.services.explain_and_detect import run_explanation_pipeline
from backend.services.logger import OperatorLogger
from app.components.display import draw_boxes_with_labels
from app.utils.config import MODEL_CONFIDENCE_THRESHOLD

# --------------------- PAGE CONFIGURATION ---------------------
st.set_page_config(page_title="XAI Target Recognition System", layout="wide", page_icon="🛡️")

# --------------------- TITLE & INTRO ---------------------
st.title("🛡️ Explainable AI - Autonomous Target Recognition System")
st.markdown("""
Welcome to the **XAI Target Recognition Module** of **S.O.P.H.I.E. (Series One Processor Hyper Intelligence Encryptor)**.

This advanced system combines deep learning with explainable AI (XAI) tools to detect, recognize, and explain military targets in **images** and **video feeds**.

### 🔍 Key Features:
- **Real-time object detection**
- **Visual explanations using Grad-CAM, SHAP, and LIME**
- **Operator-based human-in-the-loop decision logging**
- **Robust analytics dashboard with filtering options**

Please upload a surveillance image or short video clip to begin analysis.
""")

st.markdown("---")

# --------------------- SIDEBAR CONFIG ---------------------
st.sidebar.title("📋 Navigation & Notes")
st.sidebar.markdown("""
**System Description:**
This module processes military-grade surveillance visuals using AI. Each target identified is cross-verified by an operator, supported with visual explanations.

**Use Cases:**
- Autonomous reconnaissance missions
- Intelligence-based threat analysis
- Decision-support for defense personnel
""")

st.sidebar.markdown("---")
st.sidebar.info("⚠️ Only authorized personnel should access and make decisions on the detected targets.")

# --------------------- INPUT TYPE SELECTION ---------------------
input_type = st.radio("📂 Select Input Type", ["Upload Image", "Upload Video"], horizontal=True)

# --------------------- IMAGE ANALYSIS SECTION ---------------------
if input_type == "Upload Image":
    st.header("🖼️ Upload Image for Target Recognition")
    uploaded_file = st.file_uploader("Choose a military surveillance image (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

        # Run detection and explanation
        st.subheader("🎯 Detection & Explainability Results")
        detections, explanations = run_explanation_pipeline(image)

        # Annotated image with boxes
        annotated_img = draw_boxes_with_labels(image, detections, threshold=MODEL_CONFIDENCE_THRESHOLD)
        st.image(annotated_img, caption="📌 Detected Targets", use_column_width=True)

        # Operator Decision Interface
        for i, det in enumerate(detections):
            st.markdown(f"---\n### 🎯 Target {i + 1}: `{det['label']}`")
            cols = st.columns([1.2, 1.8])

            with cols[0]:
                if "gradcam_overlay" in explanations:
                    st.image(explanations["gradcam_overlay"], caption="🔍 Grad-CAM Heatmap", use_column_width=True)

            with cols[1]:
                if "shap_image_path" in explanations:
                    st.image(explanations["shap_image_path"], caption="🔬 SHAP Explanation", use_column_width=True)

                st.markdown(f"**📊 Model Confidence:** `{det['confidence']:.2f}`")
                st.progress(min(det["confidence"], 1.0))

                # Human-in-the-loop confirmation
                decision = st.radio(
                    f"🧠 Do you confirm this target as `{det['label']}`?",
                    ("Confirm", "Reject"),
                    key=f"decision_{i}"
                )
                remarks = st.text_input(f"✏️ Remarks for Target {i + 1} (Optional)", key=f"remarks_{i}")

                if st.button(f"✅ Submit Review for Target {i + 1}", key=f"submit_{i}"):
                    logger = OperatorLogger()
                    logger.log_detection(
                        operator_id="admin",
                        detection=det,
                        operator_decision=decision,
                        comments=remarks
                    )
                    st.success(f"📝 Decision logged for Target {i + 1}")

# --------------------- VIDEO ANALYSIS SECTION ---------------------
elif input_type == "Upload Video":
    st.header("🎥 Upload Video for Target Tracking")
    uploaded_video = st.file_uploader("Upload a short surveillance video (max ~20s)", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        st.warning("⏳ Processing video... Please wait...")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > 100:
                break

            # Analyze every 15th frame
            if frame_count % 15 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                detections, _ = run_explanation_pipeline(pil_img)
                annotated = draw_boxes_with_labels(pil_img, detections, threshold=MODEL_CONFIDENCE_THRESHOLD)
                stframe.image(annotated, channels="RGB", use_column_width=True)

            frame_count += 1

        cap.release()
        st.success("✅ Video processing complete.")

# --------------------- DECISION LOGS SECTION ---------------------
st.markdown("---")
st.header("📄 Operator Decision Logs")
st.markdown("""
Below is the log of all detection decisions reviewed by the human operator during this session.
This data is stored for future audit, analysis, and feedback improvements to the XAI system.
""")

with st.expander("🔍 View Session Logs"):
    try:
        logs = pd.read_csv("reports/session_logs.csv")
        st.dataframe(logs, use_container_width=True)
    except FileNotFoundError:
        st.info("ℹ️ No decisions have been logged yet in this session.")
