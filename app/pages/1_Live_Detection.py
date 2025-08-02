# app/pages/1_Live_Detection.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import tempfile
import pandas as pd

# Backend core functions
from backend.services.explain_and_detect import run_explanation_pipeline
from backend.services.logger import OperatorLogger

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Live Detection", layout="wide", page_icon="🎥")
st.title("🎥 Live Target Detection & Explainability (Real-Time)")
st.markdown("""
Welcome to the **Live Detection Interface** of **S.O.P.H.I.E. (Series One Processor Hyper Intelligence Encryptor)**.

This module connects to your **webcam** or any **external camera**, allowing you to:
- 📡 Continuously monitor surveillance zones
- 🧠 Detect and explain military-grade objects in real-time
- 🧑‍💻 Log decisions and feedback from the operator in a structured format

---

### 💡 How it works:
1. Select the camera index and other options from the sidebar.
2. Click `Start Live Detection` to begin.
3. The system will analyze selected frames, draw bounding boxes, and generate XAI insights (Grad-CAM, SHAP).
4. You (the operator) can **confirm or reject** each detected object and leave remarks.
""")

# -------------------- SIDEBAR CONFIG --------------------
st.sidebar.title("⚙️ System Configuration")
st.sidebar.markdown("**Choose camera and performance options before starting detection.**")

camera_index = st.sidebar.selectbox("📷 Select Camera Index", [0, 1, 2], help="0 = Default webcam")
frame_skip = st.sidebar.slider("⏩ Process Every Nth Frame", min_value=1, max_value=10, value=5,
                               help="Skip frames to reduce compute load")
max_frames = st.sidebar.slider("🎞️ Max Frames to Analyze", min_value=10, max_value=300, value=100)
display_explanation = st.sidebar.toggle("🧠 Show Grad-CAM Explanation", value=True)

st.sidebar.markdown("---")
st.sidebar.info("This module is used only for authorized surveillance operations. Ensure responsible usage.")

# -------------------- LIVE DETECTION START --------------------
st.markdown("## 🚀 Start Real-Time Surveillance")
start_button = st.button("▶️ Begin Live Detection")

if start_button:
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        st.error("🚫 Could not access the camera. Please verify index or permissions.")
    else:
        st.success("📷 Camera started successfully.")
        st.warning("⚠️ Avoid moving the camera abruptly for better detection consistency.")

        frame_counter = 0
        stframe = st.empty()

        while cap.isOpened() and frame_counter < max_frames:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Unable to read frame from the camera.")
                break

            if frame_counter % frame_skip == 0:
                # Convert OpenCV frame to PIL image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)

                # Run backend detection + explanation pipeline
                result, explanations = run_explanation_pipeline(pil_img)

                # Display the annotated frame
                stframe.image(result["annotated_image"], caption=f"📍 Frame {frame_counter}", use_column_width=True)

                # Explanation section for each object
                if display_explanation:
                    for i, obj in enumerate(explanations):
                        with st.expander(f"🎯 Target {i + 1}: `{obj['label']}`"):
                            col1, col2 = st.columns([1, 2])

                            with col1:
                                st.image(obj["gradcam"], caption="🔥 Grad-CAM Heatmap")

                            with col2:
                                st.markdown(f"**🧠 SHAP Summary:** {obj['shap_summary']}")
                                st.markdown(f"**📈 Confidence:** `{round(obj['confidence'] * 100, 2)}%`")
                                st.progress(min(obj['confidence'], 1.0))

                                # Operator feedback form
                                decision = st.radio(
                                    f"📌 Confirm `{obj['label']}`?",
                                    ("Confirm", "Reject"),
                                    key=f"decision_{frame_counter}_{i}"
                                )
                                remarks = st.text_input(
                                    f"✏️ Remarks (optional)", key=f"remarks_{frame_counter}_{i}"
                                )

                                if st.button(f"💾 Log Decision for Target {i + 1}", key=f"log_btn_{frame_counter}_{i}"):
                                    logger = OperatorLogger()
                                    logger.log_detection(
                                        operator_id="admin",
                                        detection=obj,
                                        operator_decision=decision,
                                        comments=remarks
                                    )
                                    st.success(f"✅ Decision logged for Target {i + 1}")

            frame_counter += 1

        cap.release()
        st.success("📹 Live detection session ended.")
        st.balloons()

# -------------------- VIEW LOGS SECTION --------------------
st.markdown("---")
st.header("📄 Operator Review Logs")
st.markdown("""
Every detection made during this session is stored in a structured format and can be reviewed below.

This log supports **audit trails**, **post-operation reviews**, and **training datasets** for retraining AI models.
""")

with st.expander("🔍 View Session Logs"):
    try:
        logs = pd.read_csv("reports/session_logs.csv")
        st.dataframe(logs, use_container_width=True)
    except FileNotFoundError:
        st.info("No decisions have been logged yet.")
