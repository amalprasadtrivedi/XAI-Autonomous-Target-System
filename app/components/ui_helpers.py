# app/components/ui_helpers.py

import streamlit as st
from PIL import Image
import base64
from typing import Tuple


# 1. Display a target summary card
def target_card(index: int, label: str, confidence: float, gradcam: Image.Image, shap_summary: str):
    """
    Render a collapsible card showing target info, Grad-CAM image and SHAP explanation
    """
    with st.expander(f"🎯 Target {index + 1}: `{label}` | Confidence: {round(confidence * 100, 2)}%"):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(gradcam, caption="Grad-CAM Heatmap", use_column_width=True)
        with col2:
            st.markdown(f"**🧠 SHAP Summary:**")
            st.write(shap_summary)
            st.progress(min(confidence, 1.0))


# 2. Display metric-style cards (use in Summary Stats)
def stat_card(title: str, value: str, delta: str = "", help_text: str = ""):
    """
    Show a metric card in a column (streamlit.metric)
    """
    st.metric(label=title, value=value, delta=delta, help=help_text)


# 3. Colored tags
def status_tag(label: str, color: str = "blue"):
    """
    Render a stylized colored tag
    """
    tag_style = f"""
        <span style='
            background-color: {color};
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        '>{label}</span>
    """
    st.markdown(tag_style, unsafe_allow_html=True)


# 4. Section divider
def section_divider(title: str = "", icon: str = "🧭"):
    """
    Render a divider with title
    """
    st.markdown("---")
    st.markdown(f"### {icon} {title}")


# 5. Centered image display
def show_centered_image(img: Image.Image, caption: str = "", width: int = 600):
    """
    Display a centered image with optional caption
    """
    st.image(img, caption=caption, width=width)


# 6. File-to-base64 (for icons, badge display, or image encoding)
def image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 string
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# 7. Toggle row for decision logging
def decision_radio(target_id: int, label: str) -> Tuple[str, str]:
    """
    Render radio input for Confirm/Reject and remarks
    """
    st.markdown(f"#### 🎯 Target `{label}`")
    decision = st.radio(
        "Operator Decision:",
        ("Confirm", "Reject"),
        key=f"decision_{target_id}"
    )
    remarks = st.text_input("Remarks (Optional)", key=f"remarks_{target_id}")
    return decision, remarks
