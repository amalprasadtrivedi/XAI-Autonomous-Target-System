# app/utils/config.py

import os

# -------------------------------
# Model & Confidence Settings
# -------------------------------

# Confidence threshold for detections to be considered
MODEL_CONFIDENCE_THRESHOLD = 0.5

# Path to your YOLO or detection model (update this based on actual use)
DETECTION_MODEL_PATH = "models/yolov8n.pt"

# Name of the backend model used for Grad-CAM (e.g., "resnet50", "efficientnet", etc.)
XAI_BACKBONE_MODEL = "resnet50"

# Target image size for model input
MODEL_IMAGE_SIZE = (640, 640)


# -------------------------------
# Data & Reports Paths
# -------------------------------

# Where session logs are saved
SESSION_LOG_FILE = "reports/session_logs.csv"

# Where Grad-CAM images and SHAP plots are saved
EXPLANATION_IMAGES_PATH = "reports/"

# Optional: Save raw detection images or frames
RAW_IMAGE_DUMP_PATH = "reports/raw_inputs/"

# Export folder (for future PDF, XLSX, etc.)
EXPORT_FOLDER_PATH = "exports/"


# -------------------------------
# Visualization Settings
# -------------------------------

# Color palette for class-specific drawing
COLOR_PALETTE = {
    "Tank": (255, 0, 0),       # Red
    "Jeep": (0, 255, 0),       # Green
    "Missile": (0, 0, 255),    # Blue
    "Drone": (255, 255, 0),    # Yellow
    "Human": (255, 165, 0),    # Orange
    "Unknown": (180, 180, 180) # Gray
}

# Grad-CAM overlay transparency
GRADCAM_ALPHA = 0.5

# Default font scale and thickness for box labels
BOX_FONT_SCALE = 0.5
BOX_LINE_THICKNESS = 2


# -------------------------------
# Create required folders at runtime
# -------------------------------

def ensure_directories():
    os.makedirs(EXPLANATION_IMAGES_PATH, exist_ok=True)
    os.makedirs(RAW_IMAGE_DUMP_PATH, exist_ok=True)
    os.makedirs(EXPORT_FOLDER_PATH, exist_ok=True)
    if not os.path.exists(SESSION_LOG_FILE):
        with open(SESSION_LOG_FILE, "w") as f:
            f.write("timestamp,target_id,label,confidence,decision,remarks,false_positive,false_negative,updated_at\n")


# Call this at app start
ensure_directories()
