# app/components/display.py

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple

# ----------------------------
# 1. Color Palette (consistent per class)
# ----------------------------
COLOR_MAP = {
    "Tank": (255, 0, 0),  # Red
    "Jeep": (0, 255, 0),  # Green
    "Missile": (0, 0, 255),  # Blue
    "Drone": (255, 255, 0),  # Cyan
    "Human": (255, 165, 0),  # Orange
    "Unknown": (200, 200, 200)  # Grey
}


# ----------------------------
# 2. Draw Bounding Boxes with Labels
# ----------------------------
def draw_boxes_with_labels(
        image: Image.Image,
        detections: List[Dict[str, any]],
        box_color: Tuple[int, int, int] = (0, 255, 0),
        font_scale: float = 0.5,
        box_thickness: int = 2
) -> np.ndarray:
    """
    Draws bounding boxes and labels on the image using OpenCV
    :param image: PIL image
    :param detections: List of dicts: {box: (x1, y1, x2, y2), label: str, confidence: float}
    :return: Annotated OpenCV image
    """
    cv_image = np.array(image.convert("RGB"))
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det.get('label', 'Object')
        conf = det.get('confidence', 0.0)
        color = COLOR_MAP.get(label, box_color)

        # Draw rectangle
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, box_thickness)

        # Prepare label text
        label_text = f"{label}: {round(conf * 100, 1)}%"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        # Draw filled background for label
        cv2.rectangle(cv_image, (x1, y1 - h - 10), (x1 + w + 4, y1), color, -1)
        cv2.putText(cv_image, label_text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1)

    return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)


# ----------------------------
# 3. Overlay Grad-CAM Heatmap on Image
# ----------------------------
def overlay_gradcam_on_image(image: Image.Image, gradcam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays Grad-CAM heatmap on the image
    :param image: PIL input image
    :param gradcam: Numpy heatmap (normalized [0-1] or [0-255])
    :param alpha: Blend ratio
    :return: Blended OpenCV image
    """
    original = np.array(image.resize((gradcam.shape[1], gradcam.shape[0])))

    # Normalize heatmap
    if gradcam.max() <= 1.0:
        gradcam = (gradcam * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(gradcam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


# ----------------------------
# 4. Example Data Format (for reference)
# ----------------------------
"""
detections = [
    {
        "box": (100, 50, 300, 250),         # Bounding box (x1, y1, x2, y2)
        "label": "Tank",                    # Object class
        "confidence": 0.87                  # Confidence score
    },
    ...
]
"""
