# backend/services/object_detector.py

from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
from typing import List, Dict
from app.utils.config import DETECTION_MODEL_PATH, MODEL_CONFIDENCE_THRESHOLD


class ObjectDetector:
    def __init__(self, model_path: str = DETECTION_MODEL_PATH):
        """
        Initialize the YOLOv8 detection model.
        """
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
        except Exception as e:
            raise RuntimeError(f"❌ Error loading detection model: {e}")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Convert PIL image to OpenCV format (BGR) for YOLO input.
        """
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def detect(self, image: Image.Image, conf_threshold: float = MODEL_CONFIDENCE_THRESHOLD) -> List[Dict[str, any]]:
        """
        Perform object detection on the image and return structured results.

        Returns:
            List of dicts with:
            - 'box': (x1, y1, x2, y2)
            - 'label': class name
            - 'confidence': float
        """
        # Convert PIL to OpenCV
        frame = self.preprocess(image)

        # Run inference
        results = self.model(frame)[0]  # first result only

        detections = []
        for box in results.boxes:
            conf = float(box.conf.item())
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            label = self.class_names[cls_id]

            detections.append({
                "box": (x1, y1, x2, y2),
                "label": label,
                "confidence": round(conf, 3)
            })

        return detections

    def get_classes(self) -> List[str]:
        """
        Return list of class names in the model.
        """
        return list(self.class_names.values())
