# backend/services/explain_and_detect.py

from typing import Dict, List, Optional
from PIL import Image
import numpy as np

import torch
import torchvision.models as models

from backend.detection.object_detector import ObjectDetector
from backend.xai.grad_cam import GradCAM
from backend.xai.lime_explainer import LimeImageExplainerWrapper
from backend.xai.shap_explainer import SHAPImageExplainer
from backend.detection.model_loader import ModelLoader
from backend.utils.model_utils import remove_inplace_relu
from app.utils.config import EXPLANATION_IMAGES_PATH


class ExplainAndDetect:
    def __init__(self):
        # Load device info
        self.model_loader = ModelLoader()
        self.device = self.model_loader.get_device()

        # Load object detection model
        self.detector = ObjectDetector()

        # Load classification model for explainability
        self.xai_model = models.resnet18(pretrained=True)
        remove_inplace_relu(self.xai_model)  # Important for SHAP/GradCAM
        self.xai_model.to(self.device).eval()

        # Initialize explanation modules
        self.gradcam = GradCAM(self.xai_model, device=self.device)
        self.lime = LimeImageExplainerWrapper(self.xai_model, device=self.device)
        self.shap = SHAPImageExplainer(self.xai_model, device=self.device)

    def process(
        self,
            image: Image.Image,
            target_class_index: Optional[int] = None,
            explain_methods: List[str] = ["gradcam", "lime", "shap"]
    ) -> Dict:
        """
        Run object detection + XAI methods on an image.

        Returns:
            Dict with detection results + explanation overlays/paths.
        """
        detections = self.detector.detect(image)

        # Default class index if not provided
        if target_class_index is None and len(detections) > 0:
            target_class_index = 0  # or use label -> index mapping

        explanations = {}

        if "gradcam" in explain_methods:
            heatmap = self.gradcam.generate_heatmap(image, target_class=target_class_index)
            overlay = self.gradcam.overlay_heatmap(image, heatmap)
            explanations["gradcam_overlay"] = overlay

        if "lime" in explain_methods:
            lime_overlay = self.lime.explain(image)
            explanations["lime_overlay"] = lime_overlay

        if "shap" in explain_methods:
            shap_path = self.shap.explain(image, class_index=target_class_index)
            explanations["shap_image_path"] = shap_path

        return detections, explanations


# Global instance (singleton-like)
_explainer = ExplainAndDetect()


def run_explanation_pipeline(image: Image.Image, target_class_index: int = None) -> (List[Dict], Dict):
    """
    Trigger detection + explanation pipeline for a given image.

    Args:
        image: PIL.Image.Image
        target_class_index: Optional[int]

    Returns:
        Dict: Explanation overlays, paths, and detections
    """
    return _explainer.process(
        image=image,
        target_class_index=target_class_index,
        explain_methods=["gradcam", "lime", "shap"]
    )
