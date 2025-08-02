# backend/services/model_loader.py

import torch
import torchvision.models as models
import timm
from typing import Optional
from app.utils.config import DETECTION_MODEL_PATH, XAI_BACKBONE_MODEL


class ModelLoader:
    def __init__(self, use_cuda: Optional[bool] = None):
        self.device = self._select_device(use_cuda)
        self.detection_model = None
        self.explainer_model = None

    def _select_device(self, use_cuda: Optional[bool]) -> str:
        if use_cuda is not None:
            return "cuda" if use_cuda else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_detection_model(self, model_path: str = DETECTION_MODEL_PATH):
        """
        Load YOLOv5 detection model
        """
        try:
            self.detection_model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
            self.detection_model.to(self.device)
            self.detection_model.eval()
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load detection model: {e}")
        return self.detection_model

    def load_xai_backbone(self, model_name: str = XAI_BACKBONE_MODEL, pretrained: bool = True):
        """
        Load a classification model for Grad-CAM/SHAP use (e.g., ResNet50, EfficientNet)
        Supported: torchvision or timm models
        """
        try:
            if model_name.startswith("resnet"):
                model = getattr(models, model_name)(pretrained=pretrained)
            else:
                model = timm.create_model(model_name, pretrained=pretrained)

            model.to(self.device)
            model.eval()
            self.explainer_model = model
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load XAI model '{model_name}': {e}")
        return self.explainer_model

    def get_device(self):
        return self.device

    def get_detection_model(self):
        if self.detection_model is None:
            raise ValueError("Detection model not loaded. Call `load_detection_model()` first.")
        return self.detection_model

    def get_xai_model(self):
        if self.explainer_model is None:
            raise ValueError("XAI model not loaded. Call `load_xai_backbone()` first.")
        return self.explainer_model
