# backend/xai/shap_explainer.py

import os
import shap
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from app.utils.config import EXPLANATION_IMAGES_PATH


class SHAPImageExplainer:
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """
        model: A PyTorch image classification model (e.g., ResNet).
        device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model = self._wrap_model(model)
        self.model.to(self.device).eval()
        self.explainer = None  # Delayed initialization to avoid hook conflict

        # Preprocessing for PIL image to model input
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def _wrap_model(self, model):
        """
        Wrap model to avoid in-place ops error by cloning inputs.
        """

        class ModelWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, x):
                x = x.clone()
                return self.base_model(x)

        return ModelWrapper(model)

    def _init_explainer(self):
        """
        Initialize SHAP explainer with dummy background batch.
        """
        background = torch.randn((10, 3, 224, 224), device=self.device)
        try:
            self.explainer = shap.DeepExplainer(self.model, background)
        except Exception as e:
            raise RuntimeError(f"❌ SHAP initialization failed. Possibly due to in-place ops or hooks. Error: {e}")

    def explain(self, image: Image.Image, class_index: int = None) -> str:
        """
        Generate SHAP explanation for a single image.
        Returns: path to saved SHAP visualization.
        """
        # Step 1: Preprocess
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Step 2: Initialize SHAP explainer if not already
        if self.explainer is None:
            self._init_explainer()

        # Step 3: Run SHAP
        with torch.autograd.set_detect_anomaly(False):
            shap_values = self.explainer.shap_values(input_tensor)

        # Step 4: Plot
        np_img = np.array(image.resize((224, 224))) / 255.0
        shap_output_path = os.path.join(EXPLANATION_IMAGES_PATH, "shap_result.png")

        # Save SHAP plot
        try:
            shap.image_plot(shap_values, np.array([np_img]), show=False)
            plt.savefig(shap_output_path, bbox_inches="tight", dpi=150)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to save SHAP image: {e}")
        finally:
            plt.close()

        return shap_output_path
