# backend/xai/lime_explainer.py

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries


class LimeImageExplainerWrapper:
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.device = device

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.explainer = lime_image.LimeImageExplainer()

    def predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        Prediction function required by LIME.
        Input: images (N, H, W, C) in [0, 255]
        Output: probabilities (N, num_classes)
        """
        batch = []
        for img in images:
            img_pil = Image.fromarray(img.astype(np.uint8))
            tensor = self.transform(img_pil).unsqueeze(0)
            batch.append(tensor)

        batch_tensor = torch.cat(batch).to(self.device)
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def explain(self, image: Image.Image) -> np.ndarray:
        """
        Generates LIME explanation overlay.
        Returns a NumPy array with overlay image.
        """
        image_np = np.array(image.resize((224, 224)).convert("RGB"))

        explanation = self.explainer.explain_instance(
            image_np,
            classifier_fn=self.predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=1000
        )

        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            label=top_label,
            positive_only=True,
            hide_rest=False,
            num_features=10,
            min_weight=0.01
        )

        overlay = mark_boundaries(temp / 255.0, mask)
        return (overlay * 255).astype(np.uint8)
