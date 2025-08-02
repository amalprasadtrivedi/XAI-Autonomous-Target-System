# backend/xai/grad_cam.py

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: str = "layer4", device: str = "cpu"):
        self.model = model.eval().to(device)
        self.device = device
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_module = dict([*self.model.named_modules()])[self.target_layer]
        target_module.register_forward_hook(self._forward_hook)
        target_module.register_backward_hook(self._backward_hook)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, image: Image.Image, target_class: int = None) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap for the given image and class.
        """
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)

        return cv2.resize(heatmap, (image.width, image.height))

    def overlay_heatmap(self, image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlays the Grad-CAM heatmap on the original image.
        Returns a NumPy array image.
        """
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        image_np = np.array(image.convert("RGB"))
        overlayed = cv2.addWeighted(heatmap_colored, alpha, image_np, 1 - alpha, 0)
        return overlayed
