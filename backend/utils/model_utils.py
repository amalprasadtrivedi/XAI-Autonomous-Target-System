# backend/utils/model_utils.py

import torch

def remove_inplace_relu(model: torch.nn.Module):
    """
    Recursively replaces all inplace=True ReLU activations with inplace=False.
    This avoids gradient hook conflicts during explainability (e.g., SHAP, GradCAM).
    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.ReLU):
            setattr(model, name, torch.nn.ReLU(inplace=False))
        else:
            remove_inplace_relu(module)
