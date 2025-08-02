# error_analysis.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import make_grid

def get_misclassified_samples(model, dataloader, class_names, device):
    """
    Returns misclassified images, their true & predicted labels.
    """
    model.eval()
    misclassified = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified.append({
                        'image': images[i].cpu(),
                        'true_label': class_names[labels[i]],
                        'pred_label': class_names[preds[i]]
                    })

    return misclassified


def plot_misclassified_samples(misclassified, max_samples=10):
    """
    Plots a grid of misclassified images with predicted and true labels.
    """
    if len(misclassified) == 0:
        print("✅ No misclassified samples found!")
        return

    print(f"📦 Total Misclassifications: {len(misclassified)}")
    samples_to_show = min(len(misclassified), max_samples)
    fig, axs = plt.subplots(1, samples_to_show, figsize=(4 * samples_to_show, 4))

    for i in range(samples_to_show):
        sample = misclassified[i]
        img = sample['image']
        img = img.numpy().transpose((1, 2, 0))
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) +
                      np.array([0.485, 0.456, 0.406]), 0, 1)

        if samples_to_show == 1:
            ax = axs
        else:
            ax = axs[i]

        ax.imshow(img)
        ax.set_title(f"Pred: {sample['pred_label']}\nTrue: {sample['true_label']}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def group_misclassifications(misclassified):
    """
    Groups misclassifications by (true_label → pred_label) and counts them.
    """
    error_groups = {}

    for sample in misclassified:
        key = f"{sample['true_label']} → {sample['pred_label']}"
        error_groups[key] = error_groups.get(key, 0) + 1

    sorted_errors = sorted(error_groups.items(), key=lambda x: x[1], reverse=True)

    print("📊 Top Confusions:")
    for err, count in sorted_errors[:10]:
        print(f"{err}: {count} times")

    return sorted_errors
