# backend/services/postprocessing.py

from typing import List, Dict, Tuple
import numpy as np


def apply_confidence_threshold(
    detections: List[Dict[str, any]],
    threshold: float = 0.5
) -> List[Dict[str, any]]:
    """
    Filters detections by confidence threshold
    """
    return [det for det in detections if det["confidence"] >= threshold]


def sort_detections_by_confidence(
    detections: List[Dict[str, any]],
    descending: bool = True
) -> List[Dict[str, any]]:
    """
    Sorts detections by confidence
    """
    return sorted(detections, key=lambda x: x["confidence"], reverse=descending)


def add_fp_fn_flags(
    detections: List[Dict[str, any]],
    operator_feedback: List[Dict[str, any]]
) -> List[Dict[str, any]]:
    """
    Match detections with operator feedback and tag false positives/negatives
    Each feedback should contain: {target_id, decision, label}
    """

    for det in detections:
        det["false_positive"] = False
        det["false_negative"] = False
        det_id = det.get("target_id")

        # Match feedback
        matched = next((fb for fb in operator_feedback if fb.get("target_id") == det_id), None)
        if matched:
            if matched["decision"] == "Reject":
                # If the model predicted something that was rejected → FP
                det["false_positive"] = True
            elif matched["decision"] == "Confirm" and matched["label"] != det["label"]:
                # If confirmed, but label doesn't match expected → FN
                det["false_negative"] = True

    return detections


def format_detection_output(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_indices: np.ndarray,
    class_names: List[str]
) -> List[Dict[str, any]]:
    """
    Converts raw arrays into structured detection dicts.
    """
    results = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        results.append({
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "label": class_names[int(class_indices[i])],
            "confidence": float(scores[i])
        })
    return results
