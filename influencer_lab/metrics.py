from __future__ import annotations

import torch


def binary_classification_metrics(y_true: torch.Tensor, y_score: torch.Tensor) -> dict[str, float]:
    # Threshold the scores at 0.5 and compute the four standard binary metrics.
    y_true = y_true.float()
    y_pred = (y_score >= 0.5).float()

    # Count the confusion-matrix entries directly.
    tp = float(((y_pred == 1) & (y_true == 1)).sum().item())
    tn = float(((y_pred == 0) & (y_true == 0)).sum().item())
    fp = float(((y_pred == 1) & (y_true == 0)).sum().item())
    fn = float(((y_pred == 0) & (y_true == 1)).sum().item())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
