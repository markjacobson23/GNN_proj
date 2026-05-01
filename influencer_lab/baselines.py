from __future__ import annotations

import torch

from influencer_lab.features import FEATURE_NAMES
from influencer_lab.metrics import binary_classification_metrics


def evaluate_degree_baseline(data) -> dict[str, float]:
    # Uses follower_count as the influencer score.
    follower_index = FEATURE_NAMES.index("follower_count")
    scores = data.x[:, follower_index]
    # threshold chosen from the training split so the baseline stays fair.
    threshold = _threshold_from_train_scores(scores[data.train_mask & data.user_mask])
    predictions = (scores >= threshold).float()
    return binary_classification_metrics(data.y[data.test_mask & data.user_mask], predictions[data.test_mask & data.user_mask])


def _threshold_from_train_scores(train_scores: torch.Tensor) -> float:
    # Median is a simple robust cutoff for.
    if train_scores.numel() == 0:
        return 0.0
    return float(train_scores.median().item())
