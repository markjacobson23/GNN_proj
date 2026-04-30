from __future__ import annotations

import random

import torch
from torch_geometric.data import Data

from influencer_lab.features import FEATURE_NAMES, build_node_features
from influencer_lab.graph import NodeType
from influencer_lab.synthetic import InfluencerBenchmark


def to_pyg_data(benchmark: InfluencerBenchmark, seed: int | None = None) -> Data:
    # Convert the graph into the canonical PyG feature matrix and edge index form.
    x, node_ids = build_node_features(benchmark.graph)
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

    # Keep edges in id order so the conversion is deterministic.
    edge_pairs: list[tuple[int, int]] = []
    for edge in sorted(benchmark.graph.edges.values(), key=lambda item: item.id):
        edge_pairs.append((node_index[edge.source.id], node_index[edge.target.id]))

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    y = torch.full((len(node_ids),), -1, dtype=torch.long)

    user_mask = torch.zeros(len(node_ids), dtype=torch.bool)
    user_indices: list[int] = []
    labels: list[int] = []

    # Only USER nodes receive influencer labels.
    for idx, node_id in enumerate(node_ids):
        node = benchmark.graph.nodes[node_id]
        if node.node_type == NodeType.USER:
            user_mask[idx] = True
            user_indices.append(idx)
            label = benchmark.labels[node_id]
            y[idx] = label
            labels.append(label)

    split_seed = benchmark.metadata.seed if seed is None else seed
    # Build train/val/test masks on the user subset so evaluation stays on task.
    train_mask, val_mask, test_mask = _build_split_masks(
        user_indices,
        labels,
        split_seed,
        len(node_ids),
    )

    data = Data(x=x, edge_index=edge_index, y=y)
    data.user_mask = user_mask
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.node_ids = torch.tensor(node_ids, dtype=torch.long)
    data.feature_names = list(FEATURE_NAMES)
    return data


def _build_split_masks(
    user_indices: list[int],
    labels: list[int],
    seed: int,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # split by label so each split gets both positives and negatives when possible.
    rng = random.Random(seed)
    positives = [index for index, label in zip(user_indices, labels) if label == 1]
    negatives = [index for index, label in zip(user_indices, labels) if label == 0]

    rng.shuffle(positives)
    rng.shuffle(negatives)

    train_pos, val_pos, test_pos = _split_counts(len(positives))
    train_neg, val_neg, test_neg = _split_counts(len(negatives))

    train_indices = positives[:train_pos] + negatives[:train_neg]
    val_indices = positives[train_pos : train_pos + val_pos] + negatives[train_neg : train_neg + val_neg]
    test_indices = positives[train_pos + val_pos :] + negatives[train_neg + val_neg :]

    return (
        _mask_from_indices(train_indices, size),
        _mask_from_indices(val_indices, size),
        _mask_from_indices(test_indices, size),
    )


def _split_counts(count: int) -> tuple[int, int, int]:
    # Keep the split rule simple and deterministic, with small-count fallbacks.
    if count <= 2:
        if count == 1:
            return 1, 0, 0
        if count == 2:
            return 1, 0, 1
    train = max(1, int(count * 0.6))
    val = max(1, int(count * 0.2))
    test = count - train - val
    if test < 1:
        test = 1
        if train > val:
            train -= 1
        else:
            val -= 1
    return train, val, test


def _mask_from_indices(indices: list[int], size: int) -> torch.Tensor:
    # The output masks always match the full node count, even though only users are active.
    mask = torch.zeros(size, dtype=torch.bool)
    for index in indices:
        mask[index] = True
    return mask
