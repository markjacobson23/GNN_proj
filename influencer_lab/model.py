from __future__ import annotations

import random

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

from influencer_lab.metrics import binary_classification_metrics


class GraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, dropout: float = 0.2):
        super().__init__()
        # A small two-layer GraphSAGE is enough for the demo graph.
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Mix neighbor information, then project down to one logit per node.
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)


class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, dropout: float = 0.2):
        super().__init__()
        # This mirrors GraphSAGE so the comparison stays fair.
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Same basic shape as GraphSAGE, just with GCN layers instead.
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)


def train_graphsage(
    model: GraphSAGE,
    data,
    *,
    epochs: int = 60,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    seed: int = 7,
) -> GraphSAGE:
    # Seed the run so the demo is reproducible.
    torch.manual_seed(seed)
    random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_mask = data.train_mask & data.user_mask
    y = data.y.float()

    # A small class imbalance adjustment keeps the loss from leaning too hard toward negatives.
    positives = y[train_mask].sum().item()
    negatives = train_mask.sum().item() - positives
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
    return model


def train_gcn(
    model: GCN,
    data,
    *,
    epochs: int = 60,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    seed: int = 7,
) -> GCN:
    # The GCN loop is intentionally spelled out instead of hidden behind a shared helper.
    torch.manual_seed(seed)
    random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_mask = data.train_mask & data.user_mask
    y = data.y.float()

    positives = y[train_mask].sum().item()
    negatives = train_mask.sum().item() - positives
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
    return model


def evaluate_graphsage(model: GraphSAGE, data, mask: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits)
    return binary_classification_metrics(data.y[mask], probs[mask])


def evaluate_gcn(model: GCN, data, mask: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits)
    return binary_classification_metrics(data.y[mask], probs[mask])
