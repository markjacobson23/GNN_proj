from __future__ import annotations

import torch

from influencer_lab import (
    GCN,
    GraphSAGE,
    NodeType,
    build_demo_benchmark,
    evaluate_degree_baseline,
    evaluate_gcn,
    evaluate_graphsage,
    to_pyg_data,
    train_gcn,
    train_graphsage,
)
from influencer_lab.features import FEATURE_NAMES, build_node_features


def _graph_signature(benchmark) -> tuple[tuple, tuple]:
    # Reduce the graph to a deterministic tuple.
    nodes = tuple(
        sorted(
            (
                node.id,
                node.node_type.name,
                tuple(sorted(node.attributes.items())),
            )
            for node in benchmark.graph.nodes.values()
        )
    )
    edges = tuple(
        sorted(
            (
                edge.id,
                edge.source.id,
                edge.target.id,
                edge.edge_type.name,
                edge.weight,
                tuple(sorted(edge.attributes.items())),
            )
            for edge in benchmark.graph.edges.values()
        )
    )
    return nodes, edges


def test_demo_benchmark_is_deterministic() -> None:
    # The same seed should rebuild the exact same benchmark.
    benchmark_a = build_demo_benchmark(seed=7)
    benchmark_b = build_demo_benchmark(seed=7)

    assert _graph_signature(benchmark_a) == _graph_signature(benchmark_b)
    assert benchmark_a.labels == benchmark_b.labels
    assert benchmark_a.metadata == benchmark_b.metadata


def test_pyg_data_contract_and_masks_are_stable() -> None:
    # The PyG object should be stable and should keep the task limited to users.
    benchmark = build_demo_benchmark(seed=7)
    data_a = to_pyg_data(benchmark, seed=13)
    data_b = to_pyg_data(benchmark, seed=13)

    assert torch.equal(data_a.x, data_b.x)
    assert torch.equal(data_a.edge_index, data_b.edge_index)
    assert torch.equal(data_a.y, data_b.y)
    assert torch.equal(data_a.train_mask, data_b.train_mask)
    assert torch.equal(data_a.val_mask, data_b.val_mask)
    assert torch.equal(data_a.test_mask, data_b.test_mask)
    assert torch.equal(data_a.user_mask, data_b.user_mask)

    assert data_a.x.shape[0] == len(benchmark.graph.nodes)
    assert data_a.y.shape[0] == len(benchmark.graph.nodes)
    assert data_a.edge_index.shape[0] == 2
    assert list(data_a.feature_names) == list(FEATURE_NAMES)

    assert torch.logical_and(data_a.train_mask, ~data_a.user_mask).sum().item() == 0
    assert torch.logical_and(data_a.val_mask, ~data_a.user_mask).sum().item() == 0
    assert torch.logical_and(data_a.test_mask, ~data_a.user_mask).sum().item() == 0
    assert torch.equal(
        data_a.train_mask | data_a.val_mask | data_a.test_mask,
        data_a.user_mask,
    )


def test_feature_table_keeps_influencer_signal_simple() -> None:
    # Influencers are wired to have stronger more followers than normal users.
    benchmark = build_demo_benchmark(seed=7)
    features, node_ids = build_node_features(benchmark.graph)
    follower_idx = FEATURE_NAMES.index("follower_count")

    influencer_scores = []
    normal_scores = []
    for row_index, node_id in enumerate(node_ids):
        if benchmark.graph.nodes[node_id].node_type != NodeType.USER:
            continue
        score = float(features[row_index, follower_idx].item())
        if benchmark.labels[node_id] == 1:
            influencer_scores.append(score)
        else:
            normal_scores.append(score)

    assert sum(influencer_scores) / len(influencer_scores) > sum(normal_scores) / len(normal_scores)


def test_graphsage_and_degree_baseline_smoke_test() -> None:
    # Make sure both the learned model and the simple baseline run completely.
    benchmark = build_demo_benchmark(seed=7)
    data = to_pyg_data(benchmark)
    model = GraphSAGE(in_channels=data.x.size(1), hidden_channels=16, dropout=0.1)
    gcn = GCN(in_channels=data.x.size(1), hidden_channels=16, dropout=0.1)

    train_graphsage(model, data, epochs=25, seed=7)
    train_gcn(gcn, data, epochs=25, seed=7)
    model_metrics = evaluate_graphsage(model, data, data.test_mask & data.user_mask)
    gcn_metrics = evaluate_gcn(gcn, data, data.test_mask & data.user_mask)
    baseline_metrics = evaluate_degree_baseline(data)

    for metrics in (model_metrics, gcn_metrics, baseline_metrics):
        assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
        for value in metrics.values():
            assert 0.0 <= value <= 1.0
