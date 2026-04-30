from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from influencer_lab import (  # noqa: E402
    GCN,
    GraphSAGE,
    NodeType,
    build_demo_benchmark,
    to_pyg_data,
    train_gcn,
    train_graphsage,
)
from influencer_lab.features import FEATURE_NAMES  # noqa: E402


OUTPUT = ROOT / "docs" / "influencer_comparison.png"


def main() -> None:
    benchmark = build_demo_benchmark(seed=7)
    data = to_pyg_data(benchmark)
    node_index = {int(node_id): idx for idx, node_id in enumerate(data.node_ids.tolist())}

    # Train the tiny models once so we can compare their predictions.
    model = GraphSAGE(in_channels=data.x.size(1))
    gcn = GCN(in_channels=data.x.size(1))
    train_graphsage(model, data, epochs=60, seed=7)
    train_gcn(gcn, data, epochs=60, seed=7)

    # Convert the benchmark graph into a NetworkX graph for plotting.
    graph = nx.DiGraph()
    for node_id, node in benchmark.graph.nodes.items():
        graph.add_node(node_id, node_type=node.node_type.name, label=benchmark.labels.get(node_id, -1))
    for edge in benchmark.graph.edges.values():
        graph.add_edge(edge.source.id, edge.target.id, edge_type=edge.edge_type.name)

    layout = nx.spring_layout(graph, seed=7)

    # Build prediction maps for the two approaches we want to compare.
    follower_index = FEATURE_NAMES.index("follower_count")
    degree_scores = data.x[:, follower_index]
    degree_threshold = float(degree_scores[data.train_mask & data.user_mask].median().item())
    degree_pred = (degree_scores >= degree_threshold).long()

    model.eval()
    with torch.no_grad():
        model_probs = torch.sigmoid(model(data.x, data.edge_index))
    model_pred = (model_probs >= 0.5).long()
    gcn.eval()
    with torch.no_grad():
        gcn_probs = torch.sigmoid(gcn(data.x, data.edge_index))
    gcn_pred = (gcn_probs >= 0.5).long()

    # Draw the graph three times with the same layout so the comparison stays easy to read.
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), constrained_layout=True)
    _draw_panel(
        axes[0],
        graph,
        layout,
        benchmark,
        node_index,
        degree_pred,
        "Degree baseline",
    )
    _draw_panel(
        axes[1],
        graph,
        layout,
        benchmark,
        node_index,
        model_pred,
        "GraphSAGE",
    )
    _draw_panel(
        axes[2],
        graph,
        layout,
        benchmark,
        node_index,
        gcn_pred,
        "GCN",
    )

    fig.suptitle("Influencer labeling: baseline misses vs GraphSAGE and GCN", fontsize=16)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220)
    print(f"Wrote {OUTPUT}")


def _draw_panel(ax, graph, layout, benchmark, node_index, pred, title: str) -> None:
    true_labels = benchmark.labels
    user_nodes = [node_id for node_id, node in benchmark.graph.nodes.items() if node.node_type == NodeType.USER]
    topic_nodes = [node_id for node_id, node in benchmark.graph.nodes.items() if node.node_type == NodeType.TOPIC]
    community_nodes = [
        node_id for node_id, node in benchmark.graph.nodes.items() if node.node_type == NodeType.COMMUNITY
    ]

    # Light gray edges keep the structure visible without stealing attention from the nodes.
    nx.draw_networkx_edges(graph, layout, ax=ax, alpha=0.18, width=0.8, arrows=False)

    # Support nodes are drawn first so the user layer stays visually dominant.
    nx.draw_networkx_nodes(
        graph,
        layout,
        nodelist=topic_nodes,
        node_shape="D",
        node_color="#D9D9D9",
        node_size=260,
        linewidths=0.8,
        edgecolors="#8C8C8C",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        layout,
        nodelist=community_nodes,
        node_shape="s",
        node_color="#E6E6E6",
        node_size=280,
        linewidths=0.8,
        edgecolors="#9E9E9E",
        ax=ax,
    )

    # User nodes are colored by their true label so the eye can compare truth to prediction.
    influencer_nodes = [node_id for node_id in user_nodes if true_labels[node_id] == 1]
    normal_nodes = [node_id for node_id in user_nodes if true_labels[node_id] == 0]
    nx.draw_networkx_nodes(
        graph,
        layout,
        nodelist=normal_nodes,
        node_shape="o",
        node_color="#7FA8C9",
        node_size=360,
        linewidths=0.8,
        edgecolors="#355C7D",
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        layout,
        nodelist=influencer_nodes,
        node_shape="o",
        node_color="#F2A65A",
        node_size=420,
        linewidths=0.8,
        edgecolors="#9E5B16",
        ax=ax,
    )

    # Overlay a border that marks where the approach got the node wrong.
    false_pos = [
        node_id
        for node_id in user_nodes
        if pred[node_index[node_id]].item() == 1 and true_labels[node_id] == 0
    ]
    false_neg = [
        node_id
        for node_id in user_nodes
        if pred[node_index[node_id]].item() == 0 and true_labels[node_id] == 1
    ]
    if false_pos:
        nx.draw_networkx_nodes(
            graph,
            layout,
            nodelist=false_pos,
            node_shape="o",
            node_color="none",
            node_size=470,
            linewidths=2.5,
            edgecolors="#C00000",
            ax=ax,
        )
    if false_neg:
        nx.draw_networkx_nodes(
            graph,
            layout,
            nodelist=false_neg,
            node_shape="o",
            node_color="none",
            node_size=470,
            linewidths=2.5,
            edgecolors="#C00000",
            ax=ax,
        )

    # A small label on each user node keeps the figure readable without crowding it.
    labels = {
        node_id: f"U{node_id}"
        if benchmark.graph.nodes[node_id].node_type == NodeType.USER
        else ("T" if benchmark.graph.nodes[node_id].node_type == NodeType.TOPIC else "C")
        for node_id in graph.nodes
    }
    nx.draw_networkx_labels(graph, layout, labels=labels, font_size=8, ax=ax)

    ax.set_title(title, fontsize=14)
    ax.axis("off")

    total = len(user_nodes)
    correct = sum(
        1 for node_id in user_nodes if int(pred[node_index[node_id]].item()) == int(true_labels[node_id])
    )
    ax.text(
        0.02,
        0.02,
        f"USER accuracy: {correct}/{total}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#CCCCCC"),
    )


if __name__ == "__main__":
    main()
