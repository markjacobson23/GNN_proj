from __future__ import annotations

from influencer_lab import (
    GCN,
    GraphSAGE,
    build_demo_benchmark,
    evaluate_degree_baseline,
    evaluate_gcn,
    evaluate_graphsage,
    to_pyg_data,
    train_gcn,
    train_graphsage,
)


def main() -> None:
    # Build the demo benchmark, convert it to PyG, and train graphSAGE.
    benchmark = build_demo_benchmark(seed=7)
    data = to_pyg_data(benchmark)
    model = GraphSAGE(in_channels=data.x.size(1))
    gcn = GCN(in_channels=data.x.size(1))
    train_graphsage(model, data, epochs=60, seed=7)
    train_gcn(gcn, data, epochs=60, seed=7)
    # Evaluate only on USER nodes.
    model_metrics = evaluate_graphsage(model, data, data.test_mask & data.user_mask)
    gcn_metrics = evaluate_gcn(gcn, data, data.test_mask & data.user_mask)
    baseline_metrics = evaluate_degree_baseline(data)

    print("Influencer Labeling Demo")
    print(f"Nodes: {len(benchmark.graph.nodes)}")
    print(f"Edges: {len(benchmark.graph.edges)}")
    print(f"Influencers: {len(benchmark.metadata.influencer_ids)}")
    print(f"GraphSAGE: {model_metrics}")
    print(f"GCN: {gcn_metrics}")
    print(f"Degree: {baseline_metrics}")


if __name__ == "__main__":
    main()
