"""Synthetic influencer-labeling benchmark built for PyTorch Geometric."""

# Re-export the small public surface so users can import from one place.
from influencer_lab.baselines import evaluate_degree_baseline
from influencer_lab.graph import Edge, EdgeType, Graph, Node, NodeType
from influencer_lab.model import GCN, GraphSAGE, evaluate_gcn, evaluate_graphsage, train_gcn, train_graphsage
from influencer_lab.pyg import to_pyg_data
from influencer_lab.synthetic import (
    InfluencerBenchmark,
    InfluencerBenchmarkMetadata,
    InfluencerGraphConfig,
    InfluencerGraphGenerator,
    build_demo_benchmark,
)

__all__ = [
    "Edge",
    "EdgeType",
    "GCN",
    "Graph",
    "GraphSAGE",
    "InfluencerBenchmark",
    "InfluencerBenchmarkMetadata",
    "InfluencerGraphConfig",
    "InfluencerGraphGenerator",
    "Node",
    "NodeType",
    "build_demo_benchmark",
    "evaluate_degree_baseline",
    "evaluate_gcn",
    "evaluate_graphsage",
    "to_pyg_data",
    "train_gcn",
    "train_graphsage",
]
