from __future__ import annotations

import torch

from influencer_lab.graph import EdgeType, Graph, Node, NodeType

FEATURE_NAMES = (
    # One-hot node type indicators come first so the tensor is easy to inspect.
    "is_user",
    "is_topic",
    "is_community",
    # The rest are simple structural counts derived from the graph.
    "in_degree",
    "out_degree",
    "follower_count",
    "interaction_count",
    "topic_count",
    "community_bridge",
)


def build_node_features(graph: Graph) -> tuple[torch.Tensor, list[int]]:
    # Return both the tensor and the node id order so rows can be traced back later.
    node_ids = sorted(graph.nodes)
    rows: list[list[float]] = []

    for node_id in node_ids:
        node = graph.nodes[node_id]
        rows.append(_feature_row(graph, node))

    return torch.tensor(rows, dtype=torch.float32), node_ids


def _feature_row(graph: Graph, node: Node) -> list[float]:
    # Build one row per node using only graph structure and node metadata.
    is_user = float(node.node_type == NodeType.USER)
    is_topic = float(node.node_type == NodeType.TOPIC)
    is_community = float(node.node_type == NodeType.COMMUNITY)
    in_degree = float(graph.in_degree(node.id))
    out_degree = float(graph.out_degree(node.id))
    follower_count = float(_follower_count(graph, node))
    interaction_count = float(_interaction_count(graph, node))
    topic_count = float(_topic_count(graph, node))
    community_bridge = float(_community_bridge_indicator(graph, node))

    return [
        is_user,
        is_topic,
        is_community,
        in_degree,
        out_degree,
        follower_count,
        interaction_count,
        topic_count,
        community_bridge,
    ]


def _follower_count(graph: Graph, node: Node) -> int:
    # Followers are incoming follows from other user nodes.
    if node.node_type != NodeType.USER:
        return 0
    return sum(
        1
        for edge in graph.get_incoming_edges(node.id)
        if edge.edge_type == EdgeType.FOLLOWS and edge.source.node_type == NodeType.USER
    )


def _interaction_count(graph: Graph, node: Node) -> int:
    # Interaction count combines both directions because the graph stores them as directed edges.
    if node.node_type != NodeType.USER:
        return 0
    incoming = sum(
        1 for edge in graph.get_incoming_edges(node.id) if edge.edge_type == EdgeType.INTERACTION
    )
    outgoing = sum(
        1 for edge in graph.get_outgoing_edges(node.id) if edge.edge_type == EdgeType.INTERACTION
    )
    return incoming + outgoing


def _topic_count(graph: Graph, node: Node) -> int:
    # Topic count is just the number of user-to-topic edges leaving this user.
    if node.node_type != NodeType.USER:
        return 0
    return sum(
        1 for edge in graph.get_outgoing_edges(node.id) if edge.edge_type == EdgeType.USER_TOPIC
    )


def _community_bridge_indicator(graph: Graph, node: Node) -> int:
    # Mark a user as a bridge if their neighborhood spans more than one community.
    if node.node_type != NodeType.USER:
        return 0

    neighbor_communities: set[int] = set()
    for edge in graph.get_incoming_edges(node.id) + graph.get_outgoing_edges(node.id):
        if edge.edge_type not in (EdgeType.FOLLOWS, EdgeType.INTERACTION):
            continue
        other = edge.source if edge.target.id == node.id else edge.target
        if other.node_type == NodeType.USER:
            community_id = other.attributes.get("community_id")
            if isinstance(community_id, int):
                neighbor_communities.add(community_id)

    return int(len(neighbor_communities) > 1)
