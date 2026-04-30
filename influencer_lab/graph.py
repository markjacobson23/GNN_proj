from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class NodeType(Enum):
    USER = auto()
    TOPIC = auto()
    COMMUNITY = auto()


class EdgeType(Enum):
    FOLLOWS = auto()
    INTERACTION = auto()
    USER_TOPIC = auto()
    USER_COMMUNITY = auto()


@dataclass(slots=True)
class Node:
    id: int
    node_type: NodeType
    attributes: dict[str, int | float | str | bool] = field(default_factory=dict)


@dataclass(slots=True)
class Edge:
    id: int
    source: Node
    target: Node
    edge_type: EdgeType
    weight: float = 1.0
    attributes: dict[str, int | float | str | bool] = field(default_factory=dict)


class Graph:
    """A tiny directed multi-edge graph for one synthetic social network."""

    def __init__(self) -> None:
        # Store the full node and edge sets by id for quick lookup.
        self.nodes: dict[int, Node] = {}
        self.edges: dict[int, Edge] = {}
        # Keep adjacency lists so feature code and PyG conversion can walk the graph fast.
        self._outgoing: dict[int, list[Edge]] = {}
        self._incoming: dict[int, list[Edge]] = {}

    def add_node(self, node: Node) -> None:
        # Node ids are unique and must not be reused.
        if node.id in self.nodes:
            raise ValueError(f"Node-{node.id} already exists.")
        self.nodes[node.id] = node
        self._outgoing[node.id] = []
        self._incoming[node.id] = []

    def add_edge(self, edge: Edge) -> None:
        # Edges are also unique by id and must connect nodes already in the graph.
        if edge.id in self.edges:
            raise ValueError(f"Edge-{edge.id} already exists.")
        if edge.source.id not in self.nodes:
            raise ValueError(f"Node-{edge.source.id} must already exist.")
        if edge.target.id not in self.nodes:
            raise ValueError(f"Node-{edge.target.id} must already exist.")
        if edge.source.id == edge.target.id:
            raise ValueError("Self edges are not allowed.")

        self.edges[edge.id] = edge
        self._outgoing[edge.source.id].append(edge)
        self._incoming[edge.target.id].append(edge)

    def get_outgoing_edges(self, node_id: int) -> list[Edge]:
        # Return a copy so callers do not mutate the graph internals by accident.
        return list(self._outgoing[node_id])

    def get_incoming_edges(self, node_id: int) -> list[Edge]:
        # Incoming edges are used for follower counts and label-style features.
        return list(self._incoming[node_id])

    def out_degree(self, node_id: int) -> int:
        return len(self._outgoing[node_id])

    def in_degree(self, node_id: int) -> int:
        return len(self._incoming[node_id])

    def degree(self, node_id: int) -> int:
        return self.in_degree(node_id) + self.out_degree(node_id)
