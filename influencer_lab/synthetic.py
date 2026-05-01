from __future__ import annotations

from dataclasses import dataclass
import random

from influencer_lab.graph import Edge, EdgeType, Graph, Node, NodeType


@dataclass(frozen=True, slots=True)
class InfluencerGraphConfig:
    # default setup values.
    user_count: int = 42
    topic_count: int = 5
    community_count: int = 4
    influencer_count: int = 8
    same_community_follows: int = 2
    cross_community_follows: int = 1
    interactions_per_user: int = 3
    topic_links_non_influencer: int = 1
    topic_links_influencer: int = 2


@dataclass(frozen=True, slots=True)
class InfluencerBenchmarkMetadata:
    seed: int
    user_ids: tuple[int, ...]
    influencer_ids: tuple[int, ...]
    topic_ids: tuple[int, ...]
    community_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class InfluencerBenchmark:
    graph: Graph
    labels: dict[int, int]
    metadata: InfluencerBenchmarkMetadata


class InfluencerGraphGenerator:
    """Build one deterministic synthetic social graph."""

    def __init__(self, config: InfluencerGraphConfig | None = None):
        # demo override.
        self.config = config or InfluencerGraphConfig()

    @classmethod
    def demo_preset(cls) -> "InfluencerGraphGenerator":
        # The demo preset is the default used by the CLI and tests.
        return cls(InfluencerGraphConfig())

    def generate(self, seed: int = 7) -> InfluencerBenchmark:
        # Use a local Random.
        rng = random.Random(seed)
        graph = Graph()
        builder = _Builder(graph)

        # Topics and communities are support nodes that give the graph extra structure.
        topic_nodes = [
            builder.add_node(NodeType.TOPIC, {"name": f"topic_{i}"})
            for i in range(self.config.topic_count)
        ]
        community_nodes = [
            builder.add_node(NodeType.COMMUNITY, {"name": f"community_{i}"})
            for i in range(self.config.community_count)
        ]

        users: list[Node] = []
        influencer_ids: set[int] = set()

        influencer_indices = _evenly_spaced_indices(
            self.config.user_count, self.config.influencer_count
        )

        # Create the user nodes and mark which ones are influencers.
        for index in range(self.config.user_count):
            community_id = index % self.config.community_count
            user = builder.add_node(
                NodeType.USER,
                {
                    "name": f"user_{index}",
                    "community_id": community_id,
                    "is_influencer": index in influencer_indices,
                },
            )
            users.append(user)
            if index in influencer_indices:
                influencer_ids.add(user.id)

        # Group influencers by community so the generator can create local and cross-community connections.
        influencer_by_community: dict[int, list[Node]] = {}
        for user in users:
            if user.id in influencer_ids:
                community_id = int(user.attributes["community_id"])
                influencer_by_community.setdefault(community_id, []).append(user)

        # Wire each user into the graph with the relationships the model will learn from.
        for user in users:
            community_id = int(user.attributes["community_id"])
            builder.add_edge(user, community_nodes[community_id], EdgeType.USER_COMMUNITY)

            topic_links = (
                self.config.topic_links_influencer
                if user.id in influencer_ids
                else self.config.topic_links_non_influencer
            )
            for topic_index in _cyclic_indices(user.id, topic_links, self.config.topic_count):
                builder.add_edge(user, topic_nodes[topic_index], EdgeType.USER_TOPIC)

            same_community_influencers = influencer_by_community.get(community_id, [])
            other_influencers = [
                influencer
                for influencer in users
                if influencer.id in influencer_ids
                and int(influencer.attributes["community_id"]) != community_id
            ]

            if user.id not in influencer_ids:
                # Non-influencers mostly point at influencers (ex: most people follow influencers).
                for target in _sample_without_replacement(
                    rng, same_community_influencers, self.config.same_community_follows
                ):
                    if target.id != user.id:
                        builder.add_edge(user, target, EdgeType.FOLLOWS)
                for target in _sample_without_replacement(
                    rng, other_influencers, self.config.cross_community_follows
                ):
                    if target.id != user.id:
                        builder.add_edge(user, target, EdgeType.FOLLOWS)
            else:
                bridge_targets = [
                    candidate
                    for candidate in users
                    if candidate.id not in influencer_ids
                    and int(candidate.attributes["community_id"]) != community_id
                ]
                for target in _sample_without_replacement(rng, bridge_targets, 2):
                    builder.add_edge(user, target, EdgeType.INTERACTION)

            # Everyone gets some interaction edges so the graph does not become trivial.
            interaction_pool = [
                candidate
                for candidate in users
                if candidate.id != user.id
                and int(candidate.attributes["community_id"]) == community_id
            ]
            if user.id in influencer_ids:
                interaction_pool = [
                    candidate for candidate in users if candidate.id != user.id
                ]

            for target in _sample_without_replacement(
                rng, interaction_pool, self.config.interactions_per_user
            ):
                builder.add_edge(user, target, EdgeType.INTERACTION)

        labels = {user.id: int(user.id in influencer_ids) for user in users}
        metadata = InfluencerBenchmarkMetadata(
            seed=seed,
            user_ids=tuple(user.id for user in users),
            influencer_ids=tuple(sorted(influencer_ids)),
            topic_ids=tuple(node.id for node in topic_nodes),
            community_ids=tuple(node.id for node in community_nodes),
        )
        return InfluencerBenchmark(graph=graph, labels=labels, metadata=metadata)


def build_demo_benchmark(seed: int = 7) -> InfluencerBenchmark:
    return InfluencerGraphGenerator.demo_preset().generate(seed)


class _Builder:
    def __init__(self, graph: Graph):
        # separate counters for node and edge ids while constructing the graph.
        self.graph = graph
        self._next_node_id = 1
        self._next_edge_id = 1

    def add_node(self, node_type: NodeType, attributes: dict[str, int | float | str | bool]) -> Node:
        # The builder owns node id assignment so the generator never touches it.
        node = Node(self._next_node_id, node_type, dict(attributes))
        self.graph.add_node(node)
        self._next_node_id += 1
        return node

    def add_edge(self, source: Node, target: Node, edge_type: EdgeType, weight: float = 1.0) -> Edge:
        # Edges are created with a id and then attached to the graph.
        edge = Edge(self._next_edge_id, source, target, edge_type, weight=weight)
        self.graph.add_edge(edge)
        self._next_edge_id += 1
        return edge


def _evenly_spaced_indices(total: int, count: int) -> set[int]:
    # This spreads influencer labels through the user list instead of clustering them randomly.
    if count <= 0:
        return set()
    if count >= total:
        return set(range(total))
    return {min(total - 1, (index * total) // count) for index in range(count)}


def _cyclic_indices(start: int, count: int, modulus: int) -> list[int]:
    return [((start + offset) % modulus) for offset in range(count)]


def _sample_without_replacement(
    rng: random.Random,
    items: list[Node],
    count: int,
) -> list[Node]:
    # Keep sampling deterministic and avoid duplicates inside each sampled set.
    if count <= 0 or not items:
        return []
    if count >= len(items):
        shuffled = list(items)
        rng.shuffle(shuffled)
        return shuffled
    return rng.sample(items, count)
