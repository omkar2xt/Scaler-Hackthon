"""Social graph and SIR-like spread simulation for MisinfoGuard-Env."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .config import CONFIG, MisinfoGuardConfig


@dataclass
class PostSpreadState:
    """Runtime state for a single post and its spread dynamics."""

    post_id: int
    is_false: bool
    claim_falsehood_score: float
    source_node: int
    source_trust: float
    virality_score: float
    age_steps: int = 0
    infected_nodes: set[int] = field(default_factory=set)
    recovered_nodes: set[int] = field(default_factory=set)

    @property
    def reached_nodes(self) -> set[int]:
        """Return all nodes that have ever seen this post."""

        return self.infected_nodes | self.recovered_nodes

    @property
    def spread_fraction(self) -> float:
        """Placeholder spread fraction default used before network normalization."""

        return 0.0


@dataclass
class SpreadStepStats:
    """Aggregate spread metrics computed each simulation step."""

    true_reach: int
    false_reach: int
    true_reach_fraction: float
    false_reach_fraction: float


class SocialNetworkSimulator:
    """Builds and simulates a social graph with per-post SIR diffusion."""

    def __init__(
        self,
        config: MisinfoGuardConfig = CONFIG,
        seed: int | None = None,
    ) -> None:
        """Initialize simulator with a config object and optional RNG seed."""

        self.config = config
        self._rng = np.random.default_rng(seed)
        self.graph = self._build_social_graph()

    def _build_social_graph(self) -> nx.Graph:
        """Create a Watts-Strogatz graph and assign edge trust weights."""

        env_cfg = self.config.environment
        net_cfg = self.config.network

        graph = nx.watts_strogatz_graph(
            n=env_cfg.node_count,
            k=net_cfg.watts_k_neighbors,
            p=net_cfg.watts_rewire_prob,
            seed=int(self._rng.integers(0, np.iinfo(np.int32).max)),
        )

        low, high = net_cfg.trust_weight_range
        for u, v in graph.edges():
            graph[u][v]["trust_weight"] = float(self._rng.uniform(low, high))

        return graph

    def create_episode_posts(self) -> list[PostSpreadState]:
        """Create a mixed set of true/false posts for a new episode."""

        env_cfg = self.config.environment
        net_cfg = self.config.network

        false_count = int(round(env_cfg.post_count * net_cfg.false_post_ratio))
        false_count = min(max(false_count, 0), env_cfg.post_count)
        true_count = env_cfg.post_count - false_count

        post_labels = [True] * false_count + [False] * true_count
        self._rng.shuffle(post_labels)

        posts: list[PostSpreadState] = []
        for post_id, is_false in enumerate(post_labels):
            post = PostSpreadState(
                post_id=post_id,
                is_false=is_false,
                claim_falsehood_score=self._sample_falsehood_score(is_false),
                source_node=int(self._rng.integers(0, env_cfg.node_count)),
                source_trust=float(self._rng.uniform(*net_cfg.source_trust_range)),
                virality_score=(
                    net_cfg.default_false_virality
                    if is_false
                    else net_cfg.default_true_virality
                ),
            )
            post.infected_nodes = self._seed_initial_infections(post.source_node)
            posts.append(post)

        return posts

    def _sample_falsehood_score(self, is_false: bool) -> float:
        """Sample claim-falsehood score from the configured interval."""

        net_cfg = self.config.network
        low, high = (
            net_cfg.false_falsehood_score_range
            if is_false
            else net_cfg.true_falsehood_score_range
        )
        return float(self._rng.uniform(low, high))

    def _seed_initial_infections(self, source_node: int) -> set[int]:
        """Seed the initial infected set around a source node."""

        net_cfg = self.config.network
        node_count = self.config.environment.node_count

        seed_count = int(
            self._rng.integers(net_cfg.initial_seed_min, net_cfg.initial_seed_max + 1)
        )
        if seed_count <= 1:
            return {source_node}

        candidates = list(range(node_count))
        candidates.remove(source_node)
        sample_size = min(seed_count - 1, len(candidates))
        sampled = set(self._rng.choice(candidates, size=sample_size, replace=False).tolist())
        sampled.add(source_node)
        return sampled

    def step_spread(self, posts: Sequence[PostSpreadState]) -> SpreadStepStats:
        """Run one SIR spread step for each post independently."""

        node_count = self.config.environment.node_count
        decay = self.config.network.virality_decay

        for post in posts:
            if not post.infected_nodes:
                post.age_steps += 1
                post.virality_score *= decay
                continue

            currently_infected = set(post.infected_nodes)
            new_infections: set[int] = set()

            for infected_node in currently_infected:
                for neighbor in self.graph.neighbors(infected_node):
                    if neighbor in currently_infected or neighbor in post.recovered_nodes:
                        continue

                    trust_weight = float(
                        self.graph[infected_node][neighbor].get("trust_weight", 0.0)
                    )
                    infection_prob = max(0.0, min(1.0, post.virality_score * trust_weight))
                    if self._rng.random() < infection_prob:
                        new_infections.add(neighbor)

            post.recovered_nodes.update(currently_infected)
            post.infected_nodes = new_infections
            post.age_steps += 1
            post.virality_score *= decay

        return self.compute_reach_stats(posts, node_count)

    def compute_reach_stats(
        self,
        posts: Sequence[PostSpreadState],
        node_count: int | None = None,
    ) -> SpreadStepStats:
        """Compute total true/false reach across the post set."""

        total_nodes = node_count if node_count is not None else self.config.environment.node_count

        true_reached: set[int] = set()
        false_reached: set[int] = set()

        for post in posts:
            if post.is_false:
                false_reached.update(post.reached_nodes)
            else:
                true_reached.update(post.reached_nodes)

        true_reach = len(true_reached)
        false_reach = len(false_reached)

        if total_nodes <= 0:
            return SpreadStepStats(
                true_reach=true_reach,
                false_reach=false_reach,
                true_reach_fraction=0.0,
                false_reach_fraction=0.0,
            )

        return SpreadStepStats(
            true_reach=true_reach,
            false_reach=false_reach,
            true_reach_fraction=true_reach / total_nodes,
            false_reach_fraction=false_reach / total_nodes,
        )

    def adjacency_matrix(self) -> np.ndarray:
        """Return weighted graph adjacency as a dense matrix."""

        return nx.to_numpy_array(self.graph, weight="trust_weight", dtype=np.float32)

    def two_hop_adjacency(self, focal_nodes: Iterable[int], hops: int = 2) -> np.ndarray:
        """Return adjacency restricted to nodes within k hops from focal nodes."""

        if hops < 0:
            raise ValueError("hops must be non-negative")

        visited: set[int] = set()
        frontier: set[int] = set(focal_nodes)

        for _ in range(hops + 1):
            visited.update(frontier)
            next_frontier: set[int] = set()
            for node in frontier:
                next_frontier.update(self.graph.neighbors(node))
            frontier = next_frontier - visited

        subgraph = self.graph.subgraph(sorted(visited)).copy()
        return nx.to_numpy_array(subgraph, weight="trust_weight", dtype=np.float32)


def build_post_feature_matrix(
    posts: Sequence[PostSpreadState],
    node_count: int,
) -> np.ndarray:
    """Build an (M, 5) matrix of post features required by the environment."""

    if node_count <= 0:
        raise ValueError("node_count must be positive")
    if not posts:
        return np.empty((0, 5), dtype=np.float32)

    features: list[list[float]] = []
    for post in posts:
        spread_pct = len(post.reached_nodes) / node_count
        features.append(
            [
                float(post.claim_falsehood_score),
                float(spread_pct),
                float(post.age_steps),
                float(post.virality_score),
                float(post.source_trust),
            ]
        )

    return np.asarray(features, dtype=np.float32)


def post_id_to_topology_node(post_id: int, node_count: int) -> int:
    """Map a post index deterministically to a node index."""

    if node_count <= 0:
        raise ValueError("node_count must be positive")
    return post_id % node_count
