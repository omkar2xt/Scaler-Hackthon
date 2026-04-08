"""Scripted adversarial spreaders for MisinfoGuard-Env."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np

try:
    from ..config import CONFIG, MisinfoGuardConfig
    from ..network import PostSpreadState
except ImportError:
    from config import CONFIG, MisinfoGuardConfig
    from network import PostSpreadState


@dataclass
class BaseSpreaderBot:
    """Base class for scripted spreader behavior."""

    name: str

    def act(
        self,
        timestep: int,
        posts: Sequence[PostSpreadState],
        graph: nx.Graph,
        quarantined_nodes: set[int],
        config: MisinfoGuardConfig,
        rng: np.random.Generator,
        state: dict[str, bool],
    ) -> None:
        """Apply bot intervention for the current timestep."""

        raise NotImplementedError


@dataclass
class FloodBot(BaseSpreaderBot):
    """Posts many low-virality false items every timestep."""

    name: str = "FloodBot"

    def act(
        self,
        timestep: int,
        posts: Sequence[PostSpreadState],
        graph: nx.Graph,
        quarantined_nodes: set[int],
        config: MisinfoGuardConfig,
        rng: np.random.Generator,
        state: dict[str, bool],
    ) -> None:
        """Inject low-virality false spread into random nodes."""

        false_posts = [post for post in posts if post.is_false]
        if not false_posts:
            return

        node_pool = [node for node in graph.nodes if node not in quarantined_nodes]
        if not node_pool:
            return

        injection_count = min(
            config.spreader.floodbot_injections_per_step,
            len(false_posts),
        )

        selected_posts = rng.choice(false_posts, size=injection_count, replace=False)
        for post in selected_posts:
            target_node = int(rng.choice(node_pool))
            post.infected_nodes.add(target_node)
            post.virality_score = max(post.virality_score, config.spreader.injected_false_virality)


@dataclass
class TargetBot(BaseSpreaderBot):
    """Injects false content into the highest-degree node."""

    name: str = "TargetBot"

    def act(
        self,
        timestep: int,
        posts: Sequence[PostSpreadState],
        graph: nx.Graph,
        quarantined_nodes: set[int],
        config: MisinfoGuardConfig,
        rng: np.random.Generator,
        state: dict[str, bool],
    ) -> None:
        """Target graph hubs to maximize early diffusion."""

        false_posts = [post for post in posts if post.is_false]
        if not false_posts:
            return

        candidates = [node for node in graph.nodes if node not in quarantined_nodes]
        if not candidates:
            return

        hub_node = max(candidates, key=graph.degree)
        injections = min(config.spreader.targetbot_injections_per_step, len(false_posts))

        selected_posts = rng.choice(false_posts, size=injections, replace=False)
        for post in selected_posts:
            post.infected_nodes.add(hub_node)
            post.virality_score = max(post.virality_score, config.spreader.injected_false_virality)


@dataclass
class BurstBot(BaseSpreaderBot):
    """Waits for a delay and then executes a one-time flood burst."""

    name: str = "BurstBot"

    def act(
        self,
        timestep: int,
        posts: Sequence[PostSpreadState],
        graph: nx.Graph,
        quarantined_nodes: set[int],
        config: MisinfoGuardConfig,
        rng: np.random.Generator,
        state: dict[str, bool],
    ) -> None:
        """Trigger synchronized false-post injections after delay."""

        if timestep < config.spreader.burstbot_delay_steps:
            return
        if state.get("burst_fired", False):
            return

        false_posts = [post for post in posts if post.is_false]
        if not false_posts:
            return

        node_pool = [node for node in graph.nodes if node not in quarantined_nodes]
        if not node_pool:
            return

        injections = min(config.spreader.burstbot_injection_count, len(false_posts))
        selected_posts = rng.choice(false_posts, size=injections, replace=False)
        selected_nodes = rng.choice(node_pool, size=injections, replace=True)

        for post, node in zip(selected_posts, selected_nodes):
            post.infected_nodes.add(int(node))
            post.virality_score = max(post.virality_score, config.spreader.injected_false_virality)

        state["burst_fired"] = True


def run_spreaders(
    timestep: int,
    posts: Sequence[PostSpreadState],
    graph: nx.Graph,
    quarantined_nodes: set[int],
    config: MisinfoGuardConfig = CONFIG,
    rng: np.random.Generator | None = None,
    state: dict[str, bool] | None = None,
    seed: int | None = None,
) -> dict[str, bool]:
    """Run all scripted spreaders in fixed order each timestep."""

    if rng is None:
        if seed is None:
            warnings.warn(
                "run_spreaders called without rng or seed; using non-deterministic RNG.",
                RuntimeWarning,
                stacklevel=2,
            )
        rng = np.random.default_rng(seed)

    if state is None:
        state = {"burst_fired": False}
    if timestep == 0:
        state["burst_fired"] = False

    bots: list[BaseSpreaderBot] = [FloodBot(), TargetBot(), BurstBot()]

    for bot in bots:
        bot.act(
            timestep=timestep,
            posts=posts,
            graph=graph,
            quarantined_nodes=quarantined_nodes,
            config=config,
            rng=rng,
            state=state,
        )

    return state
