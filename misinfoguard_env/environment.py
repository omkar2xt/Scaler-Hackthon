"""OpenEnv-compatible MisinfoGuard reinforcement learning environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import CONFIG, MisinfoGuardConfig
from network import PostSpreadState, SocialNetworkSimulator, build_post_feature_matrix
from rewards import compute_reward

try:
    from openenv import OpenEnvBase
except ImportError:  # pragma: no cover - fallback when OpenEnv is unavailable locally
    class OpenEnvBase:  # type: ignore[override]
        """Fallback OpenEnv base class for local development."""


@dataclass
class ActionResult:
    """Outcome of applying a defender action."""

    selected_post_id: int | None
    selected_node_id: int | None
    effective_action: int
    budget_spent: int
    caught_early: bool


class MisinfoGuardEnv(OpenEnvBase, gym.Env):
    """Multi-agent misinformation defense environment."""

    metadata = {"render_modes": []}

    def __init__(self, config: MisinfoGuardConfig = CONFIG, seed: int | None = None) -> None:
        """Initialize environment state, simulator, and spaces."""

        super().__init__()
        self.config = config
        self._rng = np.random.default_rng(seed)
        self.simulator = SocialNetworkSimulator(config=config, seed=seed)

        env_cfg = self.config.environment
        self.action_space = spaces.Discrete(env_cfg.action_space_size)
        self.observation_space = spaces.Dict(
            {
                "graph_adjacency": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(env_cfg.node_count, env_cfg.node_count),
                    dtype=np.float32,
                ),
                "post_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(env_cfg.post_count, 5),
                    dtype=np.float32,
                ),
                "agent_budget": spaces.Discrete(env_cfg.initial_budget + 1),
                "timestep": spaces.Discrete(env_cfg.max_steps + 1),
            }
        )

        self.timestep = 0
        self.remaining_budget = env_cfg.initial_budget
        self.posts: list[PostSpreadState] = []
        self.flagged_posts: set[int] = set()
        self.countered_posts: set[int] = set()
        self.amplified_posts: set[int] = set()
        self.quarantined_nodes: set[int] = set()
        self.true_posts_touched: set[int] = set()
        self.visible_nodes: set[int] = set()
        self.monitor_nodes: set[int] = set()
        self.spreader_state: dict[str, bool] = {"burst_fired": False}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset episode state and return first observation and info."""

        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.simulator = SocialNetworkSimulator(config=self.config, seed=seed)

        env_cfg = self.config.environment
        self.timestep = 0
        self.remaining_budget = env_cfg.initial_budget
        self.posts = self.simulator.create_episode_posts()

        self.flagged_posts.clear()
        self.countered_posts.clear()
        self.amplified_posts.clear()
        self.quarantined_nodes.clear()
        self.true_posts_touched.clear()
        self.spreader_state = {"burst_fired": False}

        monitor_seed_count = min(max(env_cfg.monitor_seed_count, 0), env_cfg.node_count)
        if monitor_seed_count == 0:
            self.monitor_nodes = set()
        else:
            self.monitor_nodes = set(
                self._rng.choice(
                    np.arange(env_cfg.node_count),
                    size=monitor_seed_count,
                    replace=False,
                ).tolist()
            )
        self.visible_nodes = self._expand_visible_nodes(self.monitor_nodes)

        return self._build_observation(), {}

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance the environment by one timestep using a discrete action."""

        action_idx = int(action)
        prev_stats = self.simulator.compute_reach_stats(self.posts)

        self._run_spreaders()
        action_result = self._apply_action(action_idx)

        self.timestep += 1
        curr_stats = self.simulator.step_spread(self.posts)

        true_posts_not_touched = self._count_true_posts_not_touched()
        true_posts_flagged = self._count_true_posts_flagged()

        reward_breakdown = compute_reward(
            prev_false_reach=prev_stats.false_reach,
            curr_false_reach=curr_stats.false_reach,
            caught_before_threshold=action_result.caught_early,
            true_posts_not_touched=true_posts_not_touched,
            true_posts_flagged=true_posts_flagged,
            quarantined_nodes=len(self.quarantined_nodes),
            total_nodes=self.config.environment.node_count,
            remaining_budget=self.remaining_budget,
            initial_budget=self.config.environment.initial_budget,
            config=self.config,
        )

        done = self._is_done()
        truncated = self._is_truncated()
        terminated = done
        observation = self._build_observation()

        info: dict[str, Any] = reward_breakdown.to_info_dict()
        info.update(
            {
                "selected_post_id": action_result.selected_post_id,
                "selected_node_id": action_result.selected_node_id,
                "effective_action": action_result.effective_action,
                "budget_spent": action_result.budget_spent,
                "false_reach": curr_stats.false_reach,
                "true_reach": curr_stats.true_reach,
                "false_reach_fraction": curr_stats.false_reach_fraction,
                "true_reach_fraction": curr_stats.true_reach_fraction,
                "censorship_rate": (
                    len(self.quarantined_nodes) / self.config.environment.node_count
                ),
            }
        )

        return observation, reward_breakdown.total_reward, terminated, truncated, info

    def state(self) -> dict[str, Any]:
        """Return a full serializable environment snapshot for debugging/grading."""

        return {
            "timestep": self.timestep,
            "remaining_budget": self.remaining_budget,
            "flagged_posts": sorted(self.flagged_posts),
            "countered_posts": sorted(self.countered_posts),
            "amplified_posts": sorted(self.amplified_posts),
            "quarantined_nodes": sorted(self.quarantined_nodes),
            "visible_nodes": sorted(self.visible_nodes),
            "posts": [
                {
                    "post_id": post.post_id,
                    "is_false": post.is_false,
                    "claim_falsehood_score": post.claim_falsehood_score,
                    "source_node": post.source_node,
                    "source_trust": post.source_trust,
                    "virality_score": post.virality_score,
                    "age_steps": post.age_steps,
                    "infected_nodes": sorted(post.infected_nodes),
                    "recovered_nodes": sorted(post.recovered_nodes),
                }
                for post in self.posts
            ],
        }

    def _run_spreaders(self) -> None:
        """Execute scripted adversarial spreaders if available."""

        try:
            from agents.spreader import run_spreaders

            self.spreader_state = run_spreaders(
                timestep=self.timestep,
                posts=self.posts,
                graph=self.simulator.graph,
                quarantined_nodes=self.quarantined_nodes,
                config=self.config,
                rng=self._rng,
                state=self.spreader_state,
            )
        except ImportError:
            return

    def _apply_action(self, action_idx: int) -> ActionResult:
        """Apply one defender action and mutate environment state."""

        action_cost = self.config.environment.action_costs.get(action_idx, 0)
        if action_cost > self.remaining_budget:
            return ActionResult(
                selected_post_id=None,
                selected_node_id=None,
                effective_action=4,
                budget_spent=0,
                caught_early=False,
            )

        if action_idx == 0:
            result = self._flag_action(action_cost)
        elif action_idx == 1:
            result = self._counter_action(action_cost)
        elif action_idx == 2:
            result = self._amplify_action(action_cost)
        elif action_idx == 3:
            result = self._quarantine_action(action_cost)
        else:
            result = ActionResult(
                selected_post_id=None,
                selected_node_id=None,
                effective_action=4,
                budget_spent=0,
                caught_early=False,
            )

        if result.budget_spent > 0:
            self.remaining_budget -= result.budget_spent
        return result

    def _flag_action(self, action_cost: int) -> ActionResult:
        """Apply flag intervention to the highest-risk false post."""

        target = self._select_false_post_target()
        if target is None:
            return ActionResult(None, None, 4, 0, False)

        post = self.posts[target]
        post.virality_score = max(
            self.config.network.virality_floor,
            post.virality_score * self.config.network.flag_virality_scale,
        )
        self.flagged_posts.add(target)
        self._mark_touch_if_true(post)
        self._update_visibility_from_post(post)

        caught_early = (
            len(post.reached_nodes) / self.config.environment.node_count
            < self.config.reward.early_catch_threshold
        )
        return ActionResult(target, None, 0, action_cost, caught_early)

    def _counter_action(self, action_cost: int) -> ActionResult:
        """Deploy counter-narrative against a high-spread false post."""

        target = self._select_false_post_target()
        if target is None:
            return ActionResult(None, None, 4, 0, False)

        post = self.posts[target]
        post.virality_score = max(
            self.config.network.virality_floor,
            post.virality_score * self.config.network.counter_false_virality_scale,
        )
        self.countered_posts.add(target)
        self._mark_touch_if_true(post)
        self._update_visibility_from_post(post)

        return ActionResult(target, None, 1, action_cost, False)

    def _amplify_action(self, action_cost: int) -> ActionResult:
        """Boost the strongest true post as a competing narrative."""

        target = self._select_true_post_target()
        if target is None:
            return ActionResult(None, None, 4, 0, False)

        post = self.posts[target]
        post.virality_score = min(
            self.config.network.virality_cap,
            post.virality_score * self.config.network.amplify_true_virality_scale,
        )
        self.amplified_posts.add(target)
        self._mark_touch_if_true(post)
        self._update_visibility_from_post(post)

        return ActionResult(target, None, 2, action_cost, False)

    def _quarantine_action(self, action_cost: int) -> ActionResult:
        """Quarantine a high-degree node by reducing incident trust weights."""

        target_node = self._select_quarantine_node()
        if target_node is None:
            return ActionResult(None, None, 4, 0, False)

        scale = self.config.network.quarantine_trust_scale
        for neighbor in self.simulator.graph.neighbors(target_node):
            edge = self.simulator.graph[target_node][neighbor]
            edge["trust_weight"] = float(edge.get("trust_weight", 0.0) * scale)

        self.quarantined_nodes.add(target_node)
        self.monitor_nodes.add(target_node)
        self.visible_nodes = self._expand_visible_nodes(self.monitor_nodes)

        return ActionResult(None, target_node, 3, action_cost, False)

    def _select_false_post_target(self) -> int | None:
        """Return false post index with highest risk score."""

        false_candidates = [p for p in self.posts if p.is_false]
        if not false_candidates:
            return None

        target = max(
            false_candidates,
            key=lambda p: p.claim_falsehood_score * len(p.reached_nodes),
        )
        return target.post_id

    def _select_true_post_target(self) -> int | None:
        """Return true post index with highest spread potential."""

        true_candidates = [p for p in self.posts if not p.is_false]
        if not true_candidates:
            return None

        target = max(true_candidates, key=lambda p: p.virality_score + len(p.reached_nodes))
        return target.post_id

    def _select_quarantine_node(self) -> int | None:
        """Choose non-quarantined node with maximum degree."""

        candidates = [
            node
            for node in self.simulator.graph.nodes
            if node not in self.quarantined_nodes
        ]
        if not candidates:
            return None

        return max(candidates, key=self.simulator.graph.degree)

    def _mark_touch_if_true(self, post: PostSpreadState) -> None:
        """Track interventions that touched true posts for reward accounting."""

        if not post.is_false:
            self.true_posts_touched.add(post.post_id)

    def _count_true_posts_not_touched(self) -> int:
        """Count true posts untouched by defender interventions."""

        true_posts = [post.post_id for post in self.posts if not post.is_false]
        untouched = [post_id for post_id in true_posts if post_id not in self.true_posts_touched]
        return len(untouched)

    def _count_true_posts_flagged(self) -> int:
        """Count number of true posts that were flagged."""

        true_posts = {post.post_id for post in self.posts if not post.is_false}
        return len(self.flagged_posts & true_posts)

    def _is_done(self) -> bool:
        """Return whether episode reached a terminal objective state."""

        if self.timestep == 0:
            return False
        return self.simulator.compute_reach_stats(self.posts).false_reach == 0

    def _is_truncated(self) -> bool:
        """Return whether episode ended due to budget/time constraints."""

        return (
            self.timestep >= self.config.environment.max_steps
            or self.remaining_budget <= 0
        )

    def _expand_visible_nodes(self, seeds: set[int]) -> set[int]:
        """Expand visibility from seed nodes to configured hop distance."""

        visible = set(seeds)
        frontier = set(seeds)

        for _ in range(self.config.environment.observation_hops):
            next_frontier: set[int] = set()
            for node in frontier:
                next_frontier.update(self.simulator.graph.neighbors(node))
            next_frontier -= visible
            visible.update(next_frontier)
            frontier = next_frontier

        return visible

    def _update_visibility_from_post(self, post: PostSpreadState) -> None:
        """Update visible region based on the touched post's diffusion footprint."""

        new_seed_nodes = set(post.reached_nodes)
        new_seed_nodes.add(post.source_node)
        self.monitor_nodes.update(new_seed_nodes)
        self.visible_nodes = self._expand_visible_nodes(self.monitor_nodes)

    def _build_observation(self) -> dict[str, Any]:
        """Build partially observable observation dictionary."""

        adjacency = self.simulator.adjacency_matrix()
        node_mask = np.zeros(self.config.environment.node_count, dtype=np.float32)
        node_mask[list(self.visible_nodes)] = 1.0
        visibility_matrix = np.outer(node_mask, node_mask)
        masked_adjacency = adjacency * visibility_matrix

        full_post_features = build_post_feature_matrix(
            self.posts,
            self.config.environment.node_count,
        )

        post_visibility = np.zeros((len(self.posts), 1), dtype=np.float32)
        for idx, post in enumerate(self.posts):
            if post.source_node in self.visible_nodes or post.reached_nodes & self.visible_nodes:
                post_visibility[idx, 0] = 1.0

        post_features = full_post_features * post_visibility

        return {
            "graph_adjacency": masked_adjacency.astype(np.float32),
            "post_features": post_features.astype(np.float32),
            "agent_budget": int(self.remaining_budget),
            "timestep": int(self.timestep),
        }
