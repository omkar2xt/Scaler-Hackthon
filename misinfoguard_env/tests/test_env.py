"""Unit tests for MisinfoGuardEnv core API methods."""

from __future__ import annotations

from config import CONFIG
from environment import MisinfoGuardEnv


def test_reset_returns_expected_observation_schema() -> None:
    """Reset should return expected keys and fixed tensor shapes."""

    env = MisinfoGuardEnv(config=CONFIG, seed=0)
    observation, info = env.reset()

    assert set(observation.keys()) == {
        "graph_adjacency",
        "post_features",
        "agent_budget",
        "timestep",
    }
    assert observation["graph_adjacency"].shape == (
        CONFIG.environment.node_count,
        CONFIG.environment.node_count,
    )
    assert observation["post_features"].shape == (CONFIG.environment.post_count, 5)
    assert int(observation["agent_budget"]) == CONFIG.environment.initial_budget
    assert int(observation["timestep"]) == 0
    assert isinstance(info, dict)


def test_step_returns_expected_tuple_and_info_components() -> None:
    """Step should produce Gymnasium 5-tuple with reward breakdown."""

    env = MisinfoGuardEnv(config=CONFIG, seed=1)
    env.reset()

    observation, reward, terminated, truncated, info = env.step(4)

    assert isinstance(observation, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    required_reward_keys = {
        "r_spread_reduction",
        "r_early_bonus",
        "r_true_post_safe",
        "r_false_positive_penalty",
        "r_overcensorship_penalty",
        "r_budget_efficiency",
        "total_reward",
    }
    assert required_reward_keys.issubset(info.keys())


def test_state_returns_full_snapshot() -> None:
    """State snapshot should include mutable trackers and post details."""

    env = MisinfoGuardEnv(config=CONFIG, seed=2)
    env.reset()
    env.step(0)

    snapshot = env.state()
    assert "timestep" in snapshot
    assert "remaining_budget" in snapshot
    assert "posts" in snapshot
    assert len(snapshot["posts"]) == CONFIG.environment.post_count


def test_budget_decreases_with_costed_action() -> None:
    """Flag action should consume configured budget."""

    env = MisinfoGuardEnv(config=CONFIG, seed=3)
    env.reset()

    initial_budget = env.remaining_budget
    _, _, _, _, _ = env.step(0)

    assert env.remaining_budget == initial_budget - CONFIG.environment.action_costs[0]
