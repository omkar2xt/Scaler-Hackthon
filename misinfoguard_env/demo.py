"""CLI demo for observing defender performance in MisinfoGuard-Env."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from .agents.defender import HeuristicDefender
from .config import CONFIG
from .environment import MisinfoGuardEnv

logger = logging.getLogger(__name__)


def _load_demo_policy(policy_name: str) -> Any:
    """Load requested demo policy."""

    if policy_name == "heuristic":
        return HeuristicDefender()
    if policy_name == "random":
        return None
    if policy_name == "ppo":
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            logger.error("Failed to import stable_baselines3.PPO: %s", exc)
            return HeuristicDefender()

        try:
            return PPO.load(str(CONFIG.paths.best_model_path))
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            logger.error("Failed to load PPO model from %s: %s", CONFIG.paths.best_model_path, exc)
            return HeuristicDefender()
    return HeuristicDefender()


def _choose_action(policy: Any, observation: dict[str, Any], env: MisinfoGuardEnv) -> int:
    """Get action from selected policy object."""

    if policy is None:
        return int(env.action_space.sample())

    if hasattr(policy, "predict"):
        result = policy.predict(observation)
        if isinstance(result, tuple):
            return int(result[0])
        return int(result)

    return int(env.action_space.sample())


def run_demo(episodes: int, policy_name: str) -> None:
    """Run demo episodes and print key metrics to stdout."""

    env = MisinfoGuardEnv(config=CONFIG, seed=123)
    policy = _load_demo_policy(policy_name)

    for episode in range(1, episodes + 1):
        observation, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = _choose_action(policy, observation, env)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        print(
            " | ".join(
                [
                    f"episode={episode}",
                    f"policy={policy_name}",
                    f"reward={episode_reward:.3f}",
                    f"false_reach={info['false_reach_fraction']:.3f}",
                    f"precision={_episode_precision(env):.3f}",
                ]
            )
        )


def _episode_precision(env: MisinfoGuardEnv) -> float:
    """Compute episode precision from final env state."""

    false_post_ids = {post.post_id for post in env.posts if post.is_false}
    tp = len(env.flagged_posts & false_post_ids)
    total = len(env.flagged_posts)
    return float(tp / total) if total > 0 else 0.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for demo script."""

    parser = argparse.ArgumentParser(description="MisinfoGuard-Env demo runner")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument(
        "--policy",
        type=str,
        default="heuristic",
        choices=["heuristic", "random", "ppo"],
        help="Policy to run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(episodes=args.episodes, policy_name=args.policy)
