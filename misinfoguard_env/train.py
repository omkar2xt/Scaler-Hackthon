"""PPO training loop for MisinfoGuard-Env."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .config import CONFIG
from environment import MisinfoGuardEnv


class GymCompatWrapper(gym.Wrapper):
    """Pass-through wrapper for Gymnasium-compatible environment API."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset wrapped environment via its public API."""

        observation, info = self.env.reset(seed=seed, options=options)
        return observation, info

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step wrapped environment and return Gymnasium tuple."""

        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info


class RewardLoggingCallback:
    """Simple callback-compatible logger for periodic reward output."""

    def __init__(self, print_interval_steps: int) -> None:
        """Initialize callback state."""

        from stable_baselines3.common.callbacks import BaseCallback

        class _InnerCallback(BaseCallback):
            def __init__(self, interval: int) -> None:
                super().__init__()
                self.interval = interval

            def _on_step(self) -> bool:
                if self.n_calls % self.interval == 0:
                    mean_reward = float(np.mean(self.locals.get("rewards", [0.0])))
                    print(f"step={self.num_timesteps} mean_reward={mean_reward:.4f}")
                return True

        self.callback = _InnerCallback(print_interval_steps)


def _ensure_directories() -> None:
    """Create checkpoint and log directories if missing."""

    CONFIG.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.paths.log_dir.mkdir(parents=True, exist_ok=True)


def train(total_timesteps: int, output_path: Path) -> Path:
    """Train PPO for configured timesteps and save best model."""

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    _ensure_directories()

    def make_env() -> gym.Env:
        base_env = MisinfoGuardEnv(config=CONFIG)
        return GymCompatWrapper(base_env)

    vec_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=str(CONFIG.paths.log_dir),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(CONFIG.paths.checkpoint_dir),
        log_path=str(CONFIG.paths.log_dir),
        eval_freq=CONFIG.training.log_interval_steps,
        deterministic=True,
        render=False,
    )

    reward_logger = RewardLoggingCallback(CONFIG.training.log_interval_steps).callback

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, reward_logger],
        progress_bar=False,
    )

    model.save(str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse train-script CLI arguments."""

    parser = argparse.ArgumentParser(description="Train MisinfoGuard PPO agent")
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run short sanity training instead of full training",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    timesteps = (
        CONFIG.training.sanity_timesteps
        if args.sanity
        else CONFIG.training.total_timesteps
    )
    output_path = (
        CONFIG.paths.sanity_model_path if args.sanity else CONFIG.paths.best_model_path
    )
    saved_path = train(total_timesteps=timesteps, output_path=output_path)
    print(f"Training complete. Best model saved at: {saved_path}")
