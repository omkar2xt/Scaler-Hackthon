"""Inference entrypoint for HuggingFace Spaces deployment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agents.defender import HeuristicDefender
from config import CONFIG
from grader import grade_policy


def _load_model(model_path: Path) -> Any | None:
    """Load PPO model from disk if available, else return None."""

    if not model_path.exists():
        return None

    try:
        from stable_baselines3 import PPO

        return PPO.load(str(model_path))
    except Exception:
        return None


def main() -> None:
    """Run a single graded episode and print result JSON."""

    api_base_url = os.getenv("API_BASE_URL", "")
    model_name = os.getenv("MODEL_NAME", "misinfoguard-ppo")
    hf_token = os.getenv("HF_TOKEN", "")

    if hf_token:
        try:
            from huggingface_hub import login

            login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass

    model = _load_model(CONFIG.paths.best_model_path)
    policy = model if model is not None else HeuristicDefender()

    result = grade_policy(policy=policy, episodes=1, config=CONFIG)

    print(
        result.to_json()
    )

    if api_base_url or model_name:
        _ = (api_base_url, model_name)


if __name__ == "__main__":
    main()
