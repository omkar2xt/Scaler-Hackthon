"""Inference entrypoint for HuggingFace Spaces deployment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from .agents.defender import HeuristicDefender
    from .config import CONFIG
    from .grader import grade_policy
except ImportError:
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
    """Run a single graded episode and print result JSON with structured logging."""

    # START marker for structured logging
    print(json.dumps({"marker": "START", "event": "inference_begin"}))

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

    # STEP marker: policy loaded
    print(json.dumps({"marker": "STEP", "event": "policy_loaded", "policy_type": type(policy).__name__}))

    result = grade_policy(policy=policy, episodes=1, config=CONFIG)

    # STEP marker: grading complete
    print(json.dumps({"marker": "STEP", "event": "grading_complete", "result": result.to_dict() if hasattr(result, 'to_dict') else str(result)}))

    # END marker: inference complete
    print(json.dumps({"marker": "END", "event": "inference_end"}))

    print(result.to_json())

    if api_base_url or model_name:
        _ = (api_base_url, model_name)


if __name__ == "__main__":
    main()
