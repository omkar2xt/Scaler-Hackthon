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
    """Run a single graded episode and print parser-friendly structured logs."""

    import sys
    
    # Ensure stdout is not buffered and prints immediately
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    
    print("[START] task=misinfoguard episodes=1", flush=True)
    sys.stdout.flush()

    try:
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

        print(f"[STEP] step=policy_loaded policy={type(policy).__name__}", flush=True)
        sys.stdout.flush()

        result = grade_policy(policy=policy, episodes=1, config=CONFIG)

        print(f"[STEP] step=grading_complete score={result.final_score:.6f}", flush=True)
        sys.stdout.flush()
        
        print(result.to_json(), flush=True)
        sys.stdout.flush()
        
    except Exception as e:
        print(f"[ERROR] exception={type(e).__name__} msg={str(e)}", flush=True)
        sys.stdout.flush()
    finally:
        print(f"[END] task=misinfoguard score=0.0 steps=1", flush=True)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
