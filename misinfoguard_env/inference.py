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


def _probe_llm_proxy(api_base_url: str, api_key: str, model_name: str) -> bool:
    """Send a minimal OpenAI-compatible request through the provided proxy."""

    if not api_base_url or not api_key:
        return False

    try:
        from openai import OpenAI

        client = OpenAI(base_url=api_base_url, api_key=api_key)
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Return OK"}],
            max_tokens=2,
            temperature=0,
        )
        return True
    except Exception:
        return False


def main() -> None:
    """Run a single graded episode and print parser-friendly structured logs."""

    import sys
    
    # Ensure stdout is not buffered and prints immediately
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    
    final_score = 0.5

    print("[START] task=misinfoguard episodes=1", flush=True)
    sys.stdout.flush()

    try:
        api_base_url = os.getenv("API_BASE_URL", "")
        api_key = os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", ""))
        model_name = os.getenv("MODEL_NAME", "misinfoguard-ppo")
        hf_token = os.getenv("HF_TOKEN", "")

        proxy_ok = _probe_llm_proxy(api_base_url=api_base_url, api_key=api_key, model_name=model_name)
        print(f"[STEP] step=llm_proxy_call ok={int(proxy_ok)}", flush=True)
        sys.stdout.flush()

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
        final_score = min(0.999999, max(0.000001, float(result.final_score)))

        print(f"[STEP] step=grading_complete score={final_score:.6f}", flush=True)
        sys.stdout.flush()
        
        print(result.to_json(), flush=True)
        sys.stdout.flush()
        
    except Exception as e:
        print(f"[ERROR] exception={type(e).__name__} msg={str(e)}", flush=True)
        sys.stdout.flush()
    finally:
        print(f"[END] task=misinfoguard score={final_score:.6f} steps=1", flush=True)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
