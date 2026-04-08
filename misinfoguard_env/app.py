"""FastAPI application for MisinfoGuard-Env deployment on Hugging Face Spaces."""

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment import MisinfoGuardEnv

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MisinfoGuard-Env API",
    description="REST API for misinformation defense RL training",
    version="0.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env_instance: MisinfoGuardEnv | None = None
env_lock = threading.Lock()


def _to_json_safe(value: Any) -> Any:
    """Recursively convert environment outputs into JSON-safe primitives."""

    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_to_json_safe(v) for v in sorted(value)]

    if hasattr(value, "tolist"):
        try:
            return _to_json_safe(value.tolist())
        except Exception:
            pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


class ResetRequest(BaseModel):
    seed: int | None = None


class StepRequest(BaseModel):
    action: int


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the environment when the Space starts."""

    global env_instance
    with env_lock:
        try:
            env_instance = MisinfoGuardEnv()
            logger.info("Environment initialized successfully")
        except Exception as exc:
            logger.exception("Failed to initialize environment: %s", exc)
            raise


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "MisinfoGuard-Env"}


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {"status": "healthy", "environment_initialized": env_instance is not None}


@app.post("/reset")
async def reset_env(request: ResetRequest | None = None) -> dict[str, Any]:
    global env_instance
    if env_instance is None:
        with env_lock:
            env_instance = MisinfoGuardEnv()

    with env_lock:
        try:
            seed = request.seed if request else None
            observation, info = env_instance.reset(seed=seed)
            return _to_json_safe({"observation": observation, "info": info})
        except Exception as exc:
            logger.exception("Reset failed: %s", exc)
            try:
                env_instance = MisinfoGuardEnv()
                observation, info = env_instance.reset(seed=seed if request else None)
                payload = _to_json_safe({"observation": observation, "info": info})
                payload["info"]["warning"] = f"Recovered from reset error: {exc}"
                return payload
            except Exception as fallback_exc:
                logger.exception("Reset recovery failed: %s", fallback_exc)
                return {
                    "observation": {},
                    "info": {
                        "error": str(exc),
                        "recovery_error": str(fallback_exc),
                    },
                }


@app.post("/step")
async def step_env(request: StepRequest) -> dict[str, Any]:
    if env_instance is None:
        return {
            "observation": {},
            "reward": 0.0,
            "terminated": True,
            "truncated": True,
            "info": {"error": "Environment not initialized"},
        }

    with env_lock:
        try:
            observation, reward, terminated, truncated, info = env_instance.step(request.action)
            return _to_json_safe({
                "observation": observation,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
            })
        except Exception as exc:
            logger.exception("Step failed: %s", exc)
            return {
                "observation": {},
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "info": {"error": str(exc)},
            }


@app.get("/state")
async def get_state() -> dict[str, Any]:
    if env_instance is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    with env_lock:
        try:
            state = env_instance.get_state()
            return _to_json_safe({"state": state, "timestep": env_instance.timestep})
        except Exception as exc:
            logger.exception("Failed to get state: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
