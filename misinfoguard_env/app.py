"""
FastAPI application for MisinfoGuard-Env inference server.
Provides REST endpoints for environment interaction with thread-safe locking.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import threading
import os
import logging

from misinfoguard_env.environment import MisinfoGuardEnv
from misinfoguard_env.models import MisinfoObservation, MisinfoAction

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MisinfoGuard-Env API",
    description="REST API for misinformation defense RL training",
    version="0.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance with thread-safe locking
env_instance = None
env_lock = threading.Lock()


class ResetRequest(BaseModel):
    seed: int = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


class ActionInput(BaseModel):
    action_type: str
    post_id: str
    confidence: float


@app.on_event("startup")
async def startup_event():
    """Initialize environment on startup."""
    global env_instance
    try:
        with env_lock:
            env_instance = MisinfoGuardEnv()
            logger.info("Environment initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment_initialized": env_instance is not None
    }


@app.post("/reset")
async def reset_env(request: ResetRequest = None):
    """Reset the environment."""
    if env_instance is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    with env_lock:
        try:
            seed = request.seed if request else None
            observation, info = env_instance.reset(seed=seed)
            return {
                "observation": observation,
                "info": info
            }
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step_env(request: StepRequest):
    """Execute one environment step."""
    if env_instance is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    with env_lock:
        try:
            action = request.action
            observation, reward, terminated, truncated, info = env_instance.step(action)
            return {
                "observation": observation,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info
            }
        except Exception as e:
            logger.error(f"Step failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
async def get_state():
    """Get current environment state."""
    if env_instance is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")
    
    with env_lock:
        try:
            state = env_instance.get_state()
            return {
                "state": state,
                "timestep": env_instance.timestep
            }
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
