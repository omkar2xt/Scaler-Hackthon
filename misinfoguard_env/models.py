"""
Pydantic v2 models for MisinfoGuard-Env type contracts.
Provides typed observation, action, reward, and state schemas.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PostFeature(BaseModel):
    """Individual post feature vector."""

    post_id: str
    virality_score: float = Field(ge=0.0, le=1.0)
    content_similarity: float = Field(ge=0.0, le=1.0)
    engagement_rate: float = Field(ge=0.0, le=1.0)


class MisinfoObservation(BaseModel):
    """Environment observation returned by reset() and step()."""

    posts: List[PostFeature]
    network_state: List[float]
    infected_count: int = Field(ge=0)
    recovered_count: int = Field(ge=0)
    quarantined_count: int = Field(ge=0)
    timestep: int = Field(ge=0)


class MisinfoAction(BaseModel):
    """Action taken by the agent."""

    action_type: str
    post_id: str
    confidence: float = Field(ge=0.0, le=1.0)


class MisinfoReward(BaseModel):
    """Structured reward signal."""

    step_reward: float
    episode_bonus: Optional[float] = None
    false_reach_penalty: float
    precision_bonus: float
    total: float


class StepResult(BaseModel):
    """Result from a single environment step."""

    observation: MisinfoObservation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict


class EnvState(BaseModel):
    """Complete environment state snapshot."""

    observation: MisinfoObservation
    trajectory: Dict
    metadata: Dict
