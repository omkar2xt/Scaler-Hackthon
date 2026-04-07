"""Centralized configuration for MisinfoGuard-Env."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration values that define episode mechanics."""

    node_count: int = 500
    post_count: int = 20
    max_steps: int = 50
    initial_budget: int = 30
    observation_hops: int = 2
    monitor_seed_count: int = 3

    action_space_size: int = 5
    action_costs: dict[int, int] = field(
        default_factory=lambda: {
            0: 1,  # flag(post_id)
            1: 2,  # counter(post_id)
            2: 2,  # amplify(post_id)
            3: 3,  # quarantine(node_id)
            4: 0,  # wait()
        }
    )


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration values for social graph and spread simulation."""

    watts_k_neighbors: int = 8
    watts_rewire_prob: float = 0.15
    trust_weight_range: tuple[float, float] = (0.3, 1.0)

    virality_decay: float = 0.90
    default_true_virality: float = 0.08
    default_false_virality: float = 0.12
    true_falsehood_score_range: tuple[float, float] = (0.0, 0.35)
    false_falsehood_score_range: tuple[float, float] = (0.65, 1.0)
    source_trust_range: tuple[float, float] = (0.2, 1.0)

    false_post_ratio: float = 0.5

    initial_seed_min: int = 1
    initial_seed_max: int = 8
    quarantine_trust_scale: float = 0.1
    flag_virality_scale: float = 0.7
    counter_false_virality_scale: float = 0.8
    amplify_true_virality_scale: float = 1.2
    virality_floor: float = 0.01
    virality_cap: float = 1.0


@dataclass(frozen=True)
class SpreaderConfig:
    """Configuration values for scripted adversarial spreaders."""

    floodbot_injections_per_step: int = 2
    targetbot_injections_per_step: int = 1
    burstbot_delay_steps: int = 5
    burstbot_injection_count: int = 6
    injected_false_virality: float = 0.18


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights and threshold values."""

    spread_reduction_weight: float = 0.5
    early_catch_bonus: float = 0.3
    early_catch_threshold: float = 0.10
    true_post_safe_weight: float = 0.1
    false_positive_penalty_weight: float = -0.4
    overcensorship_penalty: float = -0.5
    overcensorship_threshold: float = 0.15
    budget_efficiency_weight: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    """Training parameters for PPO."""

    algorithm: str = "PPO"
    total_timesteps: int = 100_000
    sanity_timesteps: int = 5_000
    log_interval_steps: int = 1_000
    tensorboard_log_dir: str = "logs"


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths used by training and inference."""

    project_root: Path = Path(__file__).resolve().parent
    checkpoint_dir: Path = field(init=False)
    best_model_path: Path = field(init=False)
    sanity_model_path: Path = field(init=False)
    log_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Compute derived paths from project_root for each instance."""

        checkpoint_dir = self.project_root / "checkpoints"
        object.__setattr__(self, "checkpoint_dir", checkpoint_dir)
        object.__setattr__(self, "best_model_path", checkpoint_dir / "best_model.zip")
        object.__setattr__(self, "sanity_model_path", checkpoint_dir / "sanity_model.zip")
        object.__setattr__(self, "log_dir", self.project_root / "logs")


@dataclass(frozen=True)
class GraderConfig:
    """Grader evaluation settings."""

    evaluation_episodes: int = 10
    composite_weights: dict[str, float] = field(
        default_factory=lambda: {
            "spread_reduction_score": 0.35,
            "precision": 0.20,
            "recall": 0.20,
            "f1_score": 0.20,
            "censorship_penalty": 0.05,
        }
    )


@dataclass(frozen=True)
class MisinfoGuardConfig:
    """Top-level application configuration object."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    spreader: SpreaderConfig = field(default_factory=SpreaderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    grader: GraderConfig = field(default_factory=GraderConfig)


CONFIG = MisinfoGuardConfig()
