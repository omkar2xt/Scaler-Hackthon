"""Programmatic grader for MisinfoGuard-Env hackathon evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

from agents.defender import HeuristicDefender
from .config import CONFIG, MisinfoGuardConfig
from environment import MisinfoGuardEnv


class Predictor(Protocol):
    """Prediction protocol compatible with baseline and RL models."""

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        """Return action prediction result."""


@dataclass
class GraderResult:
    """Aggregated grading metrics expected by hackathon judges."""

    spread_reduction_score: float
    precision: float
    recall: float
    f1_score: float
    censorship_rate: float
    final_score: float

    def to_json(self) -> str:
        """Serialize score payload to JSON string."""

        return json.dumps(asdict(self), indent=2)


def _safe_div(numerator: float, denominator: float) -> float:
    """Divide with zero-denominator protection."""

    return float(numerator / denominator) if denominator > 0 else 0.0


def _predict_action(policy: Any, observation: dict[str, Any]) -> int:
    """Extract action from either SB3 model or heuristic policy."""

    if hasattr(policy, "predict"):
        result = policy.predict(observation, deterministic=True)
        if isinstance(result, tuple):
            return int(result[0])
        return int(result)
    raise TypeError("Policy object must expose a predict method")


def grade_policy(
    policy: Any | None = None,
    episodes: int | None = None,
    config: MisinfoGuardConfig = CONFIG,
    seed: int = 42,
) -> GraderResult:
    """Run multi-episode grading and return averaged metrics."""

    env = MisinfoGuardEnv(config=config, seed=seed)
    effective_policy = policy if policy is not None else HeuristicDefender()
    eval_episodes = episodes if episodes is not None else config.grader.evaluation_episodes

    spread_scores: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    censorship_rates: list[float] = []

    for _ in range(eval_episodes):
        observation, _ = env.reset()
        done = False

        initial_false_reach = env.simulator.compute_reach_stats(env.posts).false_reach

        while not done:
            action = _predict_action(effective_policy, observation)
            observation, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        final_false_reach = env.simulator.compute_reach_stats(env.posts).false_reach
        spread_reduction_score = _safe_div(
            max(0, initial_false_reach - final_false_reach),
            max(1, initial_false_reach),
        )

        false_post_ids = {post.post_id for post in env.posts if post.is_false}

        true_positives = len(env.flagged_posts & false_post_ids)
        all_flagged = len(env.flagged_posts)
        all_false_posts = len(false_post_ids)

        precision = _safe_div(true_positives, all_flagged)
        recall = _safe_div(true_positives, all_false_posts)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)

        censorship_rate = _safe_div(
            len(env.quarantined_nodes),
            config.environment.node_count,
        )

        spread_scores.append(spread_reduction_score)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        censorship_rates.append(censorship_rate)

    avg_spread = float(np.mean(spread_scores))
    avg_precision = float(np.mean(precisions))
    avg_recall = float(np.mean(recalls))
    avg_f1 = float(np.mean(f1_scores))
    avg_censorship = float(np.mean(censorship_rates))

    weights = config.grader.composite_weights
    final_score = (
        weights["spread_reduction_score"] * avg_spread
        + weights["precision"] * avg_precision
        + weights["recall"] * avg_recall
        + weights["f1_score"] * avg_f1
        + weights["censorship_penalty"] * (1.0 - avg_censorship)
    )

    return GraderResult(
        spread_reduction_score=avg_spread,
        precision=avg_precision,
        recall=avg_recall,
        f1_score=avg_f1,
        censorship_rate=avg_censorship,
        final_score=float(final_score),
    )
