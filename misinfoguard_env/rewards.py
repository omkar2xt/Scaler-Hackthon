"""Programmatically verifiable reward functions for MisinfoGuard-Env."""

from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    from .config import CONFIG, MisinfoGuardConfig
except ImportError:
    from config import CONFIG, MisinfoGuardConfig


@dataclass
class RewardBreakdown:
    """Per-component reward values required by the grader."""

    r_spread_reduction: float
    r_early_bonus: float
    r_true_post_safe: float
    r_false_positive_penalty: float
    r_overcensorship_penalty: float
    r_budget_efficiency: float

    @property
    def total_reward(self) -> float:
        """Return total reward as sum of all components."""

        return (
            self.r_spread_reduction
            + self.r_early_bonus
            + self.r_true_post_safe
            + self.r_false_positive_penalty
            + self.r_overcensorship_penalty
            + self.r_budget_efficiency
        )

    def to_info_dict(self) -> dict[str, float]:
        """Convert reward object into an info-compatible dictionary."""

        payload = asdict(self)
        payload["total_reward"] = self.total_reward
        return payload


def compute_reward(
    prev_false_reach: int,
    curr_false_reach: int,
    caught_before_threshold: bool,
    true_posts_not_touched: int,
    true_posts_flagged: int,
    quarantined_nodes: int,
    total_nodes: int,
    remaining_budget: int,
    initial_budget: int,
    config: MisinfoGuardConfig = CONFIG,
) -> RewardBreakdown:
    """Compute all verifiable reward components from scalar metrics."""

    reward_cfg = config.reward

    r_spread_reduction = (
        prev_false_reach - curr_false_reach
    ) * reward_cfg.spread_reduction_weight

    r_early_bonus = reward_cfg.early_catch_bonus if caught_before_threshold else 0.0
    r_true_post_safe = reward_cfg.true_post_safe_weight * true_posts_not_touched
    r_false_positive_penalty = (
        reward_cfg.false_positive_penalty_weight * true_posts_flagged
    )

    censorship_rate = (quarantined_nodes / total_nodes) if total_nodes > 0 else 0.0
    r_overcensorship_penalty = (
        reward_cfg.overcensorship_penalty
        if censorship_rate > reward_cfg.overcensorship_threshold
        else 0.0
    )

    budget_ratio = (remaining_budget / initial_budget) if initial_budget > 0 else 0.0
    r_budget_efficiency = reward_cfg.budget_efficiency_weight * budget_ratio

    return RewardBreakdown(
        r_spread_reduction=float(r_spread_reduction),
        r_early_bonus=float(r_early_bonus),
        r_true_post_safe=float(r_true_post_safe),
        r_false_positive_penalty=float(r_false_positive_penalty),
        r_overcensorship_penalty=float(r_overcensorship_penalty),
        r_budget_efficiency=float(r_budget_efficiency),
    )
