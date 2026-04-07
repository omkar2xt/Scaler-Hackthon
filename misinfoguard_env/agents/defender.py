"""Defender agent interfaces for MisinfoGuard-Env."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class DefenderPolicy(Protocol):
    """Protocol for defender policies used by demo/grader."""

    def predict(self, observation: dict[str, Any]) -> int:
        """Return one discrete action index for the current observation."""


@dataclass
class HeuristicDefender:
    """Simple baseline policy that reacts to post risk signals."""

    risk_threshold: float = 0.55

    def predict(self, observation: dict[str, Any]) -> int:
        """Choose actions using post falsehood and spread features."""

        post_features = np.asarray(observation["post_features"], dtype=np.float32)
        if post_features.size == 0:
            return 4
        if post_features.ndim == 1:
            post_features = post_features.reshape(1, -1)
        if post_features.ndim != 2 or post_features.shape[1] < 4:
            return 4

        falsehood_scores = post_features[:, 0]
        spread_scores = post_features[:, 1]
        virality_scores = post_features[:, 3]

        risk = falsehood_scores * spread_scores * np.maximum(virality_scores, 1e-6)
        top_risk = float(np.max(risk))

        if top_risk >= self.risk_threshold:
            return 0
        if float(np.mean(virality_scores)) > 0.2:
            return 1
        if float(np.mean(falsehood_scores)) < 0.4:
            return 2
        return 4
