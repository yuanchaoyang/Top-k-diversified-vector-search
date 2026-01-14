from __future__ import annotations

"""Heuristics to choose a query-adaptive MMR lambda based on candidate scores.

Strategies:
  - gap_piecewise (default): lambda from gap = s1 - s_k via a simple 3-level rule.
  - entropy: lambda from normalized entropy of a temperature-scaled softmax.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AdaptiveLambdaConfig:
    """Config for query-adaptive lambda.

    Args:
        lambda_min: Lower bound for lambda when the query looks ambiguous.
        lambda_max: Upper bound for lambda when the query looks very specific.
        lambda_mid: Mid value used by gap_piecewise strategy.
        strategy: "entropy" or "gap_piecewise".
        temperature: Softmax temperature for the score distribution.
        entropy_power: Exponent to re-shape entropy ( >1 pushes values toward extremes).
        topM: Optional cap on how many top scores to look at when computing entropy.
        gap_k: Use gap between s1 and s_k.
        gap_t_low / gap_t_high: thresholds for piecewise mapping.
    """

    lambda_min: float = 0.2
    lambda_max: float = 0.95
    lambda_mid: float = 0.8
    strategy: str = "entropy"  # or "gap_piecewise"
    temperature: float = 0.08
    entropy_power: float = 1.0
    topM: Optional[int] = 50
    gap_k: int = 10
    gap_t_low: float = 0.02
    gap_t_high: float = 0.08


def _normalized_entropy(scores: np.ndarray, cfg: AdaptiveLambdaConfig) -> float:
    """Return entropy in [0, 1] (1 = most ambiguous / flat)."""
    x = np.asarray(scores, dtype=np.float32).reshape(-1)
    if cfg.topM is not None and int(cfg.topM) > 0:
        x = x[: int(cfg.topM)]

    if x.size == 0:
        return 1.0

    # stabilize softmax
    x = x - float(np.max(x))
    denom = max(float(cfg.temperature), 1e-6)
    probs = np.exp(x / denom)
    s = float(np.sum(probs))
    if s <= 0.0 or not np.isfinite(s):
        return 1.0
    probs = probs / s

    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    if x.size <= 1:
        return 0.0
    norm = float(np.log(x.size))
    return float(np.clip(entropy / norm, 0.0, 1.0))


def _gap_value(scores: np.ndarray, cfg: AdaptiveLambdaConfig) -> float:
    """Gap between top1 and top-k score."""
    x = np.asarray(scores, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    k = max(1, int(cfg.gap_k))
    idx = min(k - 1, x.size - 1)
    return float(x[0] - x[idx])


def adaptive_lambda_from_scores(scores: np.ndarray, cfg: AdaptiveLambdaConfig) -> float:
    """Map score statistics -> lambda in [lambda_min, lambda_max]."""
    if cfg.strategy == "gap_piecewise":
        g = _gap_value(scores, cfg)
        if g >= float(cfg.gap_t_high):
            lam = float(cfg.lambda_max)
        elif g <= float(cfg.gap_t_low):
            lam = float(cfg.lambda_min)
        else:
            lam = float(cfg.lambda_mid)
        return float(np.clip(lam, float(cfg.lambda_min), float(cfg.lambda_max)))

    e = _normalized_entropy(scores, cfg)
    weight = (1.0 - e) ** float(cfg.entropy_power)
    lam = float(cfg.lambda_min) + weight * (float(cfg.lambda_max) - float(cfg.lambda_min))
    return float(np.clip(lam, float(cfg.lambda_min), float(cfg.lambda_max)))


def batch_adaptive_lambdas(score_matrix: np.ndarray, cfg: AdaptiveLambdaConfig) -> np.ndarray:
    """Vectorized helper for (nq, topN) scores."""
    S = np.asarray(score_matrix, dtype=np.float32)
    out = np.empty(S.shape[0], dtype=np.float32)
    for i in range(S.shape[0]):
        out[i] = adaptive_lambda_from_scores(S[i], cfg)
    return out
