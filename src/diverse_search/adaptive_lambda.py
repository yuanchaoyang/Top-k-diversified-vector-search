from __future__ import annotations

"""Heuristics to choose a query-adaptive MMR lambda based on candidate scores.

Strategies:
  - gap_piecewise: lambda from gap = s1 - s_k via a simple 3-level rule.
  - gap_linear: smooth linear mapping from gap to lambda.
  - entropy: lambda from normalized entropy of a temperature-scaled softmax.
  - hybrid: combines gap + variance + top1 score for robust estimation.
  - candidate_diversity: uses candidate pairwise similarity to decide lambda.
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
        strategy: "entropy", "gap_piecewise", "gap_linear", "hybrid", or "candidate_diversity".
        temperature: Softmax temperature for the score distribution.
        entropy_power: Exponent to re-shape entropy ( >1 pushes values toward extremes).
        topM: Optional cap on how many top scores to look at when computing entropy.
        gap_k: Use gap between s1 and s_k.
        gap_t_low / gap_t_high: thresholds for piecewise mapping.
        hybrid_weights: weights for [gap, variance, top1] in hybrid strategy.
        candidate_vectors: vectors for candidate_diversity strategy (set at runtime).
    """

    lambda_min: float = 0.4
    lambda_max: float = 0.8
    lambda_mid: float = 0.6
    strategy: str = "entropy"  # best performing strategy
    temperature: float = 0.05
    entropy_power: float = 1.0
    topM: Optional[int] = 50
    gap_k: int = 10
    gap_t_low: float = 0.02
    gap_t_high: float = 0.08
    # Hybrid strategy weights: [gap_weight, variance_weight, top1_weight]
    hybrid_weights: tuple = (0.5, 0.3, 0.2)


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


def _score_variance(scores: np.ndarray, topM: Optional[int] = None) -> float:
    """Variance of top-M scores (normalized by mean to be scale-invariant)."""
    x = np.asarray(scores, dtype=np.float32).reshape(-1)
    if topM is not None and topM > 0:
        x = x[:topM]
    if x.size <= 1:
        return 0.0
    mean = float(np.mean(x))
    if abs(mean) < 1e-8:
        return float(np.var(x))
    # Coefficient of variation (normalized variance)
    return float(np.std(x) / abs(mean))


def _score_slope(scores: np.ndarray, topM: Optional[int] = None) -> float:
    """Slope of score decay (steeper = clearer query)."""
    x = np.asarray(scores, dtype=np.float32).reshape(-1)
    if topM is not None and topM > 0:
        x = x[:topM]
    if x.size <= 2:
        return 0.0
    # Linear regression slope
    n = x.size
    indices = np.arange(n, dtype=np.float32)
    slope = float(np.cov(indices, x)[0, 1] / (np.var(indices) + 1e-8))
    return -slope  # negative because scores decrease; return positive value


def _gap_linear_lambda(scores: np.ndarray, cfg: AdaptiveLambdaConfig) -> float:
    """Linear mapping from gap to lambda (smoother than piecewise)."""
    g = _gap_value(scores, cfg)

    # Normalize gap to [0, 1] range using adaptive percentiles
    # gap_t_low and gap_t_high define the "typical" range
    t_low = float(cfg.gap_t_low)
    t_high = float(cfg.gap_t_high)

    if t_high <= t_low:
        t_high = t_low + 0.01

    # Linear interpolation: gap -> [0, 1] -> [lambda_min, lambda_max]
    normalized = (g - t_low) / (t_high - t_low)
    normalized = float(np.clip(normalized, 0.0, 1.0))

    lam = float(cfg.lambda_min) + normalized * (float(cfg.lambda_max) - float(cfg.lambda_min))
    return float(np.clip(lam, float(cfg.lambda_min), float(cfg.lambda_max)))


def _hybrid_lambda(scores: np.ndarray, cfg: AdaptiveLambdaConfig) -> float:
    """Combine multiple signals for robust lambda estimation.

    Signals:
    1. Gap (s1 - sk): larger gap = clearer query = higher lambda
    2. Variance: higher variance = clearer separation = higher lambda
    3. Top1 score: higher top1 = more confident match = higher lambda
    """
    x = np.asarray(scores, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return float(cfg.lambda_mid)

    topM = cfg.topM if cfg.topM and cfg.topM > 0 else 50

    # 1. Gap signal (normalized)
    gap = _gap_value(scores, cfg)
    gap_norm = (gap - cfg.gap_t_low) / (cfg.gap_t_high - cfg.gap_t_low + 1e-8)
    gap_norm = float(np.clip(gap_norm, 0.0, 1.0))

    # 2. Variance signal (coefficient of variation)
    cv = _score_variance(x, topM)
    # Typical CV range is 0.01-0.2, map to [0, 1]
    cv_norm = float(np.clip(cv / 0.15, 0.0, 1.0))

    # 3. Top1 score signal
    # Higher absolute score = more confident
    top1 = float(x[0]) if x.size > 0 else 0.0
    # Typical range for normalized vectors is 0.3-0.8
    top1_norm = float(np.clip((top1 - 0.3) / 0.5, 0.0, 1.0))

    # Weighted combination
    w = cfg.hybrid_weights
    if len(w) != 3:
        w = (0.5, 0.3, 0.2)

    combined = w[0] * gap_norm + w[1] * cv_norm + w[2] * top1_norm
    combined = combined / (sum(w) + 1e-8)  # normalize weights

    lam = float(cfg.lambda_min) + combined * (float(cfg.lambda_max) - float(cfg.lambda_min))
    return float(np.clip(lam, float(cfg.lambda_min), float(cfg.lambda_max)))


def _candidate_diversity_lambda(
    scores: np.ndarray,
    cfg: AdaptiveLambdaConfig,
    candidate_vectors: Optional[np.ndarray] = None,
) -> float:
    """Choose lambda based on candidate set diversity.

    Intuition:
    - If candidates are already diverse → don't need aggressive diversification → high lambda
    - If candidates are very similar → need more diversification → low lambda
    """
    if candidate_vectors is None or candidate_vectors.shape[0] < 2:
        # Fallback to hybrid if no vectors provided
        return _hybrid_lambda(scores, cfg)

    # Compute average pairwise similarity of top candidates
    k = min(cfg.gap_k, candidate_vectors.shape[0])
    vecs = candidate_vectors[:k]

    # Pairwise cosine similarity (assumes normalized vectors)
    sim_matrix = vecs @ vecs.T
    # Get upper triangle (excluding diagonal)
    n = sim_matrix.shape[0]
    if n <= 1:
        avg_sim = 0.5
    else:
        triu_indices = np.triu_indices(n, k=1)
        avg_sim = float(np.mean(sim_matrix[triu_indices]))

    # avg_sim close to 1 = candidates very similar = need diversity = low lambda
    # avg_sim close to 0 = candidates already diverse = high lambda
    # Typical range: 0.2 - 0.8
    diversity_score = 1.0 - avg_sim  # higher = more diverse
    diversity_norm = float(np.clip((diversity_score - 0.2) / 0.6, 0.0, 1.0))

    # Also consider gap for relevance signal
    gap = _gap_value(scores, cfg)
    gap_norm = (gap - cfg.gap_t_low) / (cfg.gap_t_high - cfg.gap_t_low + 1e-8)
    gap_norm = float(np.clip(gap_norm, 0.0, 1.0))

    # Combine: 60% diversity signal, 40% gap signal
    combined = 0.6 * diversity_norm + 0.4 * gap_norm

    lam = float(cfg.lambda_min) + combined * (float(cfg.lambda_max) - float(cfg.lambda_min))
    return float(np.clip(lam, float(cfg.lambda_min), float(cfg.lambda_max)))


def adaptive_lambda_from_scores(
    scores: np.ndarray,
    cfg: AdaptiveLambdaConfig,
    candidate_vectors: Optional[np.ndarray] = None,
) -> float:
    """Map score statistics -> lambda in [lambda_min, lambda_max].

    Args:
        scores: Candidate similarity scores (descending order).
        cfg: Configuration for adaptive lambda.
        candidate_vectors: Optional vectors for candidate_diversity strategy.
    """
    strategy = cfg.strategy.lower()

    if strategy == "gap_piecewise":
        g = _gap_value(scores, cfg)
        if g >= float(cfg.gap_t_high):
            lam = float(cfg.lambda_max)
        elif g <= float(cfg.gap_t_low):
            lam = float(cfg.lambda_min)
        else:
            lam = float(cfg.lambda_mid)
        return float(np.clip(lam, float(cfg.lambda_min), float(cfg.lambda_max)))

    if strategy == "gap_linear":
        return _gap_linear_lambda(scores, cfg)

    if strategy == "hybrid":
        return _hybrid_lambda(scores, cfg)

    if strategy == "candidate_diversity":
        return _candidate_diversity_lambda(scores, cfg, candidate_vectors)

    # Default: entropy strategy
    e = _normalized_entropy(scores, cfg)
    weight = (1.0 - e) ** float(cfg.entropy_power)
    lam = float(cfg.lambda_min) + weight * (float(cfg.lambda_max) - float(cfg.lambda_min))
    return float(np.clip(lam, float(cfg.lambda_min), float(cfg.lambda_max)))


def batch_adaptive_lambdas(
    score_matrix: np.ndarray,
    cfg: AdaptiveLambdaConfig,
    candidate_indices: Optional[np.ndarray] = None,
    xb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorized helper for (nq, topN) scores.

    Args:
        score_matrix: (nq, topN) similarity scores.
        cfg: Adaptive lambda configuration.
        candidate_indices: Optional (nq, topN) indices for candidate_diversity strategy.
        xb: Optional database vectors for candidate_diversity strategy.
    """
    S = np.asarray(score_matrix, dtype=np.float32)
    out = np.empty(S.shape[0], dtype=np.float32)

    need_vectors = cfg.strategy.lower() == "candidate_diversity" and xb is not None

    for i in range(S.shape[0]):
        if need_vectors and candidate_indices is not None:
            cand_vecs = xb[candidate_indices[i]]
            out[i] = adaptive_lambda_from_scores(S[i], cfg, cand_vecs)
        else:
            out[i] = adaptive_lambda_from_scores(S[i], cfg)
    return out


# ============================================================================
# Auto-tuning utilities
# ============================================================================


def auto_tune_thresholds(
    score_matrix: np.ndarray,
    percentile_low: float = 25.0,
    percentile_high: float = 75.0,
) -> tuple:
    """Automatically determine gap thresholds from data.

    Args:
        score_matrix: (nq, topN) scores from a sample of queries.
        percentile_low: Percentile for gap_t_low (default 25th).
        percentile_high: Percentile for gap_t_high (default 75th).

    Returns:
        (gap_t_low, gap_t_high) tuple.
    """
    S = np.asarray(score_matrix, dtype=np.float32)
    gaps = []
    for i in range(S.shape[0]):
        if S.shape[1] >= 10:
            gap = float(S[i, 0] - S[i, 9])
        else:
            gap = float(S[i, 0] - S[i, -1])
        gaps.append(gap)

    gaps = np.array(gaps)
    t_low = float(np.percentile(gaps, percentile_low))
    t_high = float(np.percentile(gaps, percentile_high))

    return t_low, t_high


def create_auto_tuned_config(
    score_matrix: np.ndarray,
    strategy: str = "hybrid",
    lambda_min: float = 0.5,
    lambda_max: float = 0.95,
) -> AdaptiveLambdaConfig:
    """Create an AdaptiveLambdaConfig with auto-tuned thresholds.

    Args:
        score_matrix: Sample of (nq, topN) scores for tuning.
        strategy: Strategy to use.
        lambda_min: Minimum lambda value.
        lambda_max: Maximum lambda value.

    Returns:
        Configured AdaptiveLambdaConfig.
    """
    t_low, t_high = auto_tune_thresholds(score_matrix)

    return AdaptiveLambdaConfig(
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        lambda_mid=(lambda_min + lambda_max) / 2,
        strategy=strategy,
        gap_t_low=t_low,
        gap_t_high=t_high,
    )
