from __future__ import annotations

"""A small, explicit input/output API for diversified top-k vector search.

This file gives you a *single* place to define the boundary of your system:

Input:
  - Query vectors q (shape: (nq, d) or (d,))
  - A vector database xb (shape: (nb, d)), typically L2-normalized
  - A candidate retriever (exact or ANN) that returns topN by relevance

Output:
  - A list S of length k for each query (indices into xb)
  - Per-item relevance scores (cosine similarity to q, assuming normalization)

Having this explicit API makes it much easier to write a clean Methods section.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np

from .adaptive_lambda import AdaptiveLambdaConfig, adaptive_lambda_from_scores
from .index import BaseRetriever
from .diversify import maxmin_rerank, threshold_greedy_rerank
from .mmr import mmr_rerank, mmr_rerank_incremental, mmr_rerank_temporal


Method = Literal["baseline", "mmr", "threshold", "maxmin"]
MMRImpl = Literal["incremental", "full"]


@dataclass(frozen=True)
class DiversifiedSearchRequest:
    """All inputs needed for one diversified search call."""

    queries: np.ndarray  # (nq, d) or (d,)
    k: int
    topN: int

    # reranking strategy
    method: Method = "mmr"

    # method parameters
    lambda_: float = 0.5  # for MMR
    adaptive_lambda: bool = False  # if True, override lambda_ per-query
    adaptive_lambda_strategy: str = "gap_piecewise"
    adaptive_lambda_min: float = 0.2
    adaptive_lambda_max: float = 0.95
    adaptive_lambda_mid: float = 0.8  # only used by gap_piecewise
    adaptive_lambda_temperature: float = 0.08
    adaptive_lambda_entropy_power: float = 1.0
    adaptive_lambda_topM: int = 50
    adaptive_gap_k: int = 10
    adaptive_gap_low: float = 0.02
    adaptive_gap_high: float = 0.08
    tau: float = 0.8  # for threshold greedy

    # temporal / recency-aware reranking
    temporal_beta: float = 0.0  # weight for freshness bonus (0 disables)
    freshness: Optional[np.ndarray] = None  # (nb,) freshness scores indexed by passage id

    # implementation choices
    mmr_impl: MMRImpl = "incremental"  # incremental is usually faster
    use_max_redundancy: bool = True
    fill_if_insufficient: bool = True  # for threshold greedy


@dataclass
class DiversifiedSearchResponse:
    """Outputs of diversified search."""

    indices: np.ndarray  # (nq, k)
    scores: np.ndarray  # (nq, k) cosine similarity to query

    # Optional debug info (useful while developing)
    candidates: Optional[np.ndarray] = None  # (nq, topN)
    candidate_scores: Optional[np.ndarray] = None  # (nq, topN)
    meta: Optional[Dict] = None


def diversified_search(
    retriever: BaseRetriever,
    xb: np.ndarray,
    request: DiversifiedSearchRequest,
    *,
    return_candidates: bool = False,
) -> DiversifiedSearchResponse:
    """Run diversified top-k search.

    Steps:
      1) Retrieve topN candidates by relevance.
      2) Rerank/reselect to get k results with diversity.
      3) Return indices + relevance scores.

    Notes:
      - This assumes cosine similarity via dot product (vectors are L2-normalized).
      - If your xb/queries are NOT normalized, you should normalize first.

    Returns:
      DiversifiedSearchResponse.
    """

    xb = np.asarray(xb, dtype=np.float32)
    if xb.ndim != 2:
        raise ValueError("xb must be 2D (nb, d)")

    q = np.asarray(request.queries, dtype=np.float32)
    if q.ndim == 1:
        q = q[None, :]
    if q.ndim != 2:
        raise ValueError("queries must be 1D or 2D")

    nq, d = q.shape
    if xb.shape[1] != d:
        raise ValueError(f"dim mismatch: xb has d={xb.shape[1]} but queries have d={d}")

    k = int(request.k)
    topN = int(request.topN)
    if k <= 0:
        raise ValueError("k must be positive")
    if topN < k:
        raise ValueError("topN must be >= k (need enough candidates to diversify)")

    # 1) retrieve
    res = retriever.search(q, topN)
    cand = np.asarray(res.indices, dtype=np.int64)
    cand_scores = np.asarray(res.scores, dtype=np.float32)

    # Some backends can return -1 for missing neighbors; filter if needed.
    # Here we simply keep them and rely on the fact that typical ANN datasets
    # always have enough vectors; you can tighten this later.

    # 2) rerank
    selected = np.empty((nq, k), dtype=np.int32)
    method = request.method.lower()
    adaptive_cfg: Optional[AdaptiveLambdaConfig] = None
    adaptive_lambdas = None
    if method == "mmr" and request.adaptive_lambda:
        adaptive_cfg = AdaptiveLambdaConfig(
            strategy=str(request.adaptive_lambda_strategy),
            lambda_min=float(request.adaptive_lambda_min),
            lambda_max=float(request.adaptive_lambda_max),
            lambda_mid=float(request.adaptive_lambda_mid),
            temperature=float(request.adaptive_lambda_temperature),
            entropy_power=float(request.adaptive_lambda_entropy_power),
            topM=int(request.adaptive_lambda_topM) if int(request.adaptive_lambda_topM) > 0 else None,
            gap_k=int(request.adaptive_gap_k),
            gap_t_low=float(request.adaptive_gap_low),
            gap_t_high=float(request.adaptive_gap_high),
        )
        adaptive_lambdas = np.empty((nq,), dtype=np.float32)

    for i in range(nq):
        cand_i = cand[i]

        if method == "baseline":
            sel = cand_i[:k]
        elif method == "mmr":
            lam = float(request.lambda_)
            if adaptive_cfg is not None:
                lam = adaptive_lambda_from_scores(cand_scores[i], adaptive_cfg)
                if adaptive_lambdas is not None:
                    adaptive_lambdas[i] = lam

            if request.mmr_impl == "full":
                sel = mmr_rerank(
                    q[i],
                    cand_i,
                    xb,
                    k=k,
                    lambda_=lam,
                    use_max_redundancy=bool(request.use_max_redundancy),
                )
            elif request.temporal_beta > 0 and request.freshness is not None:
                sel = mmr_rerank_temporal(
                    q[i],
                    cand_i,
                    xb,
                    k=k,
                    lambda_=lam,
                    freshness=request.freshness,
                    beta=float(request.temporal_beta),
                    use_max_redundancy=bool(request.use_max_redundancy),
                )
            else:
                sel = mmr_rerank_incremental(
                    q[i],
                    cand_i,
                    xb,
                    k=k,
                    lambda_=lam,
                    use_max_redundancy=bool(request.use_max_redundancy),
                )
        elif method == "threshold":
            sel = threshold_greedy_rerank(
                q[i],
                cand_i,
                xb,
                k=k,
                tau=float(request.tau),
                fill_if_insufficient=bool(request.fill_if_insufficient),
            )
        elif method == "maxmin":
            sel = maxmin_rerank(q[i], cand_i, xb, k=k)
        else:
            raise ValueError(f"Unknown method: {request.method}")

        sel = np.asarray(sel, dtype=np.int64).reshape(-1)
        if sel.size != k:
            # In normal use (topN>=k and enough data), this shouldn't happen.
            # But we handle it defensively.
            if sel.size > k:
                sel = sel[:k]
            else:
                pad = np.full((k - sel.size,), -1, dtype=np.int64)
                sel = np.concatenate([sel, pad], axis=0)

        selected[i] = sel.astype(np.int32)

    # 3) compute relevance scores for the selected list
    scores = np.empty((nq, k), dtype=np.float32)
    for i in range(nq):
        ids = selected[i].astype(np.int64)
        # handle padded -1
        mask = ids >= 0
        if not np.any(mask):
            scores[i] = -np.inf
            continue
        vecs = xb[ids[mask]]
        sc = vecs @ q[i]
        out = np.full((k,), -np.inf, dtype=np.float32)
        out[mask] = sc.astype(np.float32)
        scores[i] = out

    lambda_value = float(request.lambda_)
    if adaptive_lambdas is not None:
        lambda_value = float(np.mean(adaptive_lambdas))

    meta = {
        "method": request.method,
        "k": k,
        "topN": topN,
        "lambda": lambda_value,
        "tau": float(request.tau),
        "mmr_impl": request.mmr_impl,
        "use_max_redundancy": bool(request.use_max_redundancy),
        "adaptive_lambda": bool(request.adaptive_lambda),
        "temporal_beta": float(request.temporal_beta),
    }

    if adaptive_cfg is not None and adaptive_lambdas is not None:
        meta.update(
            {
                "adaptive_lambda_strategy": str(adaptive_cfg.strategy),
                "adaptive_lambda_min": float(adaptive_cfg.lambda_min),
                "adaptive_lambda_max": float(adaptive_cfg.lambda_max),
                "adaptive_lambda_mid": float(adaptive_cfg.lambda_mid),
                "adaptive_lambda_temperature": float(adaptive_cfg.temperature),
                "adaptive_lambda_entropy_power": float(adaptive_cfg.entropy_power),
                "adaptive_lambda_topM": (
                    int(adaptive_cfg.topM) if adaptive_cfg.topM is not None else None
                ),
                "adaptive_gap_k": int(adaptive_cfg.gap_k),
                "adaptive_gap_low": float(adaptive_cfg.gap_t_low),
                "adaptive_gap_high": float(adaptive_cfg.gap_t_high),
                "adaptive_lambda_mean": float(np.mean(adaptive_lambdas)),
                "adaptive_lambda_std": float(np.std(adaptive_lambdas)),
            }
        )

    return DiversifiedSearchResponse(
        indices=selected,
        scores=scores,
        candidates=cand.astype(np.int32) if return_candidates else None,
        candidate_scores=cand_scores.astype(np.float32) if return_candidates else None,
        meta=meta,
    )
