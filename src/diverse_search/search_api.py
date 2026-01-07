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

from .index import BaseRetriever
from .diversify import maxmin_rerank, threshold_greedy_rerank
from .mmr import mmr_rerank, mmr_rerank_incremental


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
    tau: float = 0.8  # for threshold greedy

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

    for i in range(nq):
        cand_i = cand[i]

        if method == "baseline":
            sel = cand_i[:k]
        elif method == "mmr":
            if request.mmr_impl == "full":
                sel = mmr_rerank(
                    q[i],
                    cand_i,
                    xb,
                    k=k,
                    lambda_=float(request.lambda_),
                    use_max_redundancy=bool(request.use_max_redundancy),
                )
            else:
                sel = mmr_rerank_incremental(
                    q[i],
                    cand_i,
                    xb,
                    k=k,
                    lambda_=float(request.lambda_),
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

    meta = {
        "method": request.method,
        "k": k,
        "topN": topN,
        "lambda": float(request.lambda_),
        "tau": float(request.tau),
        "mmr_impl": request.mmr_impl,
        "use_max_redundancy": bool(request.use_max_redundancy),
    }

    return DiversifiedSearchResponse(
        indices=selected,
        scores=scores,
        candidates=cand.astype(np.int32) if return_candidates else None,
        candidate_scores=cand_scores.astype(np.float32) if return_candidates else None,
        meta=meta,
    )
