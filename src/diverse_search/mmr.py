from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class MMRConfig:
    k: int
    lambda_: float = 0.5
    # if True: redundancy uses max similarity to selected set (classic)
    # if False: redundancy uses mean similarity to selected set
    use_max_redundancy: bool = True


def mmr_select_local_indices(
    q: np.ndarray,
    cand_vecs: np.ndarray,
    config: MMRConfig,
) -> List[int]:
    """MMR greedy selection returning indices in [0, len(cand_vecs)).

    Args:
        q: (d,) normalized query
        cand_vecs: (N, d) normalized candidate vectors
        config: MMRConfig

    Returns:
        Selected local indices of size k (or less if N < k).
    """
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    cand_vecs = np.asarray(cand_vecs, dtype=np.float32)

    N = int(cand_vecs.shape[0])
    k = int(min(config.k, N))
    lam = float(config.lambda_)

    # relevance: cosine(q, x) = x @ q when normalized
    sim_q = cand_vecs @ q  # (N,)

    # pairwise candidate similarity: (N,N)
    sim_mat = cand_vecs @ cand_vecs.T

    selected: List[int] = []
    used = np.zeros(N, dtype=bool)

    for _ in range(k):
        best_i = -1
        best_score = -1e18

        for i in range(N):
            if used[i]:
                continue

            if not selected:
                redundancy = 0.0
            else:
                sims = sim_mat[i, selected]
                redundancy = float(sims.max()) if config.use_max_redundancy else float(sims.mean())

            score = lam * float(sim_q[i]) - (1.0 - lam) * redundancy

            if score > best_score:
                best_score = score
                best_i = i

        if best_i < 0:
            break

        selected.append(best_i)
        used[best_i] = True

    return selected


def mmr_rerank(
    q: np.ndarray,
    cand_indices: np.ndarray,
    xb: np.ndarray,
    *,
    k: int,
    lambda_: float,
    use_max_redundancy: bool = True,
) -> np.ndarray:
    """Rerank candidate ids with MMR and return selected ids.

    Args:
        q: (d,) normalized query
        cand_indices: (N,) candidate ids into xb
        xb: (nb, d) normalized database vectors
        k: final top-k
        lambda_: tradeoff parameter in [0, 1]
        use_max_redundancy: redundancy aggregation

    Returns:
        (k,) selected ids (subset of cand_indices) in selection order.
    """
    cand_indices = np.asarray(cand_indices, dtype=np.int64).reshape(-1)
    cand_vecs = xb[cand_indices]

    sel_local = mmr_select_local_indices(
        q,
        cand_vecs,
        config=MMRConfig(k=k, lambda_=lambda_, use_max_redundancy=use_max_redundancy),
    )

    return cand_indices[np.array(sel_local, dtype=np.int64)].astype(np.int32)


def mmr_rerank_incremental(
    q: np.ndarray,
    cand_indices: np.ndarray,
    xb: np.ndarray,
    *,
    k: int,
    lambda_: float,
    use_max_redundancy: bool = True,
    min_score_ratio: float = 0.0,
) -> np.ndarray:
    """MMR rerank with *incremental* redundancy updates.

    This version avoids building the full (N,N) candidate similarity matrix.
    Instead it maintains, for each candidate, its redundancy to the selected set
    and updates it in O(N·d) each time we add one new selected vector.

    Complexity per query:
      - time:  O(k · N · d)
      - memory: O(N)

    Args:
        q: (d,) normalized query
        cand_indices: (N,) candidate ids into xb (should be topN by relevance)
        xb: (nb, d) normalized database vectors
        k: final top-k
        lambda_: tradeoff parameter in [0, 1]
        use_max_redundancy: if True use max similarity to selected set; else mean
        min_score_ratio: minimum relevance as a fraction of the top candidate's
            score. Candidates below ``top_score * min_score_ratio`` are excluded
            before MMR selection so that diversity never pulls in irrelevant
            items.  0.0 (default) disables the filter.

    Returns:
        (k,) selected ids (subset of cand_indices) in selection order.
    """
    cand_indices = np.asarray(cand_indices, dtype=np.int64).reshape(-1)
    N = int(cand_indices.shape[0])
    if N == 0 or k <= 0:
        return cand_indices[:0].astype(np.int32)

    cand_vecs = np.asarray(xb[cand_indices], dtype=np.float32)
    q = np.asarray(q, dtype=np.float32).reshape(-1)

    # --- relevance floor: drop low-relevance candidates before MMR ---
    if min_score_ratio > 0.0:
        sim_all = cand_vecs @ q
        top_score = float(sim_all.max())
        keep = sim_all >= top_score * min_score_ratio
        if keep.sum() >= k:
            cand_indices = cand_indices[keep]
            cand_vecs = cand_vecs[keep]
            N = int(cand_vecs.shape[0])

    k_eff = int(min(int(k), N))
    lam = float(lambda_)

    # relevance: cosine(q, x) = x @ q when normalized
    sim_q = cand_vecs @ q  # (N,)

    selected: List[int] = []
    used = np.zeros(N, dtype=bool)

    # pick the most relevant item first (standard in diversified retrieval)
    first = int(np.argmax(sim_q))
    selected.append(first)
    used[first] = True

    # redundancy trackers for all candidates
    # - max redundancy: keep max similarity to any selected
    # - mean redundancy: keep sum similarity to selected, divide by |S|
    max_sim = np.zeros(N, dtype=np.float32)
    sum_sim = np.zeros(N, dtype=np.float32)

    sims = cand_vecs @ cand_vecs[first]  # (N,)
    if use_max_redundancy:
        max_sim = sims
    else:
        sum_sim = sims

    for t in range(1, k_eff):
        if use_max_redundancy:
            redundancy = max_sim
        else:
            redundancy = sum_sim / float(t)

        scores = lam * sim_q - (1.0 - lam) * redundancy
        scores = scores.astype(np.float32, copy=False)
        scores[used] = -np.inf

        best = int(np.argmax(scores))
        selected.append(best)
        used[best] = True

        sims = cand_vecs @ cand_vecs[best]
        if use_max_redundancy:
            max_sim = np.maximum(max_sim, sims)
        else:
            sum_sim += sims

    return cand_indices[np.array(selected, dtype=np.int64)].astype(np.int32)
