from __future__ import annotations

from typing import List

import numpy as np


def threshold_greedy_rerank(
    q: np.ndarray,
    cand_indices: np.ndarray,
    xb: np.ndarray,
    *,
    k: int,
    tau: float,
    fill_if_insufficient: bool = True,
) -> np.ndarray:
    """Greedy selection with a cosine-similarity *threshold* for redundancy.

    We scan candidates in (assumed) descending relevance order (as returned by the retriever).
    A candidate is accepted if its max cosine similarity to the already selected set is <= tau.

    Args:
        q: (d,) normalized query (unused here, kept for a unified interface)
        cand_indices: (N,) candidate ids into xb (assumed sorted by relevance)
        xb: (nb, d) normalized database vectors
        k: final top-k
        tau: redundancy threshold in [-1, 1]. Smaller => more diverse.
        fill_if_insufficient: if True, fill remaining slots with most relevant leftovers.

    Returns:
        (k,) selected ids in selection order.
    """
    cand_indices = np.asarray(cand_indices, dtype=np.int64).reshape(-1)
    N = int(cand_indices.shape[0])
    if N == 0 or k <= 0:
        return cand_indices[:0].astype(np.int32)

    k_eff = int(min(int(k), N))
    cand_vecs = np.asarray(xb[cand_indices], dtype=np.float32)

    selected: List[int] = []
    selected_vecs: List[np.ndarray] = []

    # candidates are already sorted by relevance, so we don't re-sort.
    for i in range(N):
        if len(selected) >= k_eff:
            break
        v = cand_vecs[i]
        if not selected_vecs:
            selected.append(i)
            selected_vecs.append(v)
            continue

        S = np.stack(selected_vecs, axis=0)  # (t, d)
        max_sim = float((v @ S.T).max())
        if max_sim <= float(tau):
            selected.append(i)
            selected_vecs.append(v)

    if len(selected) < k_eff and fill_if_insufficient:
        used = set(selected)
        for i in range(N):
            if len(selected) >= k_eff:
                break
            if i in used:
                continue
            selected.append(i)
            used.add(i)

    return cand_indices[np.array(selected, dtype=np.int64)].astype(np.int32)


def maxmin_rerank(
    q: np.ndarray,
    cand_indices: np.ndarray,
    xb: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """MaxMin / farthest-first rerank within a candidate set.

    - First pick: most relevant candidate (assumed to be cand_indices[0]).
    - Subsequent picks: choose the candidate that MINIMIZES its maximum similarity
      to the already selected set (i.e., as far as possible from the set).

    This is an extreme-diversity baseline.

    Args:
        q: (d,) normalized query (unused here, kept for a unified interface)
        cand_indices: (N,) candidate ids into xb (assumed sorted by relevance)
        xb: (nb, d) normalized database vectors
        k: final top-k

    Returns:
        (k,) selected ids.
    """
    cand_indices = np.asarray(cand_indices, dtype=np.int64).reshape(-1)
    N = int(cand_indices.shape[0])
    if N == 0 or k <= 0:
        return cand_indices[:0].astype(np.int32)

    cand_vecs = np.asarray(xb[cand_indices], dtype=np.float32)
    k_eff = int(min(int(k), N))

    selected: List[int] = [0]  # assume candidates are sorted by relevance
    used = np.zeros(N, dtype=bool)
    used[0] = True

    # max similarity of each candidate to the selected set
    max_sim = cand_vecs @ cand_vecs[0]  # (N,)

    for _ in range(1, k_eff):
        # pick the candidate with smallest redundancy (min max_sim)
        scores = (-max_sim).astype(np.float32, copy=False)
        scores[used] = -np.inf
        best = int(np.argmax(scores))
        selected.append(best)
        used[best] = True

        sims = cand_vecs @ cand_vecs[best]
        max_sim = np.maximum(max_sim, sims)

    return cand_indices[np.array(selected, dtype=np.int64)].astype(np.int32)
