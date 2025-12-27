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
