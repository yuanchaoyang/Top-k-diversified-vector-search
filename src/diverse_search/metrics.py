from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def recall_at_k(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Compute Recall@k against ground truth neighbors.

    pred: (nq, k) predicted ids
    gt:   (nq, >=k) ground truth ids
    """
    pred = np.asarray(pred, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64)

    if pred.ndim != 2:
        raise ValueError("pred must be 2D")
    if gt.ndim != 2:
        raise ValueError("gt must be 2D")

    k = int(k)
    gt_k = gt[:, :k]
    hit = 0
    total = pred.shape[0] * k

    # For each query, count intersection size between pred and gt_k
    for i in range(pred.shape[0]):
        hit += len(set(pred[i, :k].tolist()) & set(gt_k[i].tolist()))

    return float(hit) / float(total)


def mean_cosine_to_query(q: np.ndarray, selected_vecs: np.ndarray) -> float:
    """Mean cosine similarity between query and selected vectors.

    Assumes q and selected_vecs are L2 normalized.
    """
    q = np.asarray(q, dtype=np.float32).reshape(1, -1)
    V = np.asarray(selected_vecs, dtype=np.float32)
    return float((V @ q.T).mean())


def avg_pairwise_cosine(selected_vecs: np.ndarray) -> float:
    """Average pairwise cosine similarity within a list.

    Lower => more diverse.
    """
    V = np.asarray(selected_vecs, dtype=np.float32)
    k = V.shape[0]
    if k < 2:
        return 0.0
    gram = V @ V.T
    # take upper triangle without diagonal
    iu = np.triu_indices(k, k=1)
    return float(gram[iu].mean())


def max_pairwise_cosine(selected_vecs: np.ndarray) -> float:
    """Maximum pairwise cosine similarity within a list.

    Lower => less redundancy.
    """
    V = np.asarray(selected_vecs, dtype=np.float32)
    k = V.shape[0]
    if k < 2:
        return 0.0
    gram = V @ V.T
    iu = np.triu_indices(k, k=1)
    return float(gram[iu].max())


@dataclass
class AggMetrics:
    recall: Optional[float]
    rel_mean_cos: float
    redundancy_avg_cos: float
    redundancy_max_cos: float

    @property
    def ild(self) -> float:
        return 1.0 - self.redundancy_avg_cos
