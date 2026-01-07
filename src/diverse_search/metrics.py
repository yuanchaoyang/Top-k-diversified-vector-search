from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def recall_at_k(
    pred: np.ndarray,
    gt: np.ndarray,
    k: int,
    *,
    max_index: Optional[int] = None,
) -> float:
    """Compute Recall@k against ground-truth neighbor ids.

    Args:
        pred: (nq, k) predicted ids.
        gt:   (nq, >=k) ground truth ids.
        k:    cutoff.
        max_index: Optional upper bound for valid ids in gt.
            This is useful when you sub-sample the database vectors: GT ids that
            point outside the retained subset would be impossible to retrieve.

    Returns:
        Recall@k in [0, 1].
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
    denom = 0

    # For each query, count intersection size between pred and gt_k.
    # If max_index is provided, ignore gt ids >= max_index.
    for i in range(pred.shape[0]):
        pred_i = set(pred[i, :k].tolist())
        gt_i = gt_k[i]
        if max_index is not None:
            gt_i = gt_i[gt_i < int(max_index)]
        gt_set = set(gt_i.tolist())
        if len(gt_set) == 0:
            continue
        hit += len(pred_i & gt_set)
        denom += len(gt_set)

    # If denom==0, return NaN to signal recall is not computable.
    return float(hit) / float(denom) if denom > 0 else float("nan")


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


def coverage_unique(labels: np.ndarray) -> int:
    """Number of unique labels (topics/clusters) in a result list."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0
    # ignore "unknown" labels if you use -1 as a sentinel
    labels = labels[labels >= 0]
    return int(np.unique(labels).size)


@dataclass
class AggMetrics:
    recall: Optional[float]
    rel_mean_cos: float
    redundancy_avg_cos: float
    redundancy_max_cos: float
    coverage_unique: Optional[float] = None
    coverage_ratio: Optional[float] = None

    @property
    def ild(self) -> float:
        return 1.0 - self.redundancy_avg_cos
