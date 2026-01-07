from __future__ import annotations

"""Topic/cluster labels for 'coverage' style diversity metrics.

ANN-Benchmarks datasets do not come with semantic metadata (topic, category, ...).
A super practical workaround is to:
  - cluster database vectors into C clusters ("pseudo topics")
  - measure how many different clusters your top-k result covers

This gives you a clean, thesis-friendly diversity metric:
  Coverage@k = (# unique clusters in result) / k

Implementation note:
- For speed on large datasets, we use MiniBatchKMeans.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class KMeansConfig:
    n_clusters: int = 100
    seed: int = 42
    batch_size: int = 2048
    max_iter: int = 100
    n_init: int = 3


def kmeans_cluster_labels(xb: np.ndarray, config: KMeansConfig) -> np.ndarray:
    """Cluster database vectors and return an integer label per vector.

    Args:
        xb: (nb, d) float32 vectors (preferably L2 normalized).
        config: KMeansConfig.

    Returns:
        labels: (nb,) int32 labels in [0, n_clusters).

    Raises:
        ImportError: if scikit-learn is not installed.
    """

    xb = np.asarray(xb, dtype=np.float32)
    if xb.ndim != 2:
        raise ValueError("xb must be 2D (nb, d)")

    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for clustering. Install with `pip install scikit-learn`."
        ) from e

    km = MiniBatchKMeans(
        n_clusters=int(config.n_clusters),
        random_state=int(config.seed),
        batch_size=int(config.batch_size),
        max_iter=int(config.max_iter),
        n_init=int(config.n_init),
        verbose=0,
    )

    labels = km.fit_predict(xb)
    return np.asarray(labels, dtype=np.int32)
