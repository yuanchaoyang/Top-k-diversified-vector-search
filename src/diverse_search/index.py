from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


Backend = Literal["numpy", "faiss"]
IndexType = Literal["flat", "hnsw"]


@dataclass
class SearchResult:
    indices: np.ndarray  # (nq, topN)
    scores: np.ndarray  # (nq, topN) higher = more similar


class BaseRetriever:
    def search(self, queries: np.ndarray, topN: int) -> SearchResult:
        raise NotImplementedError


class NumpyBruteForceRetriever(BaseRetriever):
    """Exact search by dot product (cosine if vectors normalized)."""

    def __init__(self, xb: np.ndarray):
        if xb.ndim != 2:
            raise ValueError("xb must be 2D array")
        self.xb = np.asarray(xb, dtype=np.float32)

    def search(self, queries: np.ndarray, topN: int) -> SearchResult:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]

        # scores = q @ xb.T
        scores = q @ self.xb.T

        # partial topN for speed
        if topN >= scores.shape[1]:
            idx = np.argsort(-scores, axis=1)
            idx = idx[:, :topN]
        else:
            idx_part = np.argpartition(-scores, kth=topN - 1, axis=1)[:, :topN]
            part_scores = np.take_along_axis(scores, idx_part, axis=1)
            order = np.argsort(-part_scores, axis=1)
            idx = np.take_along_axis(idx_part, order, axis=1)

        top_scores = np.take_along_axis(scores, idx, axis=1)
        return SearchResult(indices=idx.astype(np.int32), scores=top_scores.astype(np.float32))


class FaissRetriever(BaseRetriever):
    """ANN retrieval using FAISS.

    Notes:
    - This class is imported lazily; if faiss is not installed, a clear error is raised.
    - We assume vectors are normalized so that inner product equals cosine similarity.
    """

    def __init__(
        self,
        xb: np.ndarray,
        *,
        index_type: IndexType = "hnsw",
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ):
        try:
            import faiss  # type: ignore
        except Exception as e:
            raise ImportError(
                "FAISS is not installed. Install with `pip install faiss-cpu` (or conda)."
            ) from e

        xb = np.asarray(xb, dtype=np.float32)
        d = int(xb.shape[1])

        if index_type == "flat":
            index = faiss.IndexFlatIP(d)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = int(ef_construction)
            index.hnsw.efSearch = int(ef_search)
        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        index.add(xb)
        self._faiss = faiss
        self.index = index

    def search(self, queries: np.ndarray, topN: int) -> SearchResult:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        scores, idx = self.index.search(q, topN)
        # faiss returns (D, I) where D is similarity for IP
        return SearchResult(indices=idx.astype(np.int32), scores=scores.astype(np.float32))


def build_retriever(
    xb: np.ndarray,
    *,
    backend: Backend = "numpy",
    index_type: IndexType = "hnsw",
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
) -> BaseRetriever:
    """Factory for candidate retriever."""
    backend = backend.lower()  # type: ignore
    if backend == "numpy":
        return NumpyBruteForceRetriever(xb)
    if backend == "faiss":
        return FaissRetriever(
            xb,
            index_type=index_type,
            hnsw_m=hnsw_m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
    raise ValueError(f"Unknown backend: {backend}")
