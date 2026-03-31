"""检索器工厂和共享类型。

本模块定义基础类型 (SearchResult, BaseRetriever) 和工厂函数 build_retriever()。
具体算法实现拆分到独立文件：
  - brute_force.py → NumpyBruteForceRetriever  (暴力搜索)
  - ivf.py         → NumpyIVFRetriever          (倒排文件索引)
  - hnsw.py        → NumpyHNSWRetriever         (层次化近邻图)

所有公开符号通过本模块 re-export，外部 import 路径不变：
    from diverse_search.index import build_retriever, NumpyIVFRetriever, ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  共享类型
# ═══════════════════════════════════════════════════════════════════

Backend = Literal["numpy", "numpy_ivf", "numpy_hnsw", "faiss"]
IndexType = Literal["flat", "hnsw", "ivf"]


@dataclass
class SearchResult:
    indices: np.ndarray  # (nq, topN)
    scores: np.ndarray  # (nq, topN) higher = more similar


class BaseRetriever:
    def search(self, queries: np.ndarray, topN: int) -> SearchResult:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════
#  Re-export：保持 from diverse_search.index import ... 不变
# ═══════════════════════════════════════════════════════════════════

from diverse_search.brute_force import NumpyBruteForceRetriever  # noqa: E402
from diverse_search.ivf import (  # noqa: E402
    NumpyIVFRetriever,
    compute_adaptive_nprobe,
    compute_adaptive_topN,
)
from diverse_search.hnsw import (  # noqa: E402
    NumpyHNSWRetriever,
    compute_adaptive_ef,
)


# ═══════════════════════════════════════════════════════════════════
#  FaissRetriever (FAISS 封装，可选依赖)
# ═══════════════════════════════════════════════════════════════════

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
        nlist: int = 128,
        nprobe: int = 16,
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
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, int(nlist), faiss.METRIC_INNER_PRODUCT)
            index.train(xb)
            index.nprobe = int(nprobe)
        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        index.add(xb)
        self._faiss = faiss
        self.index = index
        self._index_type = index_type

    def set_nprobe(self, nprobe: int) -> None:
        """动态设置 IVF 探测桶数。仅对 ivf 类型有效。"""
        if self._index_type == "ivf" and hasattr(self.index, "nprobe"):
            self.index.nprobe = max(1, int(nprobe))

    def get_nprobe(self) -> Optional[int]:
        """返回当前 nprobe，非 IVF 返回 None。"""
        if self._index_type == "ivf" and hasattr(self.index, "nprobe"):
            return int(self.index.nprobe)
        return None

    def search(self, queries: np.ndarray, topN: int) -> SearchResult:
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        scores, idx = self.index.search(q, topN)
        # faiss returns (D, I) where D is similarity for IP
        return SearchResult(indices=idx.astype(np.int32), scores=scores.astype(np.float32))


# ═══════════════════════════════════════════════════════════════════
#  工厂函数
# ═══════════════════════════════════════════════════════════════════

def build_retriever(
    xb: np.ndarray,
    *,
    backend: Backend = "numpy",
    index_type: IndexType = "hnsw",
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
    nlist: int = 128,
    nprobe: int = 16,
) -> BaseRetriever:
    """Factory for candidate retriever."""
    backend = backend.lower()  # type: ignore
    if backend == "numpy":
        return NumpyBruteForceRetriever(xb)
    if backend == "numpy_ivf":
        return NumpyIVFRetriever(xb, nlist=nlist, nprobe=nprobe)
    if backend == "numpy_hnsw":
        return NumpyHNSWRetriever(
            xb, M=hnsw_m, ef_construction=ef_construction, ef_search=ef_search,
        )
    if backend == "faiss":
        return FaissRetriever(
            xb,
            index_type=index_type,
            hnsw_m=hnsw_m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            nlist=nlist,
            nprobe=nprobe,
        )
    raise ValueError(f"Unknown backend: {backend}")
