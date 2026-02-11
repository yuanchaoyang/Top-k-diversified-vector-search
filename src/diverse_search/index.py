from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


Backend = Literal["numpy", "numpy_ivf", "faiss"]
IndexType = Literal["flat", "hnsw", "ivf"]


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


class NumpyIVFRetriever(BaseRetriever):
    """纯 NumPy 实现的 IVF (Inverted File Index) 检索器。

    算法流程：
    1. 训练阶段 — K-Means 将 N 条向量聚成 nlist 个簇，得到 nlist 个质心
    2. 建库阶段 — 每条向量分配到最近质心，构建倒排表 inverted_lists[c] = [id1, id2, ...]
    3. 检索阶段 — 查询向量找到最近的 nprobe 个质心，只在这些簇内暴力搜索

    与暴力搜索的核心区别：
    - 暴力搜索计算 q 与全部 N 条向量的相似度 → O(N·d)
    - IVF 只计算 q 与 nprobe 个簇内的向量 → O(nprobe/nlist · N · d)
    - 代价：可能丢失不在 nprobe 个最近簇中的相关向量（召回率 < 100%）
    """

    def __init__(
        self,
        xb: np.ndarray,
        *,
        nlist: int = 128,
        nprobe: int = 16,
        kmeans_iters: int = 20,
        seed: int = 42,
    ):
        xb = np.asarray(xb, dtype=np.float32)
        if xb.ndim != 2:
            raise ValueError("xb must be 2D array")

        self._nlist = int(nlist)
        self.nprobe = int(nprobe)
        self._nb = xb.shape[0]
        self._d = xb.shape[1]
        self.xb = xb

        # ── 第一步：K-Means 训练，得到 nlist 个质心 ──
        self.centroids = self._kmeans(xb, self._nlist, kmeans_iters, seed)
        # centroids: (nlist, d)

        # ── 第二步：建立倒排表 ──
        # 每条向量分配到最近质心
        # assignments[i] = 向量 i 所属的簇编号
        assign_scores = xb @ self.centroids.T          # (N, nlist)
        assignments = np.argmax(assign_scores, axis=1)  # (N,)

        # 倒排表：inverted_lists[c] = 属于簇 c 的向量 id 列表
        self.inverted_lists: list[np.ndarray] = []
        for c in range(self._nlist):
            ids = np.where(assignments == c)[0].astype(np.int32)
            self.inverted_lists.append(ids)

    @staticmethod
    def _kmeans_pp_init(
        xb: np.ndarray, k: int, rng: np.random.Generator,
    ) -> np.ndarray:
        """K-Means++ 初始化（内积空间）。

        比随机初始化收敛更快、簇更均匀。
        1. 随机选第一个质心
        2. 之后每个质心按 D(x)^2 概率选取 —— 离已有质心越远越可能被选中
           对 L2 归一化向量: distance^2 = 2 - 2·cosine = 2·(1 - dot product)
        """
        n, d = xb.shape
        centroids = np.empty((k, d), dtype=np.float32)

        # 第一个质心随机选
        idx = int(rng.integers(n))
        centroids[0] = xb[idx]

        for c in range(1, k):
            # 每个点到已有质心的最大内积（= 最近质心的相似度）
            sims = xb @ centroids[:c].T          # (N, c)
            nearest_sim = np.max(sims, axis=1)   # (N,)

            # D(x)^2 = 2·(1 - nearest_sim)，与 (1 - nearest_sim) 成正比即可
            dist_sq = np.maximum(0.0, 1.0 - nearest_sim)
            total = dist_sq.sum()
            if total < 1e-12:
                # 极端情况：所有点都极近，退化为随机选
                idx = int(rng.integers(n))
            else:
                probs = dist_sq / total
                idx = int(rng.choice(n, p=probs))
            centroids[c] = xb[idx]

        return centroids

    @staticmethod
    def _kmeans(
        xb: np.ndarray, k: int, n_iters: int, seed: int,
    ) -> np.ndarray:
        """K-Means 聚类（内积空间，适用于 L2 归一化向量）。

        1. K-Means++ 选 k 个初始质心（保证初始分散性）
        2. 迭代：每条向量分配到内积最大的质心 → 重新计算质心均值并 L2 归一化
        """
        rng = np.random.default_rng(seed)
        n = xb.shape[0]

        # K-Means++ 初始化
        centroids = NumpyIVFRetriever._kmeans_pp_init(xb, min(k, n), rng)

        for _ in range(n_iters):
            # E-step：计算每条向量到所有质心的内积，分配到最大的
            sims = xb @ centroids.T            # (N, k)
            assignments = np.argmax(sims, axis=1)  # (N,)

            # M-step：重新计算每个簇的质心（均值 + L2 归一化）
            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                members = xb[assignments == c]
                if len(members) > 0:
                    center = members.mean(axis=0)
                    norm = np.linalg.norm(center)
                    new_centroids[c] = center / (norm + 1e-8)
                else:
                    # 空簇：随机重新分配
                    new_centroids[c] = xb[rng.integers(n)]

            centroids = new_centroids

        return centroids

    def set_nprobe(self, nprobe: int) -> None:
        """动态设置搜索时探测的簇数。"""
        self.nprobe = max(1, int(nprobe))

    def get_nprobe(self) -> Optional[int]:
        """返回当前 nprobe。"""
        return self.nprobe

    def search(self, queries: np.ndarray, topN: int) -> SearchResult:
        """IVF 检索：只在最近的 nprobe 个簇内搜索。

        1. 查询向量与 nlist 个质心计算内积 → 找到最近的 nprobe 个
        2. 合并这 nprobe 个簇的倒排表 → 得到候选集
        3. 查询向量只与候选集内的向量计算内积 → 取 top-K
        """
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        nq = q.shape[0]

        all_indices = np.full((nq, topN), -1, dtype=np.int32)
        all_scores = np.full((nq, topN), -1e9, dtype=np.float32)

        # 批量计算查询到所有质心的相似度
        centroid_sims = q @ self.centroids.T  # (nq, nlist)

        for i in range(nq):
            # 找到最近的 nprobe 个质心
            probe_clusters = np.argpartition(
                -centroid_sims[i], kth=min(self.nprobe, self._nlist) - 1,
            )[:self.nprobe]

            # 合并这些簇的倒排表，得到候选 ID
            candidate_ids = np.concatenate(
                [self.inverted_lists[c] for c in probe_clusters
                 if len(self.inverted_lists[c]) > 0]
            )

            if len(candidate_ids) == 0:
                continue

            # 只计算候选集内的相似度（这就是 IVF 加速的核心）
            candidate_vecs = self.xb[candidate_ids]     # (n_cand, d)
            scores = candidate_vecs @ q[i]              # (n_cand,)

            # 取 top-K
            k = min(topN, len(scores))
            if k >= len(scores):
                order = np.argsort(-scores)
            else:
                part_idx = np.argpartition(-scores, kth=k - 1)[:k]
                order = part_idx[np.argsort(-scores[part_idx])]

            all_indices[i, :k] = candidate_ids[order[:k]]
            all_scores[i, :k] = scores[order[:k]]

        return SearchResult(indices=all_indices, scores=all_scores)


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


def _piecewise_exploration(
    intent_score: float,
    temporal_score: float = 0.0,
) -> float:
    """分段映射：intent_score → exploration 系数 [0, 1]。

    三段式设计（比线性映射更精确）：
      intent ∈ [0.0, 0.3) → 歧义区：exploration 从 1.0 陡降到 0.6
      intent ∈ [0.3, 0.7) → 过渡区：exploration 从 0.6 平缓降到 0.3
      intent ∈ [0.7, 1.0] → 明确区：exploration 从 0.3 陡降到 0.0

    对比线性: exploration = 1 - intent
      intent=0.15 → 线性 0.85, 分段 0.80  (差异小)
      intent=0.50 → 线性 0.50, 分段 0.45  (中间区平缓)
      intent=0.85 → 线性 0.15, 分段 0.15  (差异小)
      intent=0.25 → 线性 0.75, 分段 0.67  (歧义区陡)
      intent=0.75 → 线性 0.25, 分段 0.25  (明确区陡)

    关键区别：歧义/明确的边界（0.3/0.7附近）变化更敏感，
    中间过渡区变化平缓，避免"模棱两可"的查询频繁跳变。
    """
    s = float(intent_score)
    if s < 0.3:
        # 歧义区：[0, 0.3) → exploration [1.0, 0.6)  斜率 ≈ -1.33
        t = s / 0.3
        exploration = 1.0 - 0.4 * t
    elif s < 0.7:
        # 过渡区：[0.3, 0.7) → exploration [0.6, 0.3)  斜率 ≈ -0.75
        t = (s - 0.3) / 0.4
        exploration = 0.6 - 0.3 * t
    else:
        # 明确区：[0.7, 1.0] → exploration [0.3, 0.0]  斜率 ≈ -1.0
        t = (s - 0.7) / 0.3
        exploration = 0.3 - 0.3 * t

    # 时效性 boost：时效查询适度扩大探索范围
    temporal_boost = 0.15 * temporal_score
    return min(1.0, max(0.0, exploration + temporal_boost))


def compute_adaptive_nprobe(
    intent_score: float,
    temporal_score: float = 0.0,
    *,
    nprobe_min: int = 8,
    nprobe_max: int = 48,
) -> int:
    """根据 intent 和 temporal 信号计算自适应 nprobe。

    歧义查询（低 intent_score）→ 探索更多聚类桶；
    明确查询（高 intent_score）→ 快速聚焦少量桶。
    使用分段映射，边界区变化更敏感，中间区更平稳。
    """
    combined = _piecewise_exploration(intent_score, temporal_score)
    return max(1, int(nprobe_min + combined * (nprobe_max - nprobe_min)))


def compute_adaptive_topN(
    intent_score: float,
    temporal_score: float = 0.0,
    *,
    topN_min: int = 100,
    topN_max: int = 400,
) -> int:
    """根据 intent 和 temporal 信号计算自适应候选池大小。

    歧义查询（低 intent_score）→ 更大候选池，为多样性重排提供更多素材；
    明确查询（高 intent_score）→ 较小候选池，top 结果已足够相关。
    使用分段映射，边界区变化更敏感，中间区更平稳。
    """
    combined = _piecewise_exploration(intent_score, temporal_score)
    return int(topN_min + combined * (topN_max - topN_min))


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
