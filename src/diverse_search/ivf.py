"""IVF (Inverted File Index) 检索器 — 纯 NumPy 实现。

将 N 条向量用 K-Means 聚成 nlist 个簇，建立倒排表。
检索时只在最近的 nprobe 个簇内搜索，大幅降低计算量。

复杂度：O(nprobe/nlist · N · d) per query
优点：检索速度快（nprobe << nlist 时数倍加速）
缺点：召回率 < 100%（可能丢失不在探测簇中的相关向量）

自适应参数函数：
- compute_adaptive_nprobe — 根据查询歧义度动态调整探测簇数
- compute_adaptive_topN   — 根据查询歧义度动态调整候选池大小
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from diverse_search.index import BaseRetriever, SearchResult


# ═══════════════════════════════════════════════════════════════════
#  NumpyIVFRetriever
# ═══════════════════════════════════════════════════════════════════

class NumpyIVFRetriever(BaseRetriever):
    """纯 NumPy 实现的 IVF (Inverted File Index) 检索器。

    算法流程：
    1. 训练阶段 — K-Means++ 将 N 条向量聚成 nlist 个簇，得到 nlist 个质心
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

        # 每条向量的簇编号，供簇感知 MMR 使用
        self.assignments = assignments.astype(np.int32)   # (N,)

        # 倒排表：inverted_lists[c] = 属于簇 c 的向量 id 列表
        self.inverted_lists: list[np.ndarray] = []
        for c in range(self._nlist):
            ids = np.where(assignments == c)[0].astype(np.int32)
            self.inverted_lists.append(ids)

    # ─────────────────────────────────────────────
    #  K-Means++ 初始化
    # ─────────────────────────────────────────────

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

    # ─────────────────────────────────────────────
    #  K-Means 聚类
    # ─────────────────────────────────────────────

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

    # ─────────────────────────────────────────────
    #  nprobe 动态调整
    # ─────────────────────────────────────────────

    def set_nprobe(self, nprobe: int) -> None:
        """动态设置搜索时探测的簇数。"""
        self.nprobe = max(1, int(nprobe))

    def get_nprobe(self) -> Optional[int]:
        """返回当前 nprobe。"""
        return self.nprobe

    # ─────────────────────────────────────────────
    #  搜索
    # ─────────────────────────────────────────────

    def search(
        self, queries: np.ndarray, topN: int, *, cluster_balanced: bool = False,
    ) -> SearchResult:
        """IVF 检索：只在最近的 nprobe 个簇内搜索。

        Args:
            queries: 查询向量 (nq, d) 或 (d,)
            topN: 返回的候选数
            cluster_balanced: 簇级均匀采样模式。
                False（默认）: 合并所有簇后按分数全局排序取 top-N，相关性优先。
                True: 每个簇各取 top-per_cluster 条再合并，保证不同簇（≈不同语义）
                      都有代表进入候选池，多样性优先。适合歧义查询。
        """
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        nq = q.shape[0]

        all_indices = np.full((nq, topN), -1, dtype=np.int32)
        all_scores = np.full((nq, topN), -1e9, dtype=np.float32)

        # 批量计算查询到所有质心的相似度
        centroid_sims = q @ self.centroids.T  # (nq, nlist)

        # 簇级均匀采样：每个簇的配额（向上取整，确保总数 ≥ topN）
        per_cluster = max(1, (topN + self.nprobe - 1) // self.nprobe) if cluster_balanced else 0

        for i in range(nq):
            # 找到最近的 nprobe 个质心
            probe_clusters = np.argpartition(
                -centroid_sims[i], kth=min(self.nprobe, self._nlist) - 1,
            )[:self.nprobe]

            if cluster_balanced:
                # ── 簇级均匀采样 ──
                # 每个簇各自取 top-per_cluster，保证多含义均有代表
                collected_ids = []
                collected_scores = []
                for c in probe_clusters:
                    ids_c = self.inverted_lists[c]
                    if len(ids_c) == 0:
                        continue
                    vecs_c = self.xb[ids_c]
                    scores_c = vecs_c @ q[i]
                    # 这个簇内取 top-per_cluster
                    take = min(per_cluster, len(scores_c))
                    if take >= len(scores_c):
                        top_idx = np.argsort(-scores_c)
                    else:
                        part = np.argpartition(-scores_c, kth=take - 1)[:take]
                        top_idx = part[np.argsort(-scores_c[part])]
                    collected_ids.append(ids_c[top_idx])
                    collected_scores.append(scores_c[top_idx])

                if not collected_ids:
                    continue

                candidate_ids = np.concatenate(collected_ids)
                scores = np.concatenate(collected_scores)
            else:
                # ── 全局排序（原逻辑）──
                candidate_ids = np.concatenate(
                    [self.inverted_lists[c] for c in probe_clusters
                     if len(self.inverted_lists[c]) > 0]
                )
                if len(candidate_ids) == 0:
                    continue
                candidate_vecs = self.xb[candidate_ids]
                scores = candidate_vecs @ q[i]

            # 从候选中取 top-N
            k = min(topN, len(scores))
            if k >= len(scores):
                order = np.argsort(-scores)
            else:
                part_idx = np.argpartition(-scores, kth=k - 1)[:k]
                order = part_idx[np.argsort(-scores[part_idx])]

            all_indices[i, :k] = candidate_ids[order[:k]]
            all_scores[i, :k] = scores[order[:k]]

        return SearchResult(indices=all_indices, scores=all_scores)


# ═══════════════════════════════════════════════════════════════════
#  自适应参数：分段映射函数
# ═══════════════════════════════════════════════════════════════════

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
