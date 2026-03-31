"""暴力搜索 (Brute-Force) 检索器 — 精确最近邻。

对查询向量与全部 N 条数据库向量计算内积（L2 归一化后等价于余弦相似度），
返回分数最高的 topN 条结果。

复杂度：O(N·d) per query
优点：召回率 100%，无近似误差
缺点：N 大时延迟高
"""
from __future__ import annotations

import numpy as np

from diverse_search.index import BaseRetriever, SearchResult


class NumpyBruteForceRetriever(BaseRetriever):
    """Exact search by dot product (cosine if vectors normalized).

    算法：
    1. 计算 scores = queries @ xb.T            — (nq, N) 相似度矩阵
    2. 对每个查询取 top-N（用 argpartition 加速） — O(N) per query
    3. 返回排序后的 (indices, scores)
    """

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
