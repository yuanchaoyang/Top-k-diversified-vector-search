"""HNSW (Hierarchical Navigable Small World) 检索器 — 纯 NumPy 实现。

构建一个多层近邻图，上层节点稀疏（"高速公路"），逐层加密到底层包含全部向量。
检索时从顶层入口点贪心下降，在底层做 beam search，兼顾速度和高召回。

与 IVF 的核心区别：
  IVF  — 聚类 + 倒排表，搜索范围由 nprobe 控制（扫描几个簇）
  HNSW — 多层近邻图，搜索范围由 ef_search 控制（beam 宽度）

复杂度：
  建图  O(N · log(N) · M · d)   — 逐条插入，每条在各层做 beam search
  搜索  O(log(N) · ef · M · d)  — 从顶层贪心下降 + 底层 beam search

关键参数：
  M              — 每层每个节点的最大邻居数（默认 16）
  ef_construction — 建图时 beam 宽度（越大图质量越高，建图越慢）
  ef_search      — 搜索时 beam 宽度（越大召回越高，搜索越慢）
"""
from __future__ import annotations

import heapq
from typing import Optional

import numpy as np

from diverse_search.index import BaseRetriever, SearchResult


# ═══════════════════════════════════════════════════════════════════
#  NumpyHNSWRetriever
# ═══════════════════════════════════════════════════════════════════

class NumpyHNSWRetriever(BaseRetriever):
    """纯 NumPy 实现的 HNSW 检索器。

    图结构示意（M=2 简化）：

        Layer 2:  [0] ─────────────────── [42]
        Layer 1:  [0] ──── [7] ──── [25] ── [42]
        Layer 0:  [0]-[1]-[3]-[7]-[12]-[18]-[25]-[33]-[42]-...  (全部 N 个节点)

    每个节点在 layer 0 必定存在，是否出现在更高层由几何分布随机决定。
    高层节点少、边跨度大 → 快速"跳跃"；低层节点多、边跨度小 → 精确定位。
    """

    def __init__(
        self,
        xb: np.ndarray,
        *,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int = 42,
    ):
        xb = np.asarray(xb, dtype=np.float32)
        if xb.ndim != 2:
            raise ValueError("xb must be 2D array")

        self.xb = xb
        self._nb, self._d = xb.shape

        # ── 超参数 ──
        self.M = int(M)                          # 每层最大邻居数
        self.M_max0 = 2 * self.M                  # layer 0 允许更多邻居（最密集层）
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.mL = 1.0 / np.log(max(self.M, 2))   # 层级生成因子

        # ── 图结构 ──
        # graph[layer] = { node_id: [neighbor_id, ...] }
        self.graph: list[dict[int, list[int]]] = [{}]
        self.node_level = np.zeros(self._nb, dtype=np.int32)  # 每个节点的最高层
        self.entry_point: int = -1   # 全局入口点
        self.max_layer: int = 0      # 当前图的最高层

        # ── 逐条插入，构建图 ──
        rng = np.random.default_rng(seed)
        for i in range(self._nb):
            self._insert(i, rng)
            if (i + 1) % 5000 == 0 or i == self._nb - 1:
                print(f"  HNSW build: {i + 1}/{self._nb} nodes inserted, "
                      f"layers={self.max_layer + 1}")

    # ─────────────────────────────────────────────
    #  随机层级生成
    # ─────────────────────────────────────────────

    def _random_level(self, rng: np.random.Generator) -> int:
        """为新节点生成随机层级（几何分布）。

        level = floor(-ln(uniform) * mL)
        mL = 1/ln(M)

        大多数节点 level=0，少数到 1，极少到 2+。
        这保证了上层稀疏、下层稠密的"跳表"结构。
        """
        r = float(rng.random())
        if r == 0.0:
            r = 1e-15
        return int(-np.log(r) * self.mL)

    # ─────────────────────────────────────────────
    #  单层 Beam Search
    # ─────────────────────────────────────────────

    def _search_layer(
        self,
        q_vec: np.ndarray,
        entry_points: list[int],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """在单层图上执行 beam search。

        维护两个堆：
          candidates (最大堆) — 待探索的节点，按相似度从高到低弹出
          results    (最小堆) — 目前找到的最好 ef 个结果

        终止条件：当前最好的 candidate 比 results 中最差的还差 → 不可能更好了。

        Args:
            q_vec: 查询向量 (d,)
            entry_points: 入口节点列表
            ef: beam 宽度（保留的最佳结果数）
            layer: 搜索的层级

        Returns:
            [(similarity, node_id), ...] 按相似度降序排列
        """
        visited: set[int] = set(entry_points)

        # candidates: 最大堆（用负数模拟，弹出最相似的）
        candidates: list[tuple[float, int]] = []
        # results: 最小堆（弹出最不相似的，用于裁剪到 ef 大小）
        results: list[tuple[float, int]] = []

        # 批量计算入口点的相似度
        ep_array = np.array(entry_points, dtype=np.int64)
        ep_sims = self.xb[ep_array] @ q_vec   # (len(entry_points),)

        for idx, ep in enumerate(entry_points):
            sim = float(ep_sims[idx])
            heapq.heappush(candidates, (-sim, ep))   # 负数 → 最大堆
            heapq.heappush(results, (sim, ep))        # 正数 → 最小堆

        while candidates:
            # 弹出最相似的候选
            neg_sim_c, c = heapq.heappop(candidates)
            sim_c = -neg_sim_c

            # results 中最差的
            sim_worst = results[0][0]

            # 剪枝：最好的候选都不如 results 中最差的 → 结束
            if sim_c < sim_worst and len(results) >= ef:
                break

            # 获取 c 在该层的邻居
            neighbors = self.graph[layer].get(c, [])
            if not neighbors:
                continue

            # 过滤已访问
            new_neighbors = [n for n in neighbors if n not in visited]
            if not new_neighbors:
                continue
            visited.update(new_neighbors)

            # 批量计算邻居的相似度
            n_array = np.array(new_neighbors, dtype=np.int64)
            n_sims = self.xb[n_array] @ q_vec   # (len(new_neighbors),)

            for j, n in enumerate(new_neighbors):
                sim_n = float(n_sims[j])
                sim_worst = results[0][0]

                if sim_n > sim_worst or len(results) < ef:
                    heapq.heappush(candidates, (-sim_n, n))
                    heapq.heappush(results, (sim_n, n))
                    if len(results) > ef:
                        heapq.heappop(results)  # 裁剪到 ef 大小

        # 按相似度降序返回
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    # ─────────────────────────────────────────────
    #  节点插入
    # ─────────────────────────────────────────────

    def _insert(self, node_id: int, rng: np.random.Generator) -> None:
        """将一个新节点插入 HNSW 图。

        步骤：
        1. 随机生成该节点的最高层 level
        2. 从全局入口点出发，在 level+1 ~ max_layer 各层贪心找最近 1 个节点 → 下降
        3. 在 level ~ 0 各层用 beam search (ef=ef_construction) 找邻居
        4. 双向连边，如果邻居超载则裁剪
        """
        level = self._random_level(rng)
        self.node_level[node_id] = level

        # 扩展图的层数
        while len(self.graph) <= level:
            self.graph.append({})

        q_vec = self.xb[node_id]

        # 第一个节点：直接设为入口点
        if self.entry_point == -1:
            self.entry_point = node_id
            self.max_layer = level
            for l in range(level + 1):
                self.graph[l][node_id] = []
            return

        ep = self.entry_point

        # ── Phase 1: 从顶层贪心下降到 level+1 ──
        # 每层只找最近的 1 个节点作为下一层的入口
        for l in range(self.max_layer, level, -1):
            results = self._search_layer(q_vec, [ep], ef=1, layer=l)
            if results:
                ep = results[0][1]

        # ── Phase 2: 在 min(max_layer, level) ~ 0 各层插入 ──
        for l in range(min(self.max_layer, level), -1, -1):
            # beam search 找候选邻居
            results = self._search_layer(q_vec, [ep], ef=self.ef_construction, layer=l)

            # 该层的最大邻居数
            M_eff = self.M_max0 if l == 0 else self.M

            # 取最相似的 M_eff 个作为邻居
            neighbors = results[:M_eff]

            # 在图中为新节点创建邻居表
            self.graph[l][node_id] = [nid for _, nid in neighbors]

            # 双向连边 + 邻居超载裁剪
            for _, nid in neighbors:
                nid_neighbors = self.graph[l].get(nid, [])
                nid_neighbors.append(node_id)

                if len(nid_neighbors) > M_eff:
                    # 超过最大邻居数 → 保留最相似的 M_eff 个
                    nid_vec = self.xb[nid]
                    n_ids = np.array(nid_neighbors, dtype=np.int64)
                    sims = self.xb[n_ids] @ nid_vec
                    top_indices = np.argsort(-sims)[:M_eff]
                    nid_neighbors = [nid_neighbors[j] for j in top_indices]

                self.graph[l][nid] = nid_neighbors

            # 用本层最近的结果作为下一层的入口
            if results:
                ep = results[0][1]

        # 更新全局入口点（如果新节点在更高层）
        if level > self.max_layer:
            self.entry_point = node_id
            self.max_layer = level

    # ─────────────────────────────────────────────
    #  ef_search 动态调整
    # ─────────────────────────────────────────────

    def set_ef_search(self, ef_search: int) -> None:
        """动态设置搜索时的 beam 宽度。"""
        self.ef_search = max(1, int(ef_search))

    def get_ef_search(self) -> Optional[int]:
        """返回当前 ef_search。"""
        return self.ef_search

    # ─────────────────────────────────────────────
    #  搜索
    # ─────────────────────────────────────────────

    def search(self, queries: np.ndarray, topN: int) -> SearchResult:
        """HNSW 检索。

        对每个查询：
        1. 从顶层 entry_point 贪心下降到 layer 1（每层 ef=1，只追踪最近的）
        2. 在 layer 0 用 ef_search 大小的 beam search 找候选
        3. 取 top-N 返回

        Args:
            queries: (nq, d) 或 (d,)
            topN: 返回数

        Returns:
            SearchResult(indices, scores)
        """
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        nq = q.shape[0]

        all_indices = np.full((nq, topN), -1, dtype=np.int32)
        all_scores = np.full((nq, topN), -1e9, dtype=np.float32)

        for i in range(nq):
            q_vec = q[i]
            ep = self.entry_point

            # 从顶层逐层贪心下降（每层只找最近 1 个）
            for l in range(self.max_layer, 0, -1):
                results = self._search_layer(q_vec, [ep], ef=1, layer=l)
                if results:
                    ep = results[0][1]

            # 在 layer 0 做完整 beam search
            ef = max(self.ef_search, topN)
            results = self._search_layer(q_vec, [ep], ef=ef, layer=0)

            k = min(topN, len(results))
            for j in range(k):
                all_scores[i, j] = results[j][0]
                all_indices[i, j] = results[j][1]

        return SearchResult(indices=all_indices, scores=all_scores)


# ═══════════════════════════════════════════════════════════════════
#  自适应 ef_search
# ═══════════════════════════════════════════════════════════════════

def compute_adaptive_ef(
    intent_score: float,
    temporal_score: float = 0.0,
    *,
    ef_min: int = 32,
    ef_max: int = 200,
) -> int:
    """根据 intent 和 temporal 信号计算自适应 ef_search。

    与 IVF 的自适应 nprobe 对称：
      歧义查询（低 intent_score）→ 更大 beam 宽度，图上游走更广
      明确查询（高 intent_score）→ 较小 beam 宽度，快速收敛

    使用与 nprobe/topN 相同的分段映射函数。
    """
    from diverse_search.ivf import _piecewise_exploration
    combined = _piecewise_exploration(intent_score, temporal_score)
    return max(1, int(ef_min + combined * (ef_max - ef_min)))
