#!/usr/bin/env python3
"""
Adaptive Pipeline Showcase — 展示自适应能力
=============================================

不追求绝对 F1 胜出固定 baseline，改为展示 per-query 自适应能力：
- Adaptation Dashboard: λ / topN / retriever 随 intent 变化
- Robustness Boxplot: F1 分布的稳定性
- Oracle Regret CDF: 逼近 per-query 最优的能力
- Speed-Quality Pareto: 速度与质量的权衡
- Parameter Adaptation Curves: 参数随 intent 平滑变化

使用方法:
    source .venv/bin/activate
    PYTHONPATH=src python scripts/experiment_adaptive.py \
      --improved-dir data/improved --k 10 --out-dir outputs/adaptive

    # 双数据集
    PYTHONPATH=src python scripts/experiment_adaptive.py \
      --improved-dir data/improved --msmarco-dir data/msmarco \
      --k 10 --out-dir outputs/adaptive
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from experiment_utils import (
    DatasetBundle, load_msmarco, load_improved,
    build_all_retrievers, compute_metrics,
)
from diverse_search.mmr import mmr_rerank_incremental, mmr_rerank_temporal
from diverse_search.diversify import maxmin_rerank
from diverse_search.intent_model import IntentClassifier
from diverse_search.temporal import query_aware_lambda_and_beta, TemporalConfig
from diverse_search.ivf import compute_adaptive_nprobe, compute_adaptive_topN
from diverse_search.hnsw import compute_adaptive_ef

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════

LAMBDA_MIN = 0.30
LAMBDA_MAX = 0.90
INTENT_THRESHOLD = 0.4   # < 0.4 → IVF, ≥ 0.4 → HNSW
BASELINE_TOPN = 200

# 6 种方法
METHODS = ["topk", "mmr_0.5", "mmr_0.7", "mmr_0.9", "maxmin", "adaptive"]

# 画图颜色和标记
METHOD_COLORS = {
    "topk": "#999999",
    "mmr_0.5": "#2196F3",
    "mmr_0.7": "#4CAF50",
    "mmr_0.9": "#FF9800",
    "maxmin": "#9C27B0",
    "adaptive": "#E91E63",
}
METHOD_MARKERS = {
    "topk": "o",
    "mmr_0.5": "s",
    "mmr_0.7": "D",
    "mmr_0.9": "^",
    "maxmin": "v",
    "adaptive": "*",
}


# ═══════════════════════════════════════════════════════════════
#  Per-query 运行
# ═══════════════════════════════════════════════════════════════

def run_per_query(
    bundle: DatasetBundle,
    brute,
    ivf,
    hnsw,
    classifier: IntentClassifier,
    k: int,
) -> list[dict]:
    """对每个查询运行 6 种方法，返回 per-query 结果行列表。"""
    emb = bundle.passage_emb
    temporal_config = TemporalConfig()
    rows: list[dict] = []

    nq = len(bundle.queries)
    print(f"  Running {nq} queries × {len(METHODS)} methods ...")

    for qi in range(nq):
        query_text = bundle.queries[qi]
        q_vec = bundle.query_emb[qi]

        # ── adaptive pipeline: 获取参数 ──
        info = query_aware_lambda_and_beta(
            query_text, classifier, temporal_config,
            lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
        )
        intent_score = info["intent_score"]
        adaptive_lam = info["lambda"]
        adaptive_beta = info["beta"]
        temporal_score = info["temporal_score"]

        # 动态选择检索器
        if intent_score < INTENT_THRESHOLD:
            retriever_name = "ivf"
            nprobe = compute_adaptive_nprobe(intent_score, temporal_score)
            ivf.set_nprobe(nprobe)
            adaptive_topN = compute_adaptive_topN(intent_score, temporal_score)
        else:
            retriever_name = "hnsw"
            ef = compute_adaptive_ef(intent_score, temporal_score)
            hnsw.set_ef_search(ef)
            adaptive_topN = compute_adaptive_topN(intent_score, temporal_score)

        # 簇 ID（来自 IVF 的 assignments）
        cluster_ids = ivf.assignments if hasattr(ivf, "assignments") else None

        # ── 对 6 种方法分别运行 ──
        for method in METHODS:
            t0 = time.perf_counter()

            if method == "adaptive":
                # 自适应管线
                if retriever_name == "ivf":
                    res = ivf.search(q_vec, adaptive_topN)
                else:
                    res = hnsw.search(q_vec, adaptive_topN)
                cand = res.indices[0]
                cand = cand[cand >= 0]

                selected = mmr_rerank_temporal(
                    q_vec, cand, emb,
                    k=k, lambda_=adaptive_lam,
                    freshness=bundle.freshness,
                    beta=adaptive_beta,
                    cluster_ids=cluster_ids,
                    cluster_penalty=0.05,
                )
            else:
                # 固定 baseline: brute force + topN=200
                res = brute.search(q_vec, BASELINE_TOPN)
                cand = res.indices[0]
                cand = cand[cand >= 0]

                if method == "topk":
                    selected = cand[:k]
                elif method.startswith("mmr_"):
                    lam = float(method.split("_")[1])
                    selected = mmr_rerank_incremental(
                        q_vec, cand, emb, k=k, lambda_=lam,
                    )
                elif method == "maxmin":
                    selected = maxmin_rerank(q_vec, cand, emb, k=k)
                else:
                    selected = cand[:k]

            elapsed_ms = (time.perf_counter() - t0) * 1000
            rel, ild, f1, ws = compute_metrics(q_vec, selected, emb)

            row = {
                "query_idx": qi,
                "query": query_text,
                "method": method,
                "rel": rel,
                "ild": ild,
                "f1": f1,
                "ws": ws,
                "search_ms": elapsed_ms,
                "intent_score": intent_score,
                "temporal_score": temporal_score,
            }
            # adaptive 特有字段
            if method == "adaptive":
                row["adaptive_lambda"] = adaptive_lam
                row["adaptive_beta"] = adaptive_beta
                row["adaptive_topN"] = adaptive_topN
                row["retriever"] = retriever_name
            else:
                row["adaptive_lambda"] = None
                row["adaptive_beta"] = None
                row["adaptive_topN"] = None
                row["retriever"] = "brute"

            rows.append(row)

        if (qi + 1) % 20 == 0 or qi == nq - 1:
            print(f"    {qi + 1}/{nq} queries done")

    return rows


# ═══════════════════════════════════════════════════════════════
#  分析 1: Adaptation Dashboard
# ═══════════════════════════════════════════════════════════════

def _plot_adaptation_dashboard(
    rows: list[dict],
    out_dir: Path,
    dataset_name: str,
) -> None:
    """4 子图面板: λ / topN / F1 delta / retriever 随 intent 变化。"""
    # 提取 adaptive 行
    adaptive_rows = [r for r in rows if r["method"] == "adaptive"]
    if not adaptive_rows:
        return

    intents = np.array([r["intent_score"] for r in adaptive_rows])
    lambdas = np.array([r["adaptive_lambda"] for r in adaptive_rows])
    topNs = np.array([r["adaptive_topN"] for r in adaptive_rows])
    retrievers = np.array([0 if r["retriever"] == "ivf" else 1 for r in adaptive_rows])

    # 计算 F1 delta: adaptive F1 - best fixed baseline F1 (per query)
    # 构建 per-query F1 dict
    f1_by_query: dict[int, dict[str, float]] = {}
    for r in rows:
        qi = r["query_idx"]
        if qi not in f1_by_query:
            f1_by_query[qi] = {}
        f1_by_query[qi][r["method"]] = r["f1"]

    fixed_methods = [m for m in METHODS if m != "adaptive"]
    f1_deltas = []
    for r in adaptive_rows:
        qi = r["query_idx"]
        best_fixed = max(f1_by_query[qi].get(m, 0.0) for m in fixed_methods)
        f1_deltas.append(r["f1"] - best_fixed)
    f1_deltas = np.array(f1_deltas)

    # 着色: intent < 0.4 → ambiguous (蓝), ≥ 0.4 → specific (橙)
    colors = np.where(intents < INTENT_THRESHOLD, "#2196F3", "#FF9800")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) intent → adaptive λ
    ax = axes[0, 0]
    ax.scatter(intents, lambdas, c=colors, alpha=0.7, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Intent Score")
    ax.set_ylabel("Adaptive λ")
    ax.set_title("(a) λ Adaptation")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="λ=0.5")
    ax.axhline(y=0.7, color="gray", linestyle=":", alpha=0.4, label="λ=0.7")
    ax.axvline(x=INTENT_THRESHOLD, color="red", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)

    # (b) intent → adaptive topN
    ax = axes[0, 1]
    ax.scatter(intents, topNs, c=colors, alpha=0.7, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Intent Score")
    ax.set_ylabel("Adaptive topN")
    ax.set_title("(b) topN Adaptation")
    ax.axvline(x=INTENT_THRESHOLD, color="red", linestyle="--", alpha=0.3)

    # (c) intent → F1 delta
    ax = axes[1, 0]
    pos_mask = f1_deltas >= 0
    ax.scatter(intents[pos_mask], f1_deltas[pos_mask], c="#4CAF50", alpha=0.6,
               label=f"win ({pos_mask.sum()})", edgecolors="white", linewidths=0.5)
    ax.scatter(intents[~pos_mask], f1_deltas[~pos_mask], c="#F44336", alpha=0.6,
               label=f"lose ({(~pos_mask).sum()})", edgecolors="white", linewidths=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axvline(x=INTENT_THRESHOLD, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel("Intent Score")
    ax.set_ylabel("F1(adaptive) − F1(best fixed)")
    ax.set_title("(c) Per-Query F1 Gain")
    ax.legend(fontsize=8)

    # (d) intent → retriever choice
    ax = axes[1, 1]
    ax.scatter(intents, retrievers, c=colors, alpha=0.7, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Intent Score")
    ax.set_ylabel("Retriever (0=IVF, 1=HNSW)")
    ax.set_title("(d) Dynamic Retriever Selection")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["IVF", "HNSW"])
    ax.axvline(x=INTENT_THRESHOLD, color="red", linestyle="--", alpha=0.3, label=f"threshold={INTENT_THRESHOLD}")
    ax.legend(fontsize=8)

    fig.suptitle(f"Adaptation Dashboard — {dataset_name}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / f"adaptation_dashboard_{dataset_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
#  分析 2: Robustness Boxplot
# ═══════════════════════════════════════════════════════════════

def _plot_robustness_boxplot(
    rows: list[dict],
    out_dir: Path,
    dataset_name: str,
) -> None:
    """6 种方法的 F1 箱线图 + 统计注释。"""
    # 收集每种方法的 per-query F1
    f1_data: dict[str, list[float]] = {m: [] for m in METHODS}
    for r in rows:
        f1_data[r["method"]].append(r["f1"])

    fig, ax = plt.subplots(figsize=(10, 6))

    data_list = [f1_data[m] for m in METHODS]
    bp = ax.boxplot(
        data_list,
        labels=METHODS,
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markersize": 6},
    )

    for i, method in enumerate(METHODS):
        bp["boxes"][i].set_facecolor(METHOD_COLORS[method])
        bp["boxes"][i].set_alpha(0.7)

    ax.set_ylabel("F1 Score")
    ax.set_title(f"Robustness Analysis — {dataset_name}")
    ax.grid(axis="y", alpha=0.3)

    # 统计注释
    stats_lines = []
    for m in METHODS:
        vals = np.array(f1_data[m])
        stats_lines.append(
            f"{m:10s}  mean={vals.mean():.3f}±{vals.std():.3f}  "
            f"min={vals.min():.3f}  max={vals.max():.3f}"
        )
    stats_text = "\n".join(stats_lines)
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes, fontsize=7, family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    fig.tight_layout()
    path = out_dir / f"robustness_boxplot_{dataset_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
#  分析 3: Oracle Regret CDF
# ═══════════════════════════════════════════════════════════════

def _plot_regret_cdf(
    rows: list[dict],
    out_dir: Path,
    dataset_name: str,
) -> None:
    """Oracle regret CDF: 每方法一条线。"""
    # 构建 per-query per-method F1
    f1_by_query: dict[int, dict[str, float]] = {}
    for r in rows:
        qi = r["query_idx"]
        if qi not in f1_by_query:
            f1_by_query[qi] = {}
        f1_by_query[qi][r["method"]] = r["f1"]

    query_ids = sorted(f1_by_query.keys())

    # Oracle: per-query best F1 across all methods
    oracle_f1 = {}
    for qi in query_ids:
        oracle_f1[qi] = max(f1_by_query[qi].values())

    fig, ax = plt.subplots(figsize=(8, 6))

    for method in METHODS:
        regrets = []
        for qi in query_ids:
            method_f1 = f1_by_query[qi].get(method, 0.0)
            regrets.append(oracle_f1[qi] - method_f1)
        regrets = np.array(sorted(regrets))
        cdf = np.arange(1, len(regrets) + 1) / len(regrets)
        ax.plot(
            regrets, cdf,
            color=METHOD_COLORS[method],
            label=method,
            linewidth=2 if method == "adaptive" else 1.2,
            linestyle="-" if method == "adaptive" else "--",
        )

    ax.set_xlabel("Oracle Regret (best F1 − method F1)")
    ax.set_ylabel("Fraction of Queries (CDF)")
    ax.set_title(f"Oracle Regret CDF — {dataset_name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = out_dir / f"regret_cdf_{dataset_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
#  分析 4: Speed-Quality Pareto
# ═══════════════════════════════════════════════════════════════

def _plot_speed_quality(
    rows: list[dict],
    out_dir: Path,
    dataset_name: str,
) -> None:
    """散点图: X=avg search_ms, Y=avg F1, 标注 Pareto 前沿。"""
    # 聚合每方法的均值
    agg: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        m = r["method"]
        if m not in agg:
            agg[m] = {"f1": [], "ms": []}
        agg[m]["f1"].append(r["f1"])
        agg[m]["ms"].append(r["search_ms"])

    fig, ax = plt.subplots(figsize=(8, 6))

    points = []  # (ms, f1, method)
    for method in METHODS:
        if method not in agg:
            continue
        avg_ms = np.mean(agg[method]["ms"])
        avg_f1 = np.mean(agg[method]["f1"])
        points.append((avg_ms, avg_f1, method))

        marker_size = 200 if method == "adaptive" else 100
        ax.scatter(
            avg_ms, avg_f1,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            s=marker_size,
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
        )
        offset_y = 0.003
        ax.annotate(
            method, (avg_ms, avg_f1),
            textcoords="offset points", xytext=(8, 5),
            fontsize=8,
        )

    # Pareto 前沿 (lower ms + higher f1 is better)
    # 按 ms 升序排序，只保留 f1 单调递增的点
    points_sorted = sorted(points, key=lambda p: p[0])
    pareto = []
    best_f1 = -1.0
    # 从右到左扫描（最高 f1 在最慢处也可能）
    # 正确做法：按 ms 升序，保留"到目前为止 f1 最高"的点
    # 但 Pareto 前沿定义是：没有另一个点同时更快且更好
    # 简化：按 ms 排序后，从右往左收集 f1 递增的点
    points_by_ms = sorted(points, key=lambda p: p[0])
    pareto_pts = []
    max_f1_seen = -1.0
    for ms, f1, m in reversed(points_by_ms):
        if f1 >= max_f1_seen:
            pareto_pts.append((ms, f1))
            max_f1_seen = f1
    pareto_pts.sort(key=lambda p: p[0])

    if len(pareto_pts) >= 2:
        px = [p[0] for p in pareto_pts]
        py = [p[1] for p in pareto_pts]
        ax.plot(px, py, "k--", alpha=0.4, linewidth=1, label="Pareto frontier")
        ax.legend(fontsize=9)

    ax.set_xlabel("Average Search Time (ms)")
    ax.set_ylabel("Average F1 Score")
    ax.set_title(f"Speed-Quality Pareto — {dataset_name}")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = out_dir / f"speed_quality_{dataset_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
#  分析 5: Parameter Adaptation Curves
# ═══════════════════════════════════════════════════════════════

def _plot_parameter_curves(
    rows: list[dict],
    out_dir: Path,
    dataset_name: str,
) -> None:
    """按 intent_score 排序的 adaptive 参数曲线。"""
    adaptive_rows = [r for r in rows if r["method"] == "adaptive"]
    if not adaptive_rows:
        return

    # 按 intent_score 排序
    adaptive_rows.sort(key=lambda r: r["intent_score"])

    x = np.arange(len(adaptive_rows))
    intents = np.array([r["intent_score"] for r in adaptive_rows])
    lambdas = np.array([r["adaptive_lambda"] for r in adaptive_rows])
    topNs = np.array([r["adaptive_topN"] for r in adaptive_rows])

    # 归一化 topN 到 [0, 1]
    topN_min, topN_max = topNs.min(), topNs.max()
    if topN_max > topN_min:
        topN_norm = (topNs - topN_min) / (topN_max - topN_min)
    else:
        topN_norm = np.zeros_like(topNs)

    # F1 delta (adaptive − mmr_0.7)
    f1_by_query: dict[int, dict[str, float]] = {}
    for r in rows:
        qi = r["query_idx"]
        if qi not in f1_by_query:
            f1_by_query[qi] = {}
        f1_by_query[qi][r["method"]] = r["f1"]

    f1_delta = np.array([
        f1_by_query[r["query_idx"]].get("adaptive", 0)
        - f1_by_query[r["query_idx"]].get("mmr_0.7", 0)
        for r in adaptive_rows
    ])

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(x, lambdas, color="#E91E63", linewidth=1.5, label="Adaptive λ")
    ax1.plot(x, topN_norm, color="#2196F3", linewidth=1.5, label="topN (normalized)")
    ax1.set_xlabel("Query Index (sorted by intent score)")
    ax1.set_ylabel("λ / Normalized topN")
    ax1.set_ylim(-0.1, 1.1)

    ax2 = ax1.twinx()
    ax2.fill_between(x, 0, f1_delta, where=f1_delta >= 0,
                     color="#4CAF50", alpha=0.3, label="F1 gain")
    ax2.fill_between(x, 0, f1_delta, where=f1_delta < 0,
                     color="#F44336", alpha=0.3, label="F1 loss")
    ax2.plot(x, f1_delta, color="gray", linewidth=0.5, alpha=0.5)
    ax2.set_ylabel("F1(adaptive) − F1(mmr_0.7)")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    # 在 X 轴底部标注 intent 范围
    n = len(adaptive_rows)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        idx = min(int(frac * (n - 1)), n - 1)
        ax1.annotate(
            f"intent={intents[idx]:.2f}",
            xy=(idx, -0.05), xycoords=("data", "axes fraction"),
            fontsize=7, color="gray", ha="center",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    ax1.set_title(f"Parameter Adaptation Curves — {dataset_name}")
    fig.tight_layout()
    path = out_dir / f"parameter_curves_{dataset_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
#  CSV 保存
# ═══════════════════════════════════════════════════════════════

def save_results(
    rows: list[dict],
    out_dir: Path,
) -> None:
    """保存 per_query_results.csv 和 summary.csv。"""
    # per-query
    fieldnames = [
        "query_idx", "query", "method", "rel", "ild", "f1", "ws",
        "search_ms", "intent_score", "temporal_score",
        "adaptive_lambda", "adaptive_beta", "adaptive_topN", "retriever",
    ]
    path = out_dir / "per_query_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  Saved {path}")

    # summary
    agg: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        m = r["method"]
        if m not in agg:
            agg[m] = {"rel": [], "ild": [], "f1": [], "ws": [], "ms": []}
        agg[m]["rel"].append(r["rel"])
        agg[m]["ild"].append(r["ild"])
        agg[m]["f1"].append(r["f1"])
        agg[m]["ws"].append(r["ws"])
        agg[m]["ms"].append(r["search_ms"])

    path = out_dir / "summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "mean_rel", "mean_ild", "mean_f1", "std_f1",
            "min_f1", "max_f1", "mean_ms",
        ])
        for m in METHODS:
            if m not in agg:
                continue
            vals = np.array(agg[m]["f1"])
            writer.writerow([
                m,
                f"{np.mean(agg[m]['rel']):.4f}",
                f"{np.mean(agg[m]['ild']):.4f}",
                f"{vals.mean():.4f}",
                f"{vals.std():.4f}",
                f"{vals.min():.4f}",
                f"{vals.max():.4f}",
                f"{np.mean(agg[m]['ms']):.2f}",
            ])
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
#  单数据集实验主流程
# ═══════════════════════════════════════════════════════════════

def run_dataset(
    bundle: DatasetBundle,
    classifier: IntentClassifier,
    k: int,
    out_dir: Path,
) -> None:
    """运行单个数据集的完整实验流程。"""
    dataset_name = bundle.dataset_name
    ds_dir = out_dir / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Passages: {len(bundle.passages)}, Queries: {len(bundle.queries)}")
    print(f"  Output: {ds_dir}")
    print(f"{'='*60}")

    # 构建检索器
    print("\n[1/3] Building retrievers...")
    brute, ivf, hnsw = build_all_retrievers(bundle.passage_emb)

    # per-query 运行
    print("\n[2/3] Running per-query evaluation...")
    rows = run_per_query(bundle, brute, ivf, hnsw, classifier, k)

    # 保存 CSV
    print("\n[3/3] Saving results and plots...")
    save_results(rows, ds_dir)

    # 5 种可视化
    _plot_adaptation_dashboard(rows, ds_dir, dataset_name)
    _plot_robustness_boxplot(rows, ds_dir, dataset_name)
    _plot_regret_cdf(rows, ds_dir, dataset_name)
    _plot_speed_quality(rows, ds_dir, dataset_name)
    _plot_parameter_curves(rows, ds_dir, dataset_name)

    # 打印摘要
    print(f"\n  Summary for {dataset_name}:")
    agg: dict[str, list[float]] = {}
    for r in rows:
        m = r["method"]
        if m not in agg:
            agg[m] = []
        agg[m].append(r["f1"])
    for m in METHODS:
        if m in agg:
            vals = np.array(agg[m])
            print(f"    {m:12s}  F1={vals.mean():.4f}±{vals.std():.4f}")


# ═══════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive Pipeline Showcase — 展示自适应能力",
    )
    parser.add_argument("--improved-dir", type=str, default="data/improved",
                        help="improved corpus 目录")
    parser.add_argument("--msmarco-dir", type=str, default=None,
                        help="MS MARCO 数据目录 (可选)")
    parser.add_argument("--k", type=int, default=10, help="最终返回 top-k")
    parser.add_argument("--out-dir", type=str, default="outputs/adaptive",
                        help="输出目录")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载 intent 分类器
    print("[Init] Loading intent classifier...")
    classifier = IntentClassifier.load("models/intent_v3")

    # 主数据集: improved
    improved_dir = Path(args.improved_dir)
    if (improved_dir / "passages.txt").exists():
        print("\n[Dataset] Loading improved corpus...")
        bundle_improved = load_improved(str(improved_dir))
        run_dataset(bundle_improved, classifier, args.k, out_dir)
    else:
        print(f"[WARN] Improved corpus not found at {improved_dir}, skipping.")

    # 辅助数据集: msmarco (可选)
    if args.msmarco_dir:
        msmarco_dir = Path(args.msmarco_dir)
        if (msmarco_dir / "msmarco_passages.txt").exists():
            print("\n[Dataset] Loading MS MARCO corpus...")
            bundle_msmarco = load_msmarco(str(msmarco_dir))
            run_dataset(bundle_msmarco, classifier, args.k, out_dir)
        else:
            print(f"[WARN] MS MARCO data not found at {msmarco_dir}, skipping.")

    print("\n[Done] All experiments completed.")
    print(f"  Results in: {out_dir}")


if __name__ == "__main__":
    main()
