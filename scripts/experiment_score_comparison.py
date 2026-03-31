#!/usr/bin/env python3
"""
Score Comparison — 对比 ambiguous vs specific 查询上各方法的表现差异
===================================================================

产出:
  1. 分组柱状图: ambiguous vs specific 的 Relevance / ILD / F1
  2. 汇总表 CSV

使用方法:
    source .venv/bin/activate
    PYTHONPATH=src python scripts/experiment_score_comparison.py \
      --improved-dir data/improved --k 10 --out-dir outputs/score_comparison
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from experiment_utils import (
    DatasetBundle, load_improved,
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
INTENT_THRESHOLD = 0.4
BASELINE_TOPN = 200

METHODS = [
    "topk", "mmr_0.5", "mmr_0.7", "mmr_0.9", "maxmin",
    "ivf_topk", "ivf_mmr_0.7", "hnsw_topk", "hnsw_mmr_0.7",
    "adaptive",
]
METHOD_LABELS = {
    "topk": "Top-k",
    "mmr_0.5": "MMR λ=0.5",
    "mmr_0.7": "MMR λ=0.7",
    "mmr_0.9": "MMR λ=0.9",
    "maxmin": "MaxMin",
    "ivf_topk": "IVF Top-k",
    "ivf_mmr_0.7": "IVF+MMR 0.7",
    "hnsw_topk": "HNSW Top-k",
    "hnsw_mmr_0.7": "HNSW+MMR 0.7",
    "adaptive": "Adaptive",
}

# IVF / HNSW 固定参数
IVF_NPROBE = 16
HNSW_EF = 64


# ═══════════════════════════════════════════════════════════════
#  Per-query 运行 (复用 experiment_adaptive 逻辑)
# ═══════════════════════════════════════════════════════════════

def run_per_query(
    bundle: DatasetBundle,
    brute, ivf, hnsw,
    classifier: IntentClassifier,
    k: int,
) -> list[dict]:
    """对每个查询运行 6 种方法。"""
    emb = bundle.passage_emb
    temporal_config = TemporalConfig()
    rows: list[dict] = []
    nq = len(bundle.queries)

    for qi in range(nq):
        query_text = bundle.queries[qi]
        q_vec = bundle.query_emb[qi]

        # adaptive 参数
        info = query_aware_lambda_and_beta(
            query_text, classifier, temporal_config,
            lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
        )
        intent_score = info["intent_score"]
        adaptive_lam = info["lambda"]
        adaptive_beta = info["beta"]
        temporal_score = info["temporal_score"]

        if intent_score < INTENT_THRESHOLD:
            retriever_name = "ivf"
            ivf.set_nprobe(compute_adaptive_nprobe(intent_score, temporal_score))
            adaptive_topN = compute_adaptive_topN(intent_score, temporal_score)
        else:
            retriever_name = "hnsw"
            hnsw.set_ef_search(compute_adaptive_ef(intent_score, temporal_score))
            adaptive_topN = compute_adaptive_topN(intent_score, temporal_score)

        cluster_ids = ivf.assignments if hasattr(ivf, "assignments") else None

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
            elif method.startswith("ivf_"):
                # IVF 固定参数 baseline
                ivf.set_nprobe(IVF_NPROBE)
                res = ivf.search(q_vec, BASELINE_TOPN)
                cand = res.indices[0]
                cand = cand[cand >= 0]
                if method == "ivf_topk":
                    selected = cand[:k]
                else:  # ivf_mmr_0.7
                    selected = mmr_rerank_incremental(
                        q_vec, cand, emb, k=k, lambda_=0.7,
                    )
            elif method.startswith("hnsw_"):
                # HNSW 固定参数 baseline
                hnsw.set_ef_search(HNSW_EF)
                res = hnsw.search(q_vec, BASELINE_TOPN)
                cand = res.indices[0]
                cand = cand[cand >= 0]
                if method == "hnsw_topk":
                    selected = cand[:k]
                else:  # hnsw_mmr_0.7
                    selected = mmr_rerank_incremental(
                        q_vec, cand, emb, k=k, lambda_=0.7,
                    )
            else:
                # Brute force baseline
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

            rows.append({
                "query_idx": qi,
                "query": query_text,
                "method": method,
                "intent_score": intent_score,
                "query_type": "ambiguous" if intent_score < INTENT_THRESHOLD else "specific",
                "rel": rel, "ild": ild, "f1": f1,
                "adaptive_lambda": adaptive_lam if method == "adaptive" else None,
            })

        if (qi + 1) % 20 == 0 or qi == nq - 1:
            print(f"    {qi + 1}/{nq} queries done")

    return rows


# ═══════════════════════════════════════════════════════════════
#  图 1: 分组柱状图 — Ambiguous vs Specific
# ═══════════════════════════════════════════════════════════════

def plot_grouped_bars(rows: list[dict], out_dir: Path) -> None:
    """三行图: Relevance / ILD / F1, 每行 ambiguous vs specific 的分组柱状图。"""

    # 按 method × query_type 聚合
    agg: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"rel": [], "ild": [], "f1": []})
    )
    for r in rows:
        a = agg[r["method"]][r["query_type"]]
        a["rel"].append(r["rel"])
        a["ild"].append(r["ild"])
        a["f1"].append(r["f1"])

    metrics = [("Relevance", "rel"), ("Intra-List Diversity (ILD)", "ild"), ("F1 Score", "f1")]
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    x = np.arange(len(METHODS))
    width = 0.35
    colors_ambig = "#5B9BD5"
    colors_spec = "#ED7D31"

    for ax, (metric_label, metric_key) in zip(axes, metrics):
        vals_ambig = []
        vals_spec = []
        for m in METHODS:
            a = agg[m]["ambiguous"][metric_key]
            s = agg[m]["specific"][metric_key]
            vals_ambig.append(np.mean(a) if a else 0)
            vals_spec.append(np.mean(s) if s else 0)

        bars1 = ax.bar(x - width / 2, vals_ambig, width, label="Ambiguous Queries",
                        color=colors_ambig, edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, vals_spec, width, label="Specific Queries",
                        color=colors_spec, edgecolor="white", linewidth=0.5)

        # 数值标注
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6.5)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6.5)

        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=8.5,
                           rotation=25, ha="right")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # 分组分隔线: brute force | ANN | adaptive
        sep1 = 4.5   # maxmin 和 ivf_topk 之间
        sep2 = 8.5   # hnsw_mmr_0.7 和 adaptive 之间
        ax.axvline(x=sep1, color="gray", linestyle=":", alpha=0.4)
        ax.axvline(x=sep2, color="gray", linestyle=":", alpha=0.4)

        # Y 轴范围留些空间给标注
        ymin = min(min(vals_ambig), min(vals_spec))
        ymax = max(max(vals_ambig), max(vals_spec))
        margin = (ymax - ymin) * 0.15
        ax.set_ylim(max(0, ymin - margin), ymax + margin)

    # 在第一个子图顶部加分组标签
    axes[0].text(2.0, axes[0].get_ylim()[1], "Brute Force", ha="center",
                 fontsize=9, fontstyle="italic", color="gray")
    axes[0].text(6.5, axes[0].get_ylim()[1], "ANN (fixed params)", ha="center",
                 fontsize=9, fontstyle="italic", color="gray")
    axes[0].text(9.0, axes[0].get_ylim()[1], "Ours", ha="center",
                 fontsize=9, fontstyle="italic", color="#E91E63")

    fig.suptitle("Score Comparison: Ambiguous vs Specific Queries",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "score_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
#  图 2: Adaptive λ 分布 — 展示自适应行为
# ═══════════════════════════════════════════════════════════════

def plot_lambda_distribution(rows: list[dict], out_dir: Path) -> None:
    """展示 adaptive 对 ambiguous 和 specific 查询给出不同的 λ。"""
    adaptive_rows = [r for r in rows if r["method"] == "adaptive"]

    ambig_lam = [r["adaptive_lambda"] for r in adaptive_rows if r["query_type"] == "ambiguous"]
    spec_lam = [r["adaptive_lambda"] for r in adaptive_rows if r["query_type"] == "specific"]

    fig, ax = plt.subplots(figsize=(8, 4))

    bins = np.linspace(0.25, 0.95, 20)
    ax.hist(ambig_lam, bins=bins, alpha=0.7, color="#5B9BD5",
            label=f"Ambiguous (n={len(ambig_lam)}, mean={np.mean(ambig_lam):.2f})",
            edgecolor="white")
    ax.hist(spec_lam, bins=bins, alpha=0.7, color="#ED7D31",
            label=f"Specific (n={len(spec_lam)}, mean={np.mean(spec_lam):.2f})",
            edgecolor="white")

    ax.set_xlabel("Adaptive λ", fontsize=11)
    ax.set_ylabel("Number of Queries", fontsize=11)
    ax.set_title("Adaptive λ Distribution by Query Type", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # 标注固定 baseline 的 λ 位置
    for lam, label in [(0.5, "MMR λ=0.5"), (0.7, "MMR λ=0.7"), (0.9, "MMR λ=0.9")]:
        ax.axvline(x=lam, color="gray", linestyle="--", alpha=0.5)
        ax.text(lam, ax.get_ylim()[1] * 0.95, label, ha="center", fontsize=7, color="gray")

    fig.tight_layout()
    path = out_dir / "lambda_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
#  汇总表
# ═══════════════════════════════════════════════════════════════

def save_summary_table(rows: list[dict], out_dir: Path) -> None:
    """保存 CSV: method × query_type × (rel, ild, f1)。"""
    agg: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"rel": [], "ild": [], "f1": []})
    )
    for r in rows:
        a = agg[r["method"]][r["query_type"]]
        a["rel"].append(r["rel"])
        a["ild"].append(r["ild"])
        a["f1"].append(r["f1"])

    path = out_dir / "score_comparison.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method", "Query Type", "n_queries",
            "Relevance", "ILD", "F1",
        ])
        for m in METHODS:
            for qt in ["ambiguous", "specific"]:
                a = agg[m][qt]
                n = len(a["f1"])
                writer.writerow([
                    METHOD_LABELS[m], qt, n,
                    f"{np.mean(a['rel']):.4f}",
                    f"{np.mean(a['ild']):.4f}",
                    f"{np.mean(a['f1']):.4f}",
                ])
    print(f"  Saved {path}")

    # 也打印到终端
    print(f"\n  {'Method':14s} {'Type':10s} {'n':>4s} {'Rel':>8s} {'ILD':>8s} {'F1':>8s}")
    print(f"  {'-'*56}")
    for m in METHODS:
        for qt in ["ambiguous", "specific"]:
            a = agg[m][qt]
            n = len(a["f1"])
            print(f"  {METHOD_LABELS[m]:14s} {qt:10s} {n:4d}"
                  f" {np.mean(a['rel']):8.4f} {np.mean(a['ild']):8.4f}"
                  f" {np.mean(a['f1']):8.4f}")
        print()


# ═══════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Comparison — ambiguous vs specific",
    )
    parser.add_argument("--improved-dir", type=str, default="data/improved")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="outputs/score_comparison")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载
    print("[1/4] Loading intent classifier...")
    classifier = IntentClassifier.load("models/intent_v3")

    print("[2/4] Loading improved corpus...")
    bundle = load_improved(str(args.improved_dir))

    print("[3/4] Building retrievers...")
    brute, ivf, hnsw = build_all_retrievers(bundle.passage_emb)

    print("[4/4] Running per-query evaluation...")
    rows = run_per_query(bundle, brute, ivf, hnsw, classifier, args.k)

    # 输出
    print("\nSaving results...")
    save_summary_table(rows, out_dir)
    plot_grouped_bars(rows, out_dir)
    plot_lambda_distribution(rows, out_dir)

    print(f"\nDone. Results in: {out_dir}/")
    print(f"  - score_comparison.png  (主图: 分组柱状图)")
    print(f"  - lambda_distribution.png  (λ 自适应分布)")
    print(f"  - score_comparison.csv  (数据表)")


if __name__ == "__main__":
    main()
