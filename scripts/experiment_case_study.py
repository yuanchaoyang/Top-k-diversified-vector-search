#!/usr/bin/env python3
"""
Case Study — 定性案例展示
=========================

挑选代表性查询，side-by-side 展示各方法的 top-k 实际 passage，
让读者直观看到完整自适应管线相比 baseline 的多样性提升。

方法 (3 种):
  baseline        — brute force top-k，无重排（高冗余）
  fixed_mmr       — brute force + MMR λ=0.7（固定多样性）
  full_pipeline   — 动态 IVF/HNSW + 自适应 λ + 簇感知 MMR + 质量过滤

使用方法:
    source .venv/bin/activate
    PYTHONPATH=src python scripts/experiment_case_study.py \
      --msmarco-dir data/msmarco --improved-dir data/improved \
      --k 5 --out-dir outputs/case_study
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from experiment_utils import (
    DatasetBundle, load_msmarco, load_improved,
    build_all_retrievers, compute_metrics, filter_low_quality,
    _extract_title, AMBIGUOUS_WORDS,
)
from diverse_search.mmr import mmr_rerank_incremental, mmr_rerank_temporal
from diverse_search.intent_model import IntentClassifier
from diverse_search.temporal import (
    query_aware_lambda_and_beta, TemporalConfig,
)
from diverse_search.ivf import compute_adaptive_nprobe, compute_adaptive_topN
from diverse_search.hnsw import compute_adaptive_ef

# ═══════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════

LAMBDA_MIN = 0.30
LAMBDA_MAX = 0.90
INTENT_THRESHOLD = 0.4

# improved corpus ambiguous 查询：主选 + 备选
_IMPROVED_AMBIG_PRIMARY = ["crane", "spring", "cell", "bank"]
_IMPROVED_AMBIG_BACKUP = ["mercury", "seal", "bass", "apple"]


# ═══════════════════════════════════════════════════════════════
#  Improved corpus: 自动选择多义查询
# ═══════════════════════════════════════════════════════════════

def _select_improved_ambiguous(
    brute,
    emb: np.ndarray,
    k: int = 10,
    ild_threshold: float = 0.5,
) -> list[str]:
    """从主选+备选列表中选 ILD > threshold 的多义查询。"""
    from experiment_utils import encode_queries
    from diverse_search.metrics import avg_pairwise_cosine

    candidates = _IMPROVED_AMBIG_PRIMARY + _IMPROVED_AMBIG_BACKUP
    vecs = encode_queries(candidates)
    selected = []

    for word, vec in zip(candidates, vecs):
        res = brute.search(vec.reshape(1, -1), k)
        sel_vecs = emb[res.indices[0]]
        avg_pw = float(avg_pairwise_cosine(sel_vecs))
        ild = 1.0 - avg_pw
        if ild > ild_threshold:
            selected.append(word)
        if len(selected) >= 4:
            break

    if len(selected) < 2:
        selected = _IMPROVED_AMBIG_PRIMARY[:4]

    return selected[:4]


# ═══════════════════════════════════════════════════════════════
#  查询选择
# ═══════════════════════════════════════════════════════════════

def select_case_study_queries(
    bundle: DatasetBundle,
    intent_classifier: IntentClassifier,
    brute,
    *,
    n_per_type: int = 4,
    seed: int = 42,
) -> dict[str, list[tuple[str, np.ndarray, dict]]]:
    """挑选 ambiguous / specific (/ temporal for msmarco) 各 n_per_type 条查询。"""
    from experiment_utils import encode_queries

    rng = np.random.default_rng(seed)
    result = {"ambiguous": [], "specific": []}

    if bundle.dataset_name == "improved":
        # ── Improved: 自动选 ambiguous ──
        ambig_words = _select_improved_ambiguous(brute, bundle.passage_emb, k=10)
        print(f"  Selected ambiguous queries: {ambig_words}")
        for q in ambig_words[:n_per_type]:
            vec = encode_queries([q])[0]
            ta = query_aware_lambda_and_beta(
                q, intent_classifier,
                temporal_config=TemporalConfig(),
                lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
            )
            result["ambiguous"].append((q, vec, ta))

        # specific: 从 passage_titles 挑
        unique_titles = list({t for t in bundle.passage_titles if 10 < len(t) < 60})
        if unique_titles:
            idx = rng.choice(len(unique_titles),
                             size=min(n_per_type, len(unique_titles)), replace=False)
            for i in idx:
                q = unique_titles[i]
                vec = encode_queries([q])[0]
                ta = query_aware_lambda_and_beta(
                    q, intent_classifier,
                    temporal_config=TemporalConfig(),
                    lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
                )
                result["specific"].append((q, vec, ta))

    else:
        # ── MS MARCO: 自动筛选 ──
        result["temporal"] = []
        print("  Analyzing all queries for case study selection...")
        candidates = {"ambiguous": [], "specific": [], "temporal": []}

        for i in range(len(bundle.queries)):
            q = bundle.queries[i]
            vec = bundle.query_emb[i]
            ta = query_aware_lambda_and_beta(
                q, intent_classifier,
                temporal_config=TemporalConfig(),
                lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
            )
            intent = ta["intent_score"]
            temporal = ta["temporal_score"]

            if intent < 0.3:
                candidates["ambiguous"].append((q, vec, ta))
            if intent > 0.7:
                candidates["specific"].append((q, vec, ta))
            if temporal > 0.3:
                candidates["temporal"].append((q, vec, ta))

        for qtype in ["ambiguous", "specific", "temporal"]:
            pool = candidates[qtype]
            if pool:
                n = min(n_per_type, len(pool))
                idx = rng.choice(len(pool), size=n, replace=False)
                result[qtype] = [pool[i] for i in idx]
            print(f"    {qtype}: {len(pool)} candidates "
                  f"-> {len(result.get(qtype, []))} selected")

    return result


# ═══════════════════════════════════════════════════════════════
#  单查询: 3 种方法对比
# ═══════════════════════════════════════════════════════════════

def run_case_methods(
    query: str,
    query_vec: np.ndarray,
    ta: dict,
    emb: np.ndarray,
    passages: list[str],
    brute, ivf, hnsw,
    *,
    k: int,
    freshness: Optional[np.ndarray] = None,
) -> list[dict]:
    """对单条查询运行 3 种方法，返回每种方法的 top-k 结果。"""
    q = query_vec
    intent_score = ta["intent_score"]
    intent_lam = ta["lambda"]
    temporal_score = ta["temporal_score"]
    beta = ta["beta"]
    cluster_ids = getattr(ivf, "assignments", None)

    methods_results = []

    # ── 1) baseline: brute force top-k, 无重排 ──
    res = brute.search(q.reshape(1, -1), 200)
    selected = res.indices[0][:k]
    methods_results.append(_build_method_result(
        "baseline", "top-k, no rerank",
        query, selected, emb, passages, q,
    ))

    # ── 2) fixed_mmr: brute force + MMR λ=0.7 ──
    selected = mmr_rerank_incremental(q, res.indices[0], emb, k=k, lambda_=0.7)
    methods_results.append(_build_method_result(
        "fixed_mmr", "MMR \u03bb=0.70",
        query, selected, emb, passages, q,
    ))

    # ── 3) full_pipeline: 完整自适应管线 ──
    # Step A: 动态检索器选择 + 自适应参数
    adaptive_topN = compute_adaptive_topN(intent_score, temporal_score)

    if intent_score < INTENT_THRESHOLD:
        # 歧义查询 → IVF (簇结构促进多样性)
        nprobe = compute_adaptive_nprobe(intent_score, temporal_score)
        ivf.set_nprobe(nprobe)
        pipe_res = ivf.search(q.reshape(1, -1), adaptive_topN, cluster_balanced=True)
        use_cluster = True
        retriever_name = f"IVF nprobe={nprobe}"
    else:
        # 明确查询 → HNSW (高召回保相关性)
        ef = compute_adaptive_ef(intent_score, temporal_score)
        hnsw.set_ef_search(ef)
        pipe_res = hnsw.search(q.reshape(1, -1), adaptive_topN)
        use_cluster = False
        retriever_name = f"HNSW ef={ef}"

    cands = pipe_res.indices[0]
    scores = pipe_res.scores[0]

    # Step B: quality boost — 候选质量低时提高 λ 保相关性
    l_lam = intent_lam
    top_score = float(scores[0]) if len(scores) > 0 else 0.0
    if top_score < 0.5:
        quality_boost = (0.5 - top_score) * 1.0
        l_lam = min(0.9, l_lam + quality_boost)

    # Step C: 簇感知 MMR 重排 + 严格候选过滤
    min_ratio = 0.65
    c_ids = cluster_ids if use_cluster else None
    gamma = 0.15 if use_cluster else 0.0

    selected = mmr_rerank_temporal(
        q, cands, emb, k=k, lambda_=float(l_lam),
        freshness=freshness, beta=float(beta),
        min_score_ratio=min_ratio,
        cluster_ids=c_ids, cluster_penalty=gamma,
    )

    # Step D: 绝对分数地板
    selected = filter_low_quality(selected, q, emb, score_floor=0.35, min_keep=2)

    desc = f"{retriever_name} + MMR \u03bb={l_lam:.2f} + cluster"
    methods_results.append(_build_method_result(
        "full_pipeline", desc,
        query, selected, emb, passages, q,
    ))

    return methods_results


def _build_method_result(
    method: str, desc: str,
    query: str,
    selected: np.ndarray, emb: np.ndarray,
    passages: list[str], query_vec: np.ndarray,
) -> dict:
    """构建方法结果字典。"""
    results_list = []
    sims = np.dot(emb[selected], query_vec)
    for rank, idx in enumerate(selected):
        p = passages[idx]
        results_list.append({
            "rank": rank + 1,
            "passage_idx": int(idx),
            "score": float(sims[rank]),
            "title": _extract_title(p),
            "snippet": _extract_snippet(p, query),
        })

    return {
        "method": method, "desc": desc, "query": query,
        "results": results_list,
    }


def _extract_snippet(passage: str, query: str) -> str:
    """提取 query-relevant snippet (2-3 句)。"""
    sentences = re.split(r'(?<=[.?!])\s+', passage.strip())
    if len(sentences) <= 2:
        snippet = passage
    else:
        query_words = set(query.lower().split())
        scored = []
        for i, sent in enumerate(sentences):
            sent_words = set(sent.lower().split())
            overlap = len(query_words & sent_words)
            scored.append((overlap - i * 0.1, i, sent))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = sorted(scored[:3], key=lambda x: x[1])
        snippet = " ".join(t[2] for t in top)
    if len(snippet) > 200:
        snippet = snippet[:197] + "..."
    return snippet


# ═══════════════════════════════════════════════════════════════
#  输出
# ═══════════════════════════════════════════════════════════════

def print_case_study(
    query_type: str,
    query: str,
    ta: dict,
    method_results: list[dict],
    k: int,
):
    """格式化打印单个查询的 case study。"""
    print(f"\n{'━'*90}")
    print(f"  Query [{query_type}]: \"{query}\"")
    print(f"  intent={ta['intent_score']:.2f}  \u03bb={ta['lambda']:.2f}")
    print(f"{'━'*90}")

    for mr in method_results:
        m = mr["method"]
        desc = mr["desc"]

        print(f"\n  ┌─ {m}  ({desc})")

        for r in mr["results"][:k]:
            print(f"  │ {r['rank']}. [{r['score']:.3f}] {r['title']}")

        print(f"  └{'─'*60}")


def save_case_study(
    all_results: list[dict],
    all_metrics: list[dict],
    out_dir: Path,
):
    """保存 CSV 文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "case_study_results.csv"
    fields = ["dataset", "query_type", "query", "method",
              "rank", "passage_idx", "score", "title", "snippet"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSaved: {csv_path}")

    csv_path = out_dir / "case_study_metrics.csv"
    metric_fields = ["dataset", "query_type", "query", "method", "desc"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=metric_fields)
        w.writeheader()
        w.writerows(all_metrics)
    print(f"Saved: {csv_path}")


# ═══════════════════════════════════════════════════════════════
#  对比可视化
# ═══════════════════════════════════════════════════════════════

def _plot_case_comparison(all_results: list[dict], out_dir: Path):
    """每个查询一个子图，展示各方法 top-k 结果的相似度分布（箱线图）。

    让读者直观看到：baseline 分数集中在高处（冗余），
    full_pipeline 分数分布更宽（覆盖不同方面）。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping case comparison plot")
        return

    from collections import OrderedDict
    queries_order = OrderedDict()
    for r in all_results:
        key = (r["dataset"], r["query_type"], r["query"])
        if key not in queries_order:
            queries_order[key] = {}
        m = r["method"]
        if m not in queries_order[key]:
            queries_order[key][m] = []
        queries_order[key][m].append(r["score"])

    n_queries = len(queries_order)
    if n_queries == 0:
        return

    ncols = min(4, n_queries)
    nrows = (n_queries + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    if n_queries == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    method_order = ["baseline", "fixed_mmr", "full_pipeline"]
    colors = {"baseline": "#a0a0a0", "fixed_mmr": "#4e79a7", "full_pipeline": "#e15759"}

    for idx, ((ds, qtype, qtext), method_scores) in enumerate(queries_order.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        methods_present = [m for m in method_order if m in method_scores]
        data = [method_scores[m] for m in methods_present]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        for patch, m in zip(bp["boxes"], methods_present):
            patch.set_facecolor(colors.get(m, "#999"))
            patch.set_alpha(0.7)

        # 散点叠加
        for i, (m, scores) in enumerate(zip(methods_present, data)):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(scores))
            ax.scatter(np.full(len(scores), i + 1) + jitter, scores,
                       color=colors.get(m, "#999"), s=30, alpha=0.8,
                       edgecolors="white", linewidths=0.5, zorder=3)

        ax.set_xticks(range(1, len(methods_present) + 1))
        ax.set_xticklabels(methods_present, fontsize=8, rotation=15)
        ax.set_ylabel("Score")
        title = f"[{qtype}] {qtext}"
        if len(title) > 35:
            title = title[:32] + "..."
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    for idx in range(n_queries, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Score Distribution per Method", fontsize=13, y=1.01)
    plt.tight_layout()
    path = out_dir / "case_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def run_case_study(
    bundle: DatasetBundle,
    intent_classifier: IntentClassifier,
    *,
    k: int = 5,
    n_per_type: int = 4,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """对一个数据集运行完整 case study。"""
    print(f"\n{'='*70}")
    print(f"  Case Study: {bundle.dataset_name}")
    print(f"{'='*70}")

    # 构建三种检索器
    print("Building retrievers...")
    brute, ivf, hnsw = build_all_retrievers(bundle.passage_emb)

    # 挑选查询
    print("Selecting representative queries...")
    selected_queries = select_case_study_queries(
        bundle, intent_classifier, brute,
        n_per_type=n_per_type, seed=seed,
    )

    all_result_rows = []
    all_metric_rows = []

    for qtype, query_list in selected_queries.items():
        for query, query_vec, ta in query_list:
            method_results = run_case_methods(
                query, query_vec, ta,
                bundle.passage_emb, bundle.passages,
                brute, ivf, hnsw,
                k=k, freshness=bundle.freshness,
            )

            print_case_study(qtype, query, ta, method_results, k)

            for mr in method_results:
                all_metric_rows.append({
                    "dataset": bundle.dataset_name,
                    "query_type": qtype,
                    "query": query,
                    "method": mr["method"],
                    "desc": mr["desc"],
                })

                for r in mr["results"]:
                    all_result_rows.append({
                        "dataset": bundle.dataset_name,
                        "query_type": qtype,
                        "query": query,
                        "method": mr["method"],
                        "rank": r["rank"],
                        "passage_idx": r["passage_idx"],
                        "score": round(r["score"], 4),
                        "title": r["title"],
                        "snippet": r["snippet"],
                    })

    return all_result_rows, all_metric_rows


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Case study: qualitative comparison")
    parser.add_argument("--msmarco-dir", default=None)
    parser.add_argument("--improved-dir", default=None)
    parser.add_argument("--k", type=int, default=5,
                        help="每种方法展示的结果数")
    parser.add_argument("--n-per-type", type=int, default=4,
                        help="每种查询类型挑选的查询数")
    parser.add_argument("--out-dir", default="outputs/case_study")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.msmarco_dir is None and args.improved_dir is None:
        parser.error("至少指定 --msmarco-dir 或 --improved-dir")

    intent_classifier = IntentClassifier()
    for model_dir in ["models/intent_v3", "models/intent_v2", "models/intent_chatgpt"]:
        model_path = Path(model_dir) / "intent_model.pkl"
        if model_path.exists():
            intent_classifier = IntentClassifier.load(model_dir)
            print(f"Intent classifier loaded from {model_dir}")
            break
    else:
        print("No intent model found, using default classifier")

    all_results = []
    all_metrics = []

    if args.msmarco_dir and Path(args.msmarco_dir).exists():
        bundle = load_msmarco(args.msmarco_dir)
        results, metrics = run_case_study(
            bundle, intent_classifier,
            k=args.k, n_per_type=args.n_per_type, seed=args.seed,
        )
        all_results.extend(results)
        all_metrics.extend(metrics)

    if args.improved_dir and Path(args.improved_dir).exists():
        bundle = load_improved(args.improved_dir, seed=args.seed)
        results, metrics = run_case_study(
            bundle, intent_classifier,
            k=args.k, n_per_type=args.n_per_type, seed=args.seed,
        )
        all_results.extend(results)
        all_metrics.extend(metrics)

    out_dir = Path(args.out_dir)
    save_case_study(all_results, all_metrics, out_dir)
    _plot_case_comparison(all_results, out_dir)

    print("\nCase study complete!")


if __name__ == "__main__":
    main()
