#!/usr/bin/env python3
"""
Quantitative Ablation — 全量消融对比实验
=========================================

36 种 baseline 网格 (retriever × topN × reranking) + L1-L5 分层消融，
产出论文主结果表和可视化。

使用方法:
    source .venv/bin/activate
    PYTHONPATH=src python scripts/experiment_quantitative.py \
      --msmarco-dir data/msmarco --improved-dir data/improved \
      --k 10 --out-dir outputs/quantitative
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
    AMBIGUOUS_WORDS, encode_queries,
)
from diverse_search.mmr import mmr_rerank_incremental, mmr_rerank_temporal
from diverse_search.diversify import threshold_greedy_rerank, maxmin_rerank
from diverse_search.adaptive_lambda import adaptive_lambda_from_scores, AdaptiveLambdaConfig
from diverse_search.intent_model import IntentClassifier
from diverse_search.temporal import (
    query_aware_lambda_and_beta, TemporalConfig, classify_temporal,
)
from diverse_search.ivf import compute_adaptive_nprobe, compute_adaptive_topN
from diverse_search.hnsw import compute_adaptive_ef

# ═══════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════

TOPN_VALUES = [50, 100, 200, 400]
RETRIEVERS = ["brute", "ivf", "hnsw"]
RERANKS = ["topk", "mmr0.5", "mmr0.7"]

LAMBDA_MIN = 0.65
LAMBDA_MAX = 0.75
INTENT_THRESHOLD = 0.4


def mix_ambiguous_queries(
    bundle: DatasetBundle,
    n_ambiguous: int = 15,
    n_specific: int = 15,
    seed: int = 42,
) -> DatasetBundle:
    """构建均衡的混合查询集: n_ambiguous 个歧义词 + n_specific 个 MS MARCO 查询。

    使用分层采样确保两种查询都有充分代表，展示 pipeline 对不同 intent 的自适应能力。
    """
    rng = np.random.default_rng(seed)

    # 1) 采样歧义查询
    n_ambig = min(n_ambiguous, len(AMBIGUOUS_WORDS))
    ambig_idx = rng.choice(len(AMBIGUOUS_WORDS), size=n_ambig, replace=False)
    ambig_queries = [AMBIGUOUS_WORDS[i] for i in ambig_idx]

    print(f"  Encoding {n_ambig} ambiguous queries...")
    ambig_emb = encode_queries(ambig_queries)

    # 2) 采样 specific 查询
    n_spec = min(n_specific, len(bundle.queries))
    spec_idx = rng.choice(len(bundle.queries), size=n_spec, replace=False)
    spec_queries = [bundle.queries[i] for i in spec_idx]
    spec_emb = bundle.query_emb[spec_idx]

    print(f"  Mixed query set: {n_ambig} ambiguous + {n_spec} specific = {n_ambig + n_spec}")

    new_queries = ambig_queries + spec_queries
    new_emb = np.vstack([ambig_emb, spec_emb])

    return DatasetBundle(
        passages=bundle.passages,
        passage_emb=bundle.passage_emb,
        queries=new_queries,
        query_emb=new_emb,
        passage_titles=bundle.passage_titles,
        passage_sources=bundle.passage_sources,
        freshness=bundle.freshness,
        dataset_name=bundle.dataset_name,
    )


# ═══════════════════════════════════════════════════════════════
#  单查询: 全方法评估
# ═══════════════════════════════════════════════════════════════

def run_all_methods_single_query(
    query: str,
    query_vec: np.ndarray,
    emb: np.ndarray,
    brute, ivf, hnsw,
    intent_classifier: IntentClassifier,
    *,
    k: int,
    freshness: Optional[np.ndarray] = None,
) -> list[dict]:
    """对单条查询运行所有 baseline + 消融层，返回结果列表。"""
    rows = []
    q = query_vec

    # ── Step 1: 检索候选 (缓存所有 retriever × topN 组合) ──
    # brute force: 检索最大 topN=400，小的直接截取
    t0 = time.perf_counter()
    brute_res = brute.search(q.reshape(1, -1), max(TOPN_VALUES))
    brute_ms = (time.perf_counter() - t0) * 1000
    brute_cache = {}
    for tn in TOPN_VALUES:
        brute_cache[tn] = (brute_res.indices[0][:tn], brute_res.scores[0][:tn], brute_ms)

    # IVF / HNSW: 每个 topN 分别检索
    ivf_cache = {}
    for tn in TOPN_VALUES:
        t0 = time.perf_counter()
        res = ivf.search(q.reshape(1, -1), tn)
        ms = (time.perf_counter() - t0) * 1000
        ivf_cache[tn] = (res.indices[0], res.scores[0], ms)

    hnsw_cache = {}
    for tn in TOPN_VALUES:
        t0 = time.perf_counter()
        res = hnsw.search(q.reshape(1, -1), tn)
        ms = (time.perf_counter() - t0) * 1000
        hnsw_cache[tn] = (res.indices[0], res.scores[0], ms)

    retriever_caches = {"brute": brute_cache, "ivf": ivf_cache, "hnsw": hnsw_cache}

    # ── Step 2: Intent 分析 (先算, baseline 行也需要 intent_score 用于分组) ──
    cands_bf200, scores_bf200, ms_bf200 = retriever_caches["brute"][200]

    ta = query_aware_lambda_and_beta(
        query, intent_classifier,
        temporal_config=TemporalConfig(),
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
    )
    intent_score = ta["intent_score"]
    intent_lam = ta["lambda"]
    temporal_score = ta["temporal_score"]
    beta = ta["beta"]

    # 分数自适应 λ (L1 用)
    score_cfg = AdaptiveLambdaConfig(
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX, strategy="hybrid",
    )
    score_lam = adaptive_lambda_from_scores(scores_bf200, score_cfg)

    cluster_ids = getattr(ivf, "assignments", None)

    # ── Step 3: Baseline 网格 (36 组合 + 2 额外多样性 baseline) ──
    for ret_name in RETRIEVERS:
        for tn in TOPN_VALUES:
            cands, scores, search_ms = retriever_caches[ret_name][tn]

            for rerank in RERANKS:
                method_name = f"{ret_name}_{tn}_{rerank}"
                t0 = time.perf_counter()

                if rerank == "topk":
                    # 直接取前 k 个
                    selected = cands[:k]
                elif rerank == "mmr0.5":
                    selected = mmr_rerank_incremental(
                        q, cands, emb, k=k, lambda_=0.5,
                    )
                elif rerank == "mmr0.7":
                    selected = mmr_rerank_incremental(
                        q, cands, emb, k=k, lambda_=0.7,
                    )
                else:
                    continue

                rerank_ms = (time.perf_counter() - t0) * 1000
                rel, ild, f1, ws = compute_metrics(q, selected, emb)
                rows.append(_make_row(
                    method_name, query, rel, ild, f1, ws,
                    retriever=ret_name, topN=tn, rerank=rerank,
                    intent_score=intent_score,
                    search_ms=search_ms + rerank_ms,
                ))

    # 额外多样性 baselines
    cands_200, scores_200, ms_200 = retriever_caches["brute"][200]

    t0 = time.perf_counter()
    sel_threshold = threshold_greedy_rerank(q, cands_200, emb, k=k, tau=0.5)
    ms_t = (time.perf_counter() - t0) * 1000
    rel, ild, f1, ws = compute_metrics(q, sel_threshold, emb)
    rows.append(_make_row(
        "brute_200_threshold0.5", query, rel, ild, f1, ws,
        retriever="brute", topN=200, rerank="threshold0.5",
        intent_score=intent_score,
        search_ms=ms_200 + ms_t,
    ))

    t0 = time.perf_counter()
    sel_maxmin = maxmin_rerank(q, cands_200, emb, k=k)
    ms_m = (time.perf_counter() - t0) * 1000
    rel, ild, f1, ws = compute_metrics(q, sel_maxmin, emb)
    rows.append(_make_row(
        "brute_200_maxmin", query, rel, ild, f1, ws,
        retriever="brute", topN=200, rerank="maxmin",
        intent_score=intent_score,
        search_ms=ms_200 + ms_m,
    ))

    # ── Step 4: 分层消融 L1-L5 ──
    t0 = time.perf_counter()
    sel_l1 = mmr_rerank_incremental(q, cands_bf200, emb, k=k, lambda_=score_lam)
    ms_l1 = (time.perf_counter() - t0) * 1000
    rel, ild, f1, ws = compute_metrics(q, sel_l1, emb)
    rows.append(_make_row(
        "L1_score_adaptive", query, rel, ild, f1, ws,
        retriever="brute", topN=200, rerank="mmr_score",
        lambda_=score_lam, intent_score=intent_score,
        search_ms=ms_bf200 + ms_l1,
    ))

    # L2: +intent — brute force + intent-aware λ
    t0 = time.perf_counter()
    sel_l2 = mmr_rerank_incremental(q, cands_bf200, emb, k=k, lambda_=intent_lam)
    ms_l2 = (time.perf_counter() - t0) * 1000
    rel, ild, f1, ws = compute_metrics(q, sel_l2, emb)
    rows.append(_make_row(
        "L2_+intent", query, rel, ild, f1, ws,
        retriever="brute", topN=200, rerank="mmr_intent",
        lambda_=intent_lam, intent_score=intent_score,
        search_ms=ms_bf200 + ms_l2,
    ))

    # L3: +cluster — 簇感知 MMR (仍用 brute force 候选, 加 IVF 聚类信号)
    l3_cluster_ids = cluster_ids   # IVF cluster assignments 对所有向量可用
    l3_gamma = 0.12 if intent_score < INTENT_THRESHOLD else 0.06

    t0 = time.perf_counter()
    sel_l3 = mmr_rerank_incremental(
        q, cands_bf200, emb, k=k, lambda_=intent_lam,
        cluster_ids=l3_cluster_ids, cluster_penalty=l3_gamma,
    )
    ms_l3 = (time.perf_counter() - t0) * 1000
    rel, ild, f1, ws = compute_metrics(q, sel_l3, emb)
    rows.append(_make_row(
        "L3_+cluster", query, rel, ild, f1, ws,
        retriever="brute", topN=200, rerank="mmr_cluster",
        lambda_=intent_lam, intent_score=intent_score,
        search_ms=ms_bf200 + ms_l3,
    ))

    # L4: +temporal — 加时效性信号
    t0 = time.perf_counter()
    sel_l4 = mmr_rerank_temporal(
        q, cands_bf200, emb, k=k, lambda_=float(intent_lam),
        freshness=freshness, beta=float(beta),
        cluster_ids=l3_cluster_ids, cluster_penalty=l3_gamma,
    )
    ms_l4 = (time.perf_counter() - t0) * 1000
    rel, ild, f1, ws = compute_metrics(q, sel_l4, emb)
    rows.append(_make_row(
        "L4_+temporal", query, rel, ild, f1, ws,
        retriever="brute", topN=200, rerank="mmr_temporal",
        lambda_=intent_lam, intent_score=intent_score,
        temporal_score=temporal_score, beta=beta,
        search_ms=ms_bf200 + ms_l4,
    ))

    # L5: +dyn_index — 动态 IVF/HNSW 检索 (速度优化, 质量接近 brute force)
    adaptive_topN = compute_adaptive_topN(intent_score, temporal_score,
                                          topN_min=100, topN_max=300)

    if intent_score < INTENT_THRESHOLD:
        nprobe = compute_adaptive_nprobe(intent_score, temporal_score)
        ivf.set_nprobe(nprobe)
        t0 = time.perf_counter()
        res_l5 = ivf.search(q.reshape(1, -1), adaptive_topN)
        ms_l5_search = (time.perf_counter() - t0) * 1000
        l5_index = "ivf"
    else:
        ef = compute_adaptive_ef(intent_score, temporal_score)
        hnsw.set_ef_search(ef)
        t0 = time.perf_counter()
        res_l5 = hnsw.search(q.reshape(1, -1), adaptive_topN)
        ms_l5_search = (time.perf_counter() - t0) * 1000
        l5_index = "hnsw"

    cands_l5 = res_l5.indices[0]
    t0 = time.perf_counter()
    sel_l5 = mmr_rerank_temporal(
        q, cands_l5, emb, k=k, lambda_=float(intent_lam),
        freshness=freshness, beta=float(beta),
        cluster_ids=l3_cluster_ids, cluster_penalty=l3_gamma,
    )
    ms_l5_rerank = (time.perf_counter() - t0) * 1000
    rel, ild, f1, ws = compute_metrics(q, sel_l5, emb)
    rows.append(_make_row(
        "L5_full", query, rel, ild, f1, ws,
        retriever=l5_index, topN=adaptive_topN, rerank="mmr_full",
        lambda_=intent_lam, intent_score=intent_score,
        temporal_score=temporal_score, beta=beta,
        search_ms=ms_l5_search + ms_l5_rerank,
    ))

    return rows


def _make_row(
    method: str, query: str,
    rel: float, ild: float, f1: float, ws: float,
    *, retriever: str = "", topN: int = 0, rerank: str = "",
    lambda_: float = 0.0, intent_score: float = 0.0,
    temporal_score: float = 0.0, beta: float = 0.0,
    search_ms: float = 0.0,
) -> dict:
    return {
        "method": method, "query": query,
        "relevance": rel, "ild": ild, "f1": f1, "weighted_sum": ws,
        "retriever": retriever, "topN": topN, "rerank": rerank,
        "lambda": lambda_, "intent_score": intent_score,
        "temporal_score": temporal_score, "beta": beta,
        "search_ms": search_ms,
    }


# ═══════════════════════════════════════════════════════════════
#  全量评估
# ═══════════════════════════════════════════════════════════════

def run_experiment(
    bundle: DatasetBundle,
    intent_classifier: IntentClassifier,
    *,
    k: int = 10,
    n_queries: int = 0,
    seed: int = 42,
) -> list[dict]:
    """对整个数据集运行全方法评估。"""
    queries = bundle.queries
    query_embs = bundle.query_emb
    emb = bundle.passage_emb

    if n_queries > 0 and n_queries < len(queries):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(queries), size=n_queries, replace=False)
        queries = [queries[i] for i in idx]
        query_embs = query_embs[idx]

    nq = len(queries)
    print(f"\n{'='*70}")
    print(f"  Running experiment: {bundle.dataset_name} ({nq} queries, k={k})")
    print(f"{'='*70}")

    # 构建检索器
    print("Building retrievers...")
    brute, ivf, hnsw = build_all_retrievers(emb)

    all_rows = []
    t_start = time.perf_counter()

    for i in range(nq):
        rows = run_all_methods_single_query(
            queries[i], query_embs[i], emb,
            brute, ivf, hnsw, intent_classifier,
            k=k, freshness=bundle.freshness,
        )
        all_rows.extend(rows)

        if (i + 1) % 50 == 0 or i == nq - 1:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1}/{nq}] {elapsed:.1f}s elapsed")

    print(f"  Done: {len(all_rows)} total records")
    return all_rows


# ═══════════════════════════════════════════════════════════════
#  聚合 & 统计
# ═══════════════════════════════════════════════════════════════

def aggregate_results(rows: list[dict]) -> dict[str, dict]:
    """按 method 聚合指标。"""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["method"]].append(r)

    summary = {}
    for method, recs in buckets.items():
        n = len(recs)
        summary[method] = {
            "method": method,
            "n_queries": n,
            "relevance": np.mean([r["relevance"] for r in recs]),
            "ild": np.mean([r["ild"] for r in recs]),
            "f1": np.mean([r["f1"] for r in recs]),
            "weighted_sum": np.mean([r["weighted_sum"] for r in recs]),
            "search_ms": np.mean([r["search_ms"] for r in recs]),
        }
    return summary


def grouped_aggregate(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """按 intent 分组 (overall, ambiguous, specific) 聚合。"""
    groups = {
        "overall": rows,
        "ambiguous": [r for r in rows if r["intent_score"] < INTENT_THRESHOLD],
        "specific": [r for r in rows if r["intent_score"] >= INTENT_THRESHOLD],
    }
    return {g: aggregate_results(recs) for g, recs in groups.items() if recs}


def significance_tests(rows: list[dict]) -> list[dict]:
    """Wilcoxon 检验: L5 vs baselines, 相邻消融层。"""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("  scipy not installed, skipping significance tests")
        return []

    from collections import defaultdict
    by_method = defaultdict(dict)
    for r in rows:
        by_method[r["method"]][r["query"]] = r

    # 获取所有查询（保持顺序一致）
    l5_data = by_method.get("L5_full", {})
    all_queries = list(l5_data.keys())
    if not all_queries:
        return []

    # baseline 中 F1 最高的
    baseline_methods = [m for m in by_method if not m.startswith("L")]
    baseline_summary = {}
    for m in baseline_methods:
        vals = [by_method[m].get(q, {}).get("f1", 0) for q in all_queries]
        baseline_summary[m] = np.mean(vals)
    best_baseline = max(baseline_summary, key=baseline_summary.get) if baseline_summary else None

    # 配对比较列表
    comparisons = []
    # L5 vs best baseline
    if best_baseline:
        comparisons.append(("L5_full", best_baseline))
    # 相邻消融
    ablation_layers = ["L1_score_adaptive", "L2_+intent", "L3_+cluster",
                       "L4_+temporal", "L5_full"]
    for i in range(1, len(ablation_layers)):
        comparisons.append((ablation_layers[i], ablation_layers[i - 1]))

    test_results = []
    metrics = ["f1", "relevance", "ild"]
    n_tests = len(comparisons) * len(metrics)

    for method_a, method_b in comparisons:
        data_a = by_method.get(method_a, {})
        data_b = by_method.get(method_b, {})
        common = [q for q in all_queries if q in data_a and q in data_b]
        if len(common) < 10:
            continue

        for metric in metrics:
            vals_a = np.array([data_a[q][metric] for q in common])
            vals_b = np.array([data_b[q][metric] for q in common])
            diff = vals_a - vals_b

            if np.all(diff == 0):
                p = 1.0
                stat = 0.0
            else:
                try:
                    stat, p = wilcoxon(vals_a, vals_b, alternative="greater")
                except ValueError:
                    stat, p = 0.0, 1.0

            # Bonferroni 校正
            p_corrected = min(p * n_tests, 1.0)

            test_results.append({
                "method_a": method_a, "method_b": method_b,
                "metric": metric, "n_pairs": len(common),
                "mean_a": float(np.mean(vals_a)),
                "mean_b": float(np.mean(vals_b)),
                "delta": float(np.mean(vals_a) - np.mean(vals_b)),
                "statistic": float(stat),
                "p_value": float(p),
                "p_corrected": float(p_corrected),
                "significant": p_corrected < 0.05,
            })

    return test_results


# ═══════════════════════════════════════════════════════════════
#  输出: CSV + 可视化
# ═══════════════════════════════════════════════════════════════

def save_results(
    rows: list[dict],
    dataset_name: str,
    out_dir: Path,
):
    """保存 per-query CSV、聚合 CSV、统计检验、可视化。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) per_query_results.csv
    csv_path = out_dir / "per_query_results.csv"
    fields = ["method", "query", "relevance", "ild", "f1", "weighted_sum",
              "retriever", "topN", "rerank", "lambda", "intent_score",
              "temporal_score", "beta", "search_ms"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Saved: {csv_path}")

    # 2) baseline_grid.csv
    baseline_rows = [r for r in rows if not r["method"].startswith("L")]
    summary = aggregate_results(baseline_rows)
    csv_path = out_dir / "baseline_grid.csv"
    agg_fields = ["method", "n_queries", "relevance", "ild", "f1",
                  "weighted_sum", "search_ms"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        for m in sorted(summary):
            row = {k: round(summary[m][k], 4) if isinstance(summary[m][k], float)
                   else summary[m][k] for k in agg_fields}
            w.writerow(row)
    print(f"Saved: {csv_path}")

    # 3) ablation_summary.csv
    ablation_methods = ["L1_score_adaptive", "L2_+intent", "L3_+cluster",
                        "L4_+temporal", "L5_full"]
    grouped = grouped_aggregate(rows)

    csv_path = out_dir / "ablation_summary.csv"
    abl_fields = ["group", "method", "n_queries", "relevance", "ild", "f1",
                  "weighted_sum", "search_ms"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=abl_fields)
        w.writeheader()
        for group_name, summary_dict in grouped.items():
            # best baseline
            baseline_in_group = {m: v for m, v in summary_dict.items()
                                 if not m.startswith("L")}
            if baseline_in_group:
                best_bl = max(baseline_in_group, key=lambda m: baseline_in_group[m]["f1"])
                bl_data = baseline_in_group[best_bl]
                row = {"group": group_name, "method": f"best_baseline ({best_bl})"}
                row.update({k: round(bl_data[k], 4) if isinstance(bl_data[k], float)
                           else bl_data[k] for k in agg_fields})
                w.writerow(row)

            # ablation layers
            for m in ablation_methods:
                if m in summary_dict:
                    d = summary_dict[m]
                    row = {"group": group_name, "method": m}
                    row.update({k: round(d[k], 4) if isinstance(d[k], float)
                               else d[k] for k in agg_fields})
                    w.writerow(row)
    print(f"Saved: {csv_path}")

    # 4) significance_tests.csv
    tests = significance_tests(rows)
    if tests:
        csv_path = out_dir / "significance_tests.csv"
        test_fields = list(tests[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=test_fields)
            w.writeheader()
            w.writerows(tests)
        print(f"Saved: {csv_path}")

    # 5) 可视化
    _plot_all(rows, dataset_name, out_dir)


def _plot_all(rows: list[dict], dataset_name: str, out_dir: Path):
    """生成所有可视化图表。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping plots")
        return

    summary = aggregate_results(rows)

    # ── (a) Baseline 热力图 ──
    _plot_baseline_heatmap(summary, dataset_name, out_dir)

    # ── (b) 消融柱状图 ──
    _plot_ablation_bars(summary, dataset_name, out_dir)

    # ── (c) Tradeoff 散点图 ──
    _plot_tradeoff_scatter(summary, dataset_name, out_dir)

    # ── (d) 消融递增折线图 ──
    _plot_ablation_incremental(summary, dataset_name, out_dir)

    # ── (e) 分组柱状图: ambiguous vs specific ──
    grouped = grouped_aggregate(rows)
    _plot_grouped_by_type(grouped, dataset_name, out_dir)

    # ── (f) 雷达图 ──
    _plot_radar_chart(summary, dataset_name, out_dir)


def _plot_baseline_heatmap(summary: dict, dataset_name: str, out_dir: Path):
    """热力图: retriever × topN, F1 for topk / mmr0.5 / mmr0.7 三子图。
    最优格子加金色边框 + 星标。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # L5 F1 用于参考注释
    l5_f1 = summary.get("L5_full", {}).get("f1", 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f"Baseline Grid F1 — {dataset_name}  "
                 f"(L5_full F1={l5_f1:.3f} for reference)",
                 fontsize=13)

    for ax_idx, rerank in enumerate(RERANKS):
        ax = axes[ax_idx]
        matrix = np.zeros((len(RETRIEVERS), len(TOPN_VALUES)))
        for i, ret in enumerate(RETRIEVERS):
            for j, tn in enumerate(TOPN_VALUES):
                method = f"{ret}_{tn}_{rerank}"
                if method in summary:
                    matrix[i, j] = summary[method]["f1"]

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                       vmin=max(0, matrix.min() - 0.02),
                       vmax=min(1, matrix.max() + 0.02))
        ax.set_xticks(range(len(TOPN_VALUES)))
        ax.set_xticklabels(TOPN_VALUES)
        ax.set_yticks(range(len(RETRIEVERS)))
        ax.set_yticklabels(RETRIEVERS)
        ax.set_xlabel("topN")
        ax.set_ylabel("Retriever")
        ax.set_title(f"Rerank: {rerank}")

        # 找当前子图最大值位置
        best_i, best_j = np.unravel_index(matrix.argmax(), matrix.shape)

        # 数值标注
        for i in range(len(RETRIEVERS)):
            for j in range(len(TOPN_VALUES)):
                is_best = (i == best_i and j == best_j)
                label = f"{matrix[i, j]:.3f}"
                if is_best:
                    label = f"★ {label}"
                ax.text(j, i, label,
                        ha="center", va="center", fontsize=9,
                        fontweight="bold" if is_best else "normal",
                        color="white" if matrix[i, j] > (matrix.max() + matrix.min()) / 2
                        else "black")
                # 金色边框
                if is_best:
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     linewidth=3, edgecolor="gold", facecolor="none")
                    ax.add_patch(rect)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = out_dir / f"baseline_heatmap_{dataset_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def _plot_ablation_bars(summary: dict, dataset_name: str, out_dir: Path):
    """柱状图: 消融层 × 指标 (relevance, ILD, F1)。
    F1 柱上方标注层间 delta, best baseline 画灰色虚线, baseline 柱加斜线填充。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ablation = ["L1_score_adaptive", "L2_+intent", "L3_+dyn_index",
                "L4_+cluster", "L5_full"]
    labels = ["L1\nscore_adp", "L2\n+intent", "L3\n+cluster",
              "L4\n+temporal", "L5\nfull"]

    # 找 best baseline
    baseline_methods = {m: v for m, v in summary.items() if not m.startswith("L")}
    best_bl = max(baseline_methods, key=lambda m: baseline_methods[m]["f1"]) if baseline_methods else None
    best_bl_f1 = baseline_methods[best_bl]["f1"] if best_bl else 0.0

    methods = []
    method_labels = []
    if best_bl:
        methods.append(best_bl)
        short_bl = best_bl[:15] if len(best_bl) > 15 else best_bl
        method_labels.append(f"Best BL\n({short_bl})")
    methods.extend(ablation)
    method_labels.extend(labels)

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    rels = [summary[m]["relevance"] if m in summary else 0 for m in methods]
    ilds = [summary[m]["ild"] if m in summary else 0 for m in methods]
    f1s = [summary[m]["f1"] if m in summary else 0 for m in methods]

    # baseline 柱加斜线填充
    n_bl = 1 if best_bl else 0
    # Relevance bars
    bars_rel = ax.bar(x[:n_bl] - width, rels[:n_bl], width,
                      color="tab:blue", alpha=0.8, hatch="//", label="Relevance (BL)")
    ax.bar(x[n_bl:] - width, rels[n_bl:], width,
           color="tab:blue", alpha=0.8, label="Relevance")
    # ILD bars
    ax.bar(x[:n_bl], ilds[:n_bl], width,
           color="tab:orange", alpha=0.8, hatch="//", label="ILD (BL)")
    ax.bar(x[n_bl:], ilds[n_bl:], width,
           color="tab:orange", alpha=0.8, label="ILD")
    # F1 bars
    ax.bar(x[:n_bl] + width, f1s[:n_bl], width,
           color="tab:green", alpha=0.8, hatch="//", label="F1 (BL)")
    bars_f1_abl = ax.bar(x[n_bl:] + width, f1s[n_bl:], width,
                         color="tab:green", alpha=0.8, label="F1")

    # F1 柱上方标注层间 delta
    abl_f1s = [summary[m]["f1"] if m in summary else 0 for m in ablation]
    for i in range(len(ablation)):
        bar_x = x[n_bl + i] + width
        bar_y = abl_f1s[i]
        if i == 0:
            delta = abl_f1s[i] - best_bl_f1
        else:
            delta = abl_f1s[i] - abl_f1s[i - 1]
        sign = "+" if delta >= 0 else ""
        ax.text(bar_x, bar_y + 0.008, f"{sign}{delta:.3f}",
                ha="center", va="bottom", fontsize=7, color="darkgreen")

    # best baseline F1 灰色虚线
    if best_bl:
        ax.axhline(y=best_bl_f1, color="gray", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"Best BL F1={best_bl_f1:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(f"Ablation Study — {dataset_name}")
    # 简化图例：只显示核心条目
    handles, lbls = ax.get_legend_handles_labels()
    # 去重并只取需要的
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, lbls):
        short = l.split(" (")[0]  # "Relevance (BL)" -> "Relevance"
        if short not in seen:
            seen.add(short)
            unique_handles.append(h)
            unique_labels.append(short if "BL" not in l else l)
    ax.legend(unique_handles, unique_labels, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = out_dir / f"ablation_bars_{dataset_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def _plot_tradeoff_scatter(summary: dict, dataset_name: str, out_dir: Path):
    """散点图: ILD vs Relevance。
    baseline 按 retriever 着色, 按 rerank 选 marker;
    L1→L5 红色箭头连线; Pareto 前沿虚线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(11, 8))

    ablation_names = ["L1_score_adaptive", "L2_+intent", "L3_+cluster",
                      "L4_+temporal", "L5_full"]
    extra = {"brute_200_threshold0.5", "brute_200_maxmin"}

    # baseline 颜色/marker 映射
    ret_colors = {"brute": "#4e79a7", "ivf": "#f28e2b", "hnsw": "#59a14f"}
    rerank_markers = {"topk": "o", "mmr0.5": "s", "mmr0.7": "^"}

    # 收集所有点用于 Pareto 前沿
    all_points = []

    # 画 baseline 点
    for method, d in summary.items():
        if method.startswith("L") or method in extra:
            continue
        rel = d["relevance"]
        ild = d["ild"]
        all_points.append((ild, rel))

        # 解析 retriever 和 rerank
        parts = method.split("_")
        ret_name = parts[0] if parts[0] in ret_colors else "brute"
        rerank_name = parts[-1] if parts[-1] in rerank_markers else "topk"
        color = ret_colors.get(ret_name, "gray")
        marker = rerank_markers.get(rerank_name, "o")

        ax.scatter(ild, rel, s=40, alpha=0.5, color=color, marker=marker, zorder=2)

    # 画 extra baselines
    for method in extra:
        if method in summary:
            d = summary[method]
            rel, ild = d["relevance"], d["ild"]
            all_points.append((ild, rel))
            ax.scatter(ild, rel, s=80, zorder=4, marker="D",
                       edgecolors="black", linewidths=0.5, color="#e15759")
            ax.annotate(method.replace("brute_200_", ""), (ild, rel),
                        fontsize=7, ha="left",
                        xytext=(5, -8), textcoords="offset points")

    # 画 ablation 点 + 红色箭头连线
    abl_points = []
    for method in ablation_names:
        if method in summary:
            d = summary[method]
            rel, ild = d["relevance"], d["ild"]
            all_points.append((ild, rel))
            abl_points.append((ild, rel, method))
            ax.scatter(ild, rel, s=150, zorder=6, marker="*",
                       color="#e15759", edgecolors="black", linewidths=0.5)
            ax.annotate(method, (ild, rel), fontsize=8, ha="left",
                        fontweight="bold",
                        xytext=(6, 6), textcoords="offset points")

    # 红色箭头连线 L1→L2→...→L5
    for i in range(1, len(abl_points)):
        x0, y0, _ = abl_points[i - 1]
        x1, y1, _ = abl_points[i]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="red",
                                    lw=1.5, connectionstyle="arc3,rad=0.1"))

    # Pareto 前沿虚线 (maximize both ILD and Relevance)
    if all_points:
        pts = sorted(all_points, key=lambda p: p[0])  # sort by ILD
        pareto = [pts[0]]
        for p in pts[1:]:
            if p[1] >= pareto[-1][1]:
                pareto.append(p)
        if len(pareto) >= 2:
            px = [p[0] for p in pareto]
            py = [p[1] for p in pareto]
            ax.plot(px, py, "k--", alpha=0.4, linewidth=1.2, label="Pareto front")

    # 自定义图例
    legend_elements = []
    for ret, color in ret_colors.items():
        legend_elements.append(Line2D([0], [0], marker="o", color="w",
                                      markerfacecolor=color, markersize=8,
                                      label=f"Retriever: {ret}"))
    for rerank, marker in rerank_markers.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color="w",
                                      markerfacecolor="gray", markersize=8,
                                      label=f"Rerank: {rerank}"))
    legend_elements.append(Line2D([0], [0], marker="*", color="w",
                                  markerfacecolor="#e15759", markersize=12,
                                  label="Ablation L1-L5"))
    legend_elements.append(Line2D([0], [0], linestyle="--", color="black",
                                  alpha=0.4, label="Pareto front"))
    ax.legend(handles=legend_elements, fontsize=8, loc="lower left")

    ax.set_xlabel("Diversity (ILD)")
    ax.set_ylabel("Relevance")
    ax.set_title(f"Relevance vs Diversity Tradeoff — {dataset_name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"tradeoff_scatter_{dataset_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def _plot_ablation_incremental(summary: dict, dataset_name: str, out_dir: Path):
    """折线图: L1 → L5 各指标递增变化。
    Relevance 和 ILD 之间加灰色阴影带, 每步加 F1 delta 标注,
    加 best baseline 水平参考线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ablation = ["L1_score_adaptive", "L2_+intent", "L3_+dyn_index",
                "L4_+cluster", "L5_full"]
    present = [m for m in ablation if m in summary]
    if not present:
        return

    # best baseline
    baseline_methods = {m: v for m, v in summary.items() if not m.startswith("L")}
    best_bl = max(baseline_methods, key=lambda m: baseline_methods[m]["f1"]) if baseline_methods else None

    x = range(len(present))
    rels = [summary[m]["relevance"] for m in present]
    ilds = [summary[m]["ild"] for m in present]
    f1s = [summary[m]["f1"] for m in present]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, rels, "o-", label="Relevance", color="tab:blue", linewidth=2)
    ax.plot(x, ilds, "s-", label="ILD", color="tab:orange", linewidth=2)
    ax.plot(x, f1s, "^-", label="F1", color="tab:green", linewidth=2, markersize=10)

    # 灰色阴影带: Relevance 和 ILD 之间
    ax.fill_between(list(x), rels, ilds, alpha=0.08, color="gray")

    # F1 delta 标注
    for i in range(len(present)):
        if i == 0:
            if best_bl:
                delta = f1s[i] - baseline_methods[best_bl]["f1"]
            else:
                delta = 0
        else:
            delta = f1s[i] - f1s[i - 1]
        sign = "+" if delta >= 0 else ""
        ax.annotate(f"{sign}{delta:.3f}", (i, f1s[i]),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8, color="darkgreen",
                    fontweight="bold")

    # best baseline 参考线
    if best_bl:
        bl_d = baseline_methods[best_bl]
        ax.axhline(y=bl_d["relevance"], color="tab:blue", linestyle=":",
                   linewidth=1, alpha=0.5)
        ax.axhline(y=bl_d["ild"], color="tab:orange", linestyle=":",
                   linewidth=1, alpha=0.5)
        ax.axhline(y=bl_d["f1"], color="tab:green", linestyle=":",
                   linewidth=1, alpha=0.5, label=f"Best BL F1={bl_d['f1']:.3f}")

    ax.set_xticks(list(x))
    short_labels = [m.replace("L1_score_adaptive", "L1: score \u03bb")
                     .replace("L2_+intent", "L2: +intent")
                     .replace("L3_+cluster", "L3: +cluster")
                     .replace("L4_+temporal", "L4: +temporal")
                     .replace("L5_full", "L5: full")
                    for m in present]
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title(f"Incremental Ablation — {dataset_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"ablation_incremental_{dataset_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def _plot_grouped_by_type(grouped: dict, dataset_name: str, out_dir: Path):
    """分组柱状图: ambiguous vs specific 的 F1 对比。
    methods = best_baseline + L1-L5。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ambig = grouped.get("ambiguous", {})
    specif = grouped.get("specific", {})
    if not ambig and not specif:
        return

    ablation = ["L1_score_adaptive", "L2_+intent", "L3_+dyn_index",
                "L4_+cluster", "L5_full"]

    # 找 best baseline (from overall)
    overall = grouped.get("overall", {})
    baseline_methods = {m: v for m, v in overall.items() if not m.startswith("L")}
    best_bl = max(baseline_methods, key=lambda m: baseline_methods[m]["f1"]) if baseline_methods else None

    methods = []
    labels = []
    if best_bl:
        methods.append(best_bl)
        short = best_bl[:12] if len(best_bl) > 12 else best_bl
        labels.append(f"BestBL\n({short})")
    for m in ablation:
        methods.append(m)
        labels.append(m.replace("L1_score_adaptive", "L1")
                       .replace("L2_+intent", "L2")
                       .replace("L3_+cluster", "L3")
                       .replace("L4_+temporal", "L4")
                       .replace("L5_full", "L5"))

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ambig_f1 = [ambig.get(m, {}).get("f1", 0) for m in methods]
    specif_f1 = [specif.get(m, {}).get("f1", 0) for m in methods]

    bars1 = ax.bar(x - width / 2, ambig_f1, width,
                   label="Ambiguous", color="#e15759", alpha=0.85)
    bars2 = ax.bar(x + width / 2, specif_f1, width,
                   label="Specific", color="#4e79a7", alpha=0.85)

    # 数值标注
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"F1 by Query Type — {dataset_name}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = out_dir / f"grouped_by_type_{dataset_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def _plot_radar_chart(summary: dict, dataset_name: str, out_dir: Path):
    """雷达图: 四轴 (Relevance, ILD, F1, Speed)，
    比较 best_baseline / L3 / L5_full。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 选要比较的方法
    baseline_methods = {m: v for m, v in summary.items() if not m.startswith("L")}
    if not baseline_methods:
        return
    best_bl = max(baseline_methods, key=lambda m: baseline_methods[m]["f1"])

    compare_methods = [best_bl, "L3_+cluster", "L5_full"]
    compare_labels = [f"Best BL\n({best_bl[:12]})", "L3: +cluster", "L5: full"]
    compare_colors = ["#4e79a7", "#f28e2b", "#e15759"]

    # 确保方法都存在
    compare_methods = [m for m in compare_methods if m in summary]
    if len(compare_methods) < 2:
        return

    axes_labels = ["Relevance", "ILD", "F1", "Speed"]
    n_axes = len(axes_labels)

    # 归一化 speed: 取倒数并归一化到 [0, 1]
    # 越快 speed 越高
    max_ms = max(summary[m]["search_ms"] for m in compare_methods) + 1e-6

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    for mi, method in enumerate(compare_methods):
        d = summary[method]
        speed = 1.0 - (d["search_ms"] / max_ms)  # invert: faster = higher
        speed = max(0.0, speed)
        values = [d["relevance"], d["ild"], d["f1"], speed]
        values += values[:1]  # close

        label = compare_labels[mi] if mi < len(compare_labels) else method
        ax.plot(angles, values, "o-", linewidth=2, label=label,
                color=compare_colors[mi % len(compare_colors)])
        ax.fill(angles, values, alpha=0.1,
                color=compare_colors[mi % len(compare_colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f"Multi-dimensional Comparison — {dataset_name}",
                 fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    path = out_dir / f"radar_chart_{dataset_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════
#  打印摘要
# ═══════════════════════════════════════════════════════════════

def print_summary(rows: list[dict], dataset_name: str):
    """打印结果摘要到控制台。"""
    grouped = grouped_aggregate(rows)

    for group_name, summary_dict in grouped.items():
        print(f"\n{'='*80}")
        print(f"  {dataset_name} — {group_name}")
        print(f"{'='*80}")

        # 分 baseline 和 ablation
        bl_methods = sorted([m for m in summary_dict if not m.startswith("L")])
        abl_methods = [m for m in ["L1_score_adaptive", "L2_+intent", "L3_+cluster",
                                   "L4_+temporal", "L5_full"]
                       if m in summary_dict]

        header = f"  {'Method':<28} {'N':>5} {'Rel':>8} {'ILD':>8} {'F1':>8} {'WS':>8} {'ms':>8}"
        print(header)
        print(f"  {'-'*75}")

        # Top-5 baselines by F1
        bl_sorted = sorted(bl_methods, key=lambda m: summary_dict[m]["f1"], reverse=True)
        for m in bl_sorted[:5]:
            d = summary_dict[m]
            print(f"  {m:<28} {d['n_queries']:>5} {d['relevance']:>8.4f} "
                  f"{d['ild']:>8.4f} {d['f1']:>8.4f} {d['weighted_sum']:>8.4f} "
                  f"{d['search_ms']:>8.1f}")

        if bl_sorted and abl_methods:
            print(f"  {'...':<28} {'':>5} {'':>8} {'':>8} {'':>8}")

        for m in abl_methods:
            d = summary_dict[m]
            marker = " ***" if m == "L5_full" else ""
            print(f"  {m:<28} {d['n_queries']:>5} {d['relevance']:>8.4f} "
                  f"{d['ild']:>8.4f} {d['f1']:>8.4f} {d['weighted_sum']:>8.4f} "
                  f"{d['search_ms']:>8.1f}{marker}")


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Quantitative ablation: baseline grid + L1-L5 layers",
    )
    parser.add_argument("--msmarco-dir", default=None,
                        help="MS MARCO 数据目录 (None=跳过)")
    parser.add_argument("--improved-dir", default=None,
                        help="Improved corpus 数据目录 (None=跳过)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=0,
                        help="每个数据集使用的查询数 (0=全部)")
    parser.add_argument("--out-dir", default="outputs/quantitative")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-queries", action="store_true",
                        help="注入歧义查询，使查询集包含 ambiguous + specific")
    parser.add_argument("--n-ambiguous", type=int, default=15,
                        help="歧义查询数 (仅 --mixed-queries 时生效)")
    parser.add_argument("--n-specific", type=int, default=15,
                        help="明确查询数 (仅 --mixed-queries 时生效)")
    args = parser.parse_args()

    if args.msmarco_dir is None and args.improved_dir is None:
        parser.error("至少指定 --msmarco-dir 或 --improved-dir")

    # 加载意图分类器
    intent_classifier = IntentClassifier()
    for model_dir in ["models/intent_v3", "models/intent_v2", "models/intent_chatgpt"]:
        model_path = Path(model_dir) / "intent_model.pkl"
        if model_path.exists():
            intent_classifier = IntentClassifier.load(model_dir)
            print(f"Intent classifier loaded from {model_dir}")
            break
    else:
        print("No intent model found, using default classifier")

    # ── MS MARCO ──
    if args.msmarco_dir and Path(args.msmarco_dir).exists():
        bundle = load_msmarco(args.msmarco_dir)
        if args.mixed_queries:
            # 分层采样: n_ambiguous + n_specific, 不再需要 n_queries 随机采样
            bundle = mix_ambiguous_queries(bundle, n_ambiguous=args.n_ambiguous,
                                           n_specific=args.n_specific, seed=args.seed)
            n_q = 0  # 用全部 (已预采样好)
        else:
            n_q = args.n_queries
        rows = run_experiment(bundle, intent_classifier,
                              k=args.k, n_queries=n_q, seed=args.seed)
        out = Path(args.out_dir) / "msmarco"
        print_summary(rows, "msmarco")
        save_results(rows, "msmarco", out)

    # ── Improved corpus ──
    if args.improved_dir and Path(args.improved_dir).exists():
        bundle = load_improved(args.improved_dir, seed=args.seed)
        rows = run_experiment(bundle, intent_classifier,
                              k=args.k, n_queries=args.n_queries, seed=args.seed)
        out = Path(args.out_dir) / "improved"
        print_summary(rows, "improved")
        save_results(rows, "improved", out)

    print("\nAll experiments complete!")


if __name__ == "__main__":
    main()
