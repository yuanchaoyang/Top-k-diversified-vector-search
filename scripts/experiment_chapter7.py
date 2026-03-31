#!/usr/bin/env python3
"""
Chapter 7 — 统一实验脚本
========================

产出论文第 7 章所有表格和定性案例所需的数据文件:

  1. overall_results.csv        — Table 7.1  总体端到端结果
  2. query_type_split.csv       — Table 7.2  按查询类型拆分
  3. ablation_results.csv       — Table 7.3  分层消融
  4. significance_tests.csv     — Table 7.4  配对显著性检验
  5. ann_sweep.csv              — Section 7.5 索引层检索实验
  6. intent_model_eval.csv      — Table 7.5  意图模型评估
  7. case_study_outputs.json    — Section 7.7 定性案例

消融层顺序 (与论文一致):
  L1: score-based adaptive-λ (brute force + hybrid score λ)
  L2: + learned intent model  (brute force + intent-aware λ)
  L3: + dynamic IVF/HNSW backend selection
  L4: + cluster-aware reranking penalty
  L5: + temporal signals / freshness-aware reranking

使用方法:
    source .venv/bin/activate
    PYTHONPATH=src python scripts/experiment_chapter7.py \
      --improved-dir data/improved --k 10 --out-dir outputs/chapter7

    # 也可指定 MS MARCO:
    PYTHONPATH=src python scripts/experiment_chapter7.py \
      --msmarco-dir data/msmarco --improved-dir data/improved --k 10

    # 单独跑某部分 (方便调试):
    PYTHONPATH=src python scripts/experiment_chapter7.py \
      --improved-dir data/improved --only ann_sweep
    PYTHONPATH=src python scripts/experiment_chapter7.py \
      --only intent_model
"""
from __future__ import annotations

import argparse
import csv
import json
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
    AMBIGUOUS_WORDS, encode_queries, _extract_title,
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

LAMBDA_MIN = 0.30
LAMBDA_MAX = 0.90
INTENT_THRESHOLD = 0.4


# ═══════════════════════════════════════════════════════════════
#  Part 1-4: 端到端 + 消融 + 查询类型 + 显著性
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
    """对单条查询运行所有方法 (2 baselines + L1-L5)，返回 per-query 行。

    消融层顺序与论文一致:
      L1  score-based adaptive-λ        (brute force, hybrid score λ)
      L2  + intent model                (brute force, intent λ)
      L3  + dynamic backend selection   (IVF/HNSW, intent λ, no cluster/temporal)
      L4  + cluster-aware penalty       (IVF/HNSW, intent λ, cluster penalty)
      L5  + temporal / freshness        (IVF/HNSW, intent λ, cluster, temporal)
    """
    rows: list[dict] = []
    q = query_vec

    # ── 公共: brute force 候选 ──
    brute_res = brute.search(q.reshape(1, -1), 200)
    cands_bf = brute_res.indices[0]
    scores_bf = brute_res.scores[0]

    # ── 信号分析 ──
    ta = query_aware_lambda_and_beta(
        query, intent_classifier,
        temporal_config=TemporalConfig(),
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
    )
    intent_score = ta["intent_score"]
    intent_lam = ta["lambda"]
    temporal_score = ta["temporal_score"]
    beta = ta["beta"]

    score_cfg = AdaptiveLambdaConfig(
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX, strategy="hybrid",
    )
    score_lam = adaptive_lambda_from_scores(scores_bf, score_cfg)

    cluster_ids = getattr(ivf, "assignments", None)

    def _row(method, selected, *, retriever="brute", topN=200,
             lambda_=0.0, extra=None):
        rel, ild, f1, ws = compute_metrics(q, selected, emb)
        r = {
            "method": method, "query": query,
            "MeanRel": rel, "ILD": ild, "F1": f1,
            "retriever": retriever, "topN": topN,
            "lambda": lambda_, "intent_score": intent_score,
            "temporal_score": temporal_score, "beta": beta,
        }
        if extra:
            r.update(extra)
        return r

    # ── Baseline 1: relevance-only (top-k, no rerank) ──
    selected = cands_bf[:k]
    rows.append(_row("Relevance-only", selected))

    # ── Baseline 2: fixed-MMR (λ=0.7) ──
    selected = mmr_rerank_incremental(q, cands_bf, emb, k=k, lambda_=0.7)
    rows.append(_row("Fixed-MMR", selected, lambda_=0.7))

    # ── L1: score-based adaptive-λ (brute force) ──
    selected = mmr_rerank_incremental(q, cands_bf, emb, k=k, lambda_=score_lam)
    rows.append(_row("L1", selected, lambda_=score_lam))

    # ── L2: + intent model (brute force, intent λ) ──
    selected = mmr_rerank_incremental(q, cands_bf, emb, k=k, lambda_=intent_lam)
    rows.append(_row("L2", selected, lambda_=intent_lam))

    # ── L3: + dynamic backend (IVF/HNSW, intent λ, no cluster/temporal) ──
    adaptive_topN = compute_adaptive_topN(intent_score, temporal_score,
                                          topN_min=100, topN_max=300)
    if intent_score < INTENT_THRESHOLD:
        nprobe = compute_adaptive_nprobe(intent_score, temporal_score)
        ivf.set_nprobe(nprobe)
        res_dyn = ivf.search(q.reshape(1, -1), adaptive_topN)
        l3_retriever = "ivf"
        l3_extra = {"nprobe": nprobe}
    else:
        ef = compute_adaptive_ef(intent_score, temporal_score)
        hnsw.set_ef_search(ef)
        res_dyn = hnsw.search(q.reshape(1, -1), adaptive_topN)
        l3_retriever = "hnsw"
        l3_extra = {"ef_search": ef}

    cands_dyn = res_dyn.indices[0]

    selected = mmr_rerank_incremental(q, cands_dyn, emb, k=k, lambda_=intent_lam)
    rows.append(_row("L3", selected, retriever=l3_retriever,
                      topN=adaptive_topN, lambda_=intent_lam, extra=l3_extra))

    # ── L4: + cluster-aware penalty (IVF/HNSW, intent λ, cluster γ) ──
    use_cluster = intent_score < INTENT_THRESHOLD
    gamma = 0.15 if use_cluster else 0.0
    c_ids = cluster_ids if use_cluster else None

    selected = mmr_rerank_incremental(
        q, cands_dyn, emb, k=k, lambda_=intent_lam,
        cluster_ids=c_ids, cluster_penalty=gamma,
    )
    rows.append(_row("L4", selected, retriever=l3_retriever,
                      topN=adaptive_topN, lambda_=intent_lam,
                      extra={**l3_extra, "gamma": gamma}))

    # ── L5: + temporal / freshness (full pipeline) ──
    # quality boost
    top_score = float(res_dyn.scores[0][0]) if len(res_dyn.scores[0]) > 0 else 0.0
    l5_lam = float(intent_lam)
    if top_score < 0.5:
        quality_boost = (0.5 - top_score) * 1.0
        l5_lam = min(0.9, l5_lam + quality_boost)

    min_ratio = 0.60 + 0.25 * intent_score

    selected = mmr_rerank_temporal(
        q, cands_dyn, emb, k=k, lambda_=l5_lam,
        freshness=freshness, beta=float(beta),
        min_score_ratio=min_ratio,
        cluster_ids=c_ids, cluster_penalty=gamma,
    )
    # 定量实验: 不做 filter_low_quality, 保持固定 k 以确保跨方法可比性。
    # filter_low_quality 仅在 case study 中展示（Section 7.7）。

    rows.append(_row("L5", selected, retriever=l3_retriever,
                      topN=adaptive_topN, lambda_=l5_lam,
                      extra={**l3_extra, "gamma": gamma,
                             "beta": float(beta), "min_score_ratio": min_ratio}))

    return rows


def run_end_to_end(
    bundle: DatasetBundle,
    intent_classifier: IntentClassifier,
    *,
    k: int = 10,
    seed: int = 42,
) -> list[dict]:
    """运行全部查询, 返回 per-query 行列表。"""
    emb = bundle.passage_emb
    queries = bundle.queries
    query_embs = bundle.query_emb

    print(f"\n{'='*70}")
    print(f"  End-to-end: {bundle.dataset_name} ({len(queries)} queries, k={k})")
    print(f"{'='*70}")

    print("Building retrievers...")
    brute, ivf, hnsw = build_all_retrievers(emb)

    all_rows: list[dict] = []
    t0 = time.perf_counter()

    for i in range(len(queries)):
        rows = run_all_methods_single_query(
            queries[i], query_embs[i], emb,
            brute, ivf, hnsw, intent_classifier,
            k=k, freshness=bundle.freshness,
        )
        all_rows.extend(rows)
        if (i + 1) % 25 == 0 or i == len(queries) - 1:
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{len(queries)}] {elapsed:.1f}s")

    return all_rows


def save_overall_results(rows: list[dict], out_dir: Path):
    """Table 7.1 — overall_results.csv"""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["method"]].append(r)

    method_order = ["Relevance-only", "Fixed-MMR", "L1", "L2", "L3", "L4", "L5"]
    out = out_dir / "overall_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Method", "MeanRel", "ILD", "Avg_F1"])
        w.writeheader()
        for m in method_order:
            recs = buckets.get(m, [])
            if not recs:
                continue
            w.writerow({
                "Method": m,
                "MeanRel": round(np.mean([r["MeanRel"] for r in recs]), 4),
                "ILD": round(np.mean([r["ILD"] for r in recs]), 4),
                "Avg_F1": round(np.mean([r["F1"] for r in recs]), 4),
            })
    print(f"Saved: {out}")


def save_query_type_split(rows: list[dict], out_dir: Path):
    """Table 7.2 — query_type_split.csv"""
    ambig = [r for r in rows if r["intent_score"] < INTENT_THRESHOLD]
    specif = [r for r in rows if r["intent_score"] >= INTENT_THRESHOLD]

    from collections import defaultdict

    def _best_non_adaptive_baseline(subset):
        """在 Relevance-only, Fixed-MMR, L1 中选 Avg. F1 最高的。"""
        buckets = defaultdict(list)
        for r in subset:
            if r["method"] in ("Relevance-only", "Fixed-MMR", "L1"):
                buckets[r["method"]].append(r)
        if not buckets:
            return None, []
        best_m = max(buckets, key=lambda m: np.mean([r["F1"] for r in buckets[m]]))
        return best_m, buckets[best_m]

    out = out_dir / "query_type_split.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "QueryType", "Method", "MeanRel", "ILD", "Avg_F1"])
        w.writeheader()

        for qtype, subset in [("Ambiguous", ambig), ("Specific", specif)]:
            if not subset:
                continue
            # best non-adaptive baseline (Relevance-only / Fixed-MMR / L1)
            best_name, best_recs = _best_non_adaptive_baseline(subset)
            if best_recs:
                w.writerow({
                    "QueryType": qtype,
                    "Method": f"Best baseline ({best_name})",
                    "MeanRel": round(np.mean([r["MeanRel"] for r in best_recs]), 4),
                    "ILD": round(np.mean([r["ILD"] for r in best_recs]), 4),
                    "Avg_F1": round(np.mean([r["F1"] for r in best_recs]), 4),
                })
            # L5
            l5_recs = [r for r in subset if r["method"] == "L5"]
            if l5_recs:
                w.writerow({
                    "QueryType": qtype,
                    "Method": "L5 full system",
                    "MeanRel": round(np.mean([r["MeanRel"] for r in l5_recs]), 4),
                    "ILD": round(np.mean([r["ILD"] for r in l5_recs]), 4),
                    "Avg_F1": round(np.mean([r["F1"] for r in l5_recs]), 4),
                })
    print(f"Saved: {out}")


def save_ablation_results(rows: list[dict], out_dir: Path):
    """Table 7.3 — ablation_results.csv"""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["method"]].append(r)

    layers = ["L1", "L2", "L3", "L4", "L5"]
    out = out_dir / "ablation_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Variant", "MeanRel", "ILD", "Avg_F1"])
        w.writeheader()
        labels = {
            "L1": "L1: score-adaptive baseline",
            "L2": "L2: + intent model",
            "L3": "L3: + dynamic backend selection",
            "L4": "L4: + cluster penalty",
            "L5": "L5: + temporal signals",
        }
        for m in layers:
            recs = buckets.get(m, [])
            if not recs:
                continue
            w.writerow({
                "Variant": labels[m],
                "MeanRel": round(np.mean([r["MeanRel"] for r in recs]), 4),
                "ILD": round(np.mean([r["ILD"] for r in recs]), 4),
                "Avg_F1": round(np.mean([r["F1"] for r in recs]), 4),
            })
    print(f"Saved: {out}")


def save_significance_tests(rows: list[dict], out_dir: Path):
    """Table 7.4 — significance_tests.csv  (Wilcoxon + Bonferroni)"""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("  scipy not installed, skipping significance tests")
        return

    from collections import defaultdict
    by_method: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_method[r["method"]][r["query"]] = r

    # 定义配对比较 (与论文 Table 7.4 一致)
    comparisons = [
        ("L2", "L1"),
        ("L4", "L3"),
        ("L5", "L4"),
    ]
    metrics = ["MeanRel", "ILD", "F1"]
    n_tests = len(comparisons) * len(metrics)  # for Bonferroni

    test_rows: list[dict] = []

    for method_a, method_b in comparisons:
        data_a = by_method.get(method_a, {})
        data_b = by_method.get(method_b, {})
        common = [q for q in data_a if q in data_b]
        if len(common) < 10:
            continue

        for metric in metrics:
            key = metric if metric != "F1" else "F1"
            vals_a = np.array([data_a[q][key] for q in common])
            vals_b = np.array([data_b[q][key] for q in common])
            diff = vals_a - vals_b

            if np.all(diff == 0):
                p = 1.0
                stat = 0.0
            else:
                try:
                    stat, p = wilcoxon(vals_a, vals_b, alternative="two-sided")
                except ValueError:
                    stat, p = 0.0, 1.0

            p_corrected = min(p * n_tests, 1.0)

            test_rows.append({
                "Comparison": f"{method_a} vs. {method_b}",
                "Metric": metric.replace("F1", "Avg. F1")
                               .replace("MeanRel", "MeanRel")
                               .replace("ILD", "ILD"),
                "n_pairs": len(common),
                "mean_A": round(float(np.mean(vals_a)), 4),
                "mean_B": round(float(np.mean(vals_b)), 4),
                "delta": round(float(np.mean(diff)), 4),
                "p_value_raw": round(float(p), 6),
                "p_value_corrected": round(float(p_corrected), 6),
                "significant_0.05": p_corrected < 0.05,
            })

    out = out_dir / "significance_tests.csv"
    if test_rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
            w.writeheader()
            w.writerows(test_rows)
        print(f"Saved: {out}")
    else:
        print("  No significance tests could be computed (too few queries?)")


def save_per_query_raw(rows: list[dict], out_dir: Path):
    """保存原始 per-query 结果, 供复查和自定义分析。"""
    out = out_dir / "per_query_raw.csv"
    fields = ["method", "query", "MeanRel", "ILD", "F1",
              "retriever", "topN", "lambda", "intent_score",
              "temporal_score", "beta"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = {k: round(r[k], 4) if isinstance(r.get(k), float) else r.get(k, "")
                   for k in fields}
            w.writerow(row)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════
#  Part 5: ANN sweep (Section 7.5)
# ═══════════════════════════════════════════════════════════════

def run_ann_sweep(data_dir: str, out_dir: Path, *, n_queries: int = 100,
                  topN: int = 10, seed: int = 42):
    """Section 7.5 — ann_sweep.csv

    IVF  nprobe  = 8, 16, 24, 32, 48
    HNSW ef      = 32, 50, 100, 150, 200

    注意: topN 默认设为 10 (即 Recall@10), 因为 HNSW search 内部会做
    ef = max(ef_search, topN), 如果 topN=200 则所有 ef<200 的配置都会
    被抬到 200, 导致 sweep 无效。topN=10 能让全部 ef_search 值生效。
    """
    from diverse_search.index import NumpyBruteForceRetriever
    from diverse_search.ivf import NumpyIVFRetriever
    from diverse_search.hnsw import NumpyHNSWRetriever

    emb_path = Path(data_dir) / "passage_embeddings.npy"
    if not emb_path.exists():
        print(f"  ANN sweep: {emb_path} not found, skipping")
        return

    print(f"\n{'='*70}")
    print(f"  ANN Sweep (Section 7.5)")
    print(f"{'='*70}")

    xb = np.load(emb_path).astype(np.float32)
    norms = np.linalg.norm(xb, axis=1, keepdims=True)
    xb = xb / (norms + 1e-8)
    n, d = xb.shape

    rng = np.random.default_rng(seed)
    nq = min(n_queries, n)
    q_idx = rng.choice(n, size=nq, replace=False)
    queries = xb[q_idx]

    print(f"  {n} vectors, dim={d}, {nq} queries, topN={topN}")

    # Ground truth
    brute = NumpyBruteForceRetriever(xb)
    brute_res = brute.search(queries, topN)
    gt_sets = [set(brute_res.indices[i].tolist()) for i in range(nq)]

    def recall(res):
        rs = []
        for i in range(nq):
            found = set(res.indices[i].tolist())
            found.discard(-1)
            rs.append(len(gt_sets[i] & found) / topN)
        return float(np.mean(rs))

    # IVF
    print("  Building IVF...")
    ivf = NumpyIVFRetriever(xb, nlist=128, nprobe=1)

    nprobe_values = [8, 16, 24, 32, 48]
    ef_values = [32, 50, 100, 150, 200]

    sweep_rows: list[dict] = []

    print(f"  IVF nprobe sweep: {nprobe_values}")
    for nprobe in nprobe_values:
        ivf.set_nprobe(nprobe)
        t0 = time.perf_counter()
        res = ivf.search(queries, topN)
        ms = (time.perf_counter() - t0) * 1000 / nq
        rec = recall(res)
        sweep_rows.append({
            "Backend": "IVF",
            "Parameter": f"nprobe={nprobe}",
            "Recall_at_k": round(rec, 4),
            "Latency_ms": round(ms, 3),
        })
        print(f"    nprobe={nprobe:>3}  recall={rec:.4f}  {ms:.3f} ms/q")

    # HNSW
    print("  Building HNSW...")
    hnsw = NumpyHNSWRetriever(xb, M=16, ef_construction=200, ef_search=32)

    print(f"  HNSW ef_search sweep: {ef_values}")
    for ef in ef_values:
        hnsw.set_ef_search(ef)
        t0 = time.perf_counter()
        res = hnsw.search(queries, topN)
        ms = (time.perf_counter() - t0) * 1000 / nq
        rec = recall(res)
        sweep_rows.append({
            "Backend": "HNSW",
            "Parameter": f"ef_search={ef}",
            "Recall_at_k": round(rec, 4),
            "Latency_ms": round(ms, 3),
        })
        print(f"    ef={ef:>5}  recall={rec:.4f}  {ms:.3f} ms/q")

    out = out_dir / "ann_sweep.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Backend", "Parameter", "Recall_at_k", "Latency_ms"])
        w.writeheader()
        w.writerows(sweep_rows)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════
#  Part 6: Intent model evaluation (Section 7.6)
# ═══════════════════════════════════════════════════════════════

def run_intent_model_eval(out_dir: Path, seed: int = 42):
    """Table 7.5 — intent_model_eval.csv

    比较 MLP / RandomForest / GradientBoosting 的 R², MAE, RMSE。
    """
    print(f"\n{'='*70}")
    print(f"  Intent Model Evaluation (Section 7.6)")
    print(f"{'='*70}")

    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"  Missing dependency: {e}. Skipping intent model eval.")
        return

    import re as _re

    # 加载标注数据
    label_files = [
        "data/chatgpt_labels.txt",
        "data/chatgpt_sentence_labels.txt",
        "data/chatgpt_diverse_labels.txt",
    ]
    all_labels: dict[str, float] = {}
    for fp in label_files:
        p = Path(fp)
        if not p.exists():
            print(f"  {fp}: not found, skipping")
            continue
        with open(p) as f:
            content = f.read()
        json_objects = _re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, _re.DOTALL)
        for js in json_objects:
            try:
                data = json.loads(js)
                for batch_name, batch_data in data.items():
                    if isinstance(batch_data, list):
                        for item in batch_data:
                            text = item["word"].lower().strip()
                            score = float(item["score"])
                            all_labels[text] = score
            except (json.JSONDecodeError, KeyError):
                continue
        print(f"  Loaded {fp}")

    if len(all_labels) < 50:
        print(f"  Too few labels ({len(all_labels)}), skipping intent eval")
        return

    texts = list(all_labels.keys())
    scores = np.array([all_labels[t] for t in texts])
    print(f"  {len(texts)} labelled samples, encoding with SentenceTransformer...")

    model_st = SentenceTransformer("all-MiniLM-L6-v2")
    X = model_st.encode(texts, show_progress_bar=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, scores, test_size=0.2, random_state=seed,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "MLP regressor": MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=1500, random_state=seed, early_stopping=True,
        ),
        "RandomForest regressor": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=seed,
        ),
        "GradientBoosting regressor": GradientBoostingRegressor(
            n_estimators=200, max_depth=6, random_state=seed,
        ),
    }

    eval_rows: list[dict] = []
    print(f"\n  {'Model':<30} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*56}")

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        eval_rows.append({
            "Model": name,
            "Test_R2": round(r2, 4),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
        })
        print(f"  {name:<30} {r2:>8.4f} {mae:>8.4f} {rmse:>8.4f}")

    out = out_dir / "intent_model_eval.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Model", "Test_R2", "MAE", "RMSE"])
        w.writeheader()
        w.writerows(eval_rows)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════
#  Part 7: Qualitative case studies (Section 7.7)
# ═══════════════════════════════════════════════════════════════

CASE_STUDY_QUERIES = [
    "spring",
    "mercury",
    "Fall of Berlin – 1945",
    "National Football League (Ireland)",
]


def run_case_studies(
    bundle: DatasetBundle,
    intent_classifier: IntentClassifier,
    *,
    k: int = 10,
    out_dir: Path,
):
    """Section 7.7 — case_study_outputs.json"""
    print(f"\n{'='*70}")
    print(f"  Qualitative Case Studies (Section 7.7)")
    print(f"{'='*70}")

    emb = bundle.passage_emb
    print("Building retrievers...")
    brute, ivf, hnsw = build_all_retrievers(emb)
    cluster_ids = getattr(ivf, "assignments", None)

    # Encode the 4 case study queries
    print(f"  Encoding {len(CASE_STUDY_QUERIES)} case study queries...")
    query_vecs = encode_queries(CASE_STUDY_QUERIES)

    cases: list[dict] = []

    for qi, (query, q_vec) in enumerate(zip(CASE_STUDY_QUERIES, query_vecs)):
        print(f"\n  Query {qi+1}: \"{query}\"")
        q = q_vec

        # Signal analysis
        ta = query_aware_lambda_and_beta(
            query, intent_classifier,
            temporal_config=TemporalConfig(),
            lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
        )
        intent_score = ta["intent_score"]
        intent_lam = ta["lambda"]
        temporal_score = ta["temporal_score"]
        beta = ta["beta"]

        # Dynamic backend selection
        adaptive_topN = compute_adaptive_topN(intent_score, temporal_score,
                                              topN_min=100, topN_max=300)
        if intent_score < INTENT_THRESHOLD:
            nprobe = compute_adaptive_nprobe(intent_score, temporal_score)
            ivf.set_nprobe(nprobe)
            res_dyn = ivf.search(q.reshape(1, -1), adaptive_topN, cluster_balanced=True)
            backend = "IVF"
            search_param = {"nprobe": int(nprobe)}
        else:
            ef = compute_adaptive_ef(intent_score, temporal_score)
            hnsw.set_ef_search(ef)
            res_dyn = hnsw.search(q.reshape(1, -1), adaptive_topN)
            backend = "HNSW"
            search_param = {"ef_search": int(ef)}

        cands_dyn = res_dyn.indices[0]
        use_cluster = intent_score < INTENT_THRESHOLD
        gamma = 0.15 if use_cluster else 0.0
        c_ids = cluster_ids if use_cluster else None

        # quality boost
        top_score = float(res_dyn.scores[0][0]) if len(res_dyn.scores[0]) > 0 else 0.0
        l5_lam = float(intent_lam)
        if top_score < 0.5:
            l5_lam = min(0.9, l5_lam + (0.5 - top_score))

        # brute force candidates for baselines
        brute_res = brute.search(q.reshape(1, -1), 200)
        cands_bf = brute_res.indices[0]

        def _format_results(selected):
            sims = emb[selected] @ q
            results_list = []
            for rank, idx in enumerate(selected):
                results_list.append({
                    "rank": rank + 1,
                    "passage_idx": int(idx),
                    "score": round(float(sims[rank]), 4),
                    "title": _extract_title(bundle.passages[idx]),
                    "snippet": bundle.passages[idx][:200],
                })
            rel, ild, f1, _ = compute_metrics(q, selected, emb)
            return results_list, round(rel, 4), round(ild, 4), round(f1, 4)

        # 1) Relevance-only
        sel_base = cands_bf[:k]
        res_base, rel_b, ild_b, f1_b = _format_results(sel_base)

        # 2) Fixed-MMR
        sel_mmr = mmr_rerank_incremental(q, cands_bf, emb, k=k, lambda_=0.7)
        res_mmr, rel_m, ild_m, f1_m = _format_results(sel_mmr)

        # 3) Full pipeline (case study 保留 filter_low_quality 以展示质量保护行为;
        #    定量实验中不使用该 filter, 见 run_all_methods_single_query)
        min_ratio = 0.60 + 0.25 * intent_score
        sel_full = mmr_rerank_temporal(
            q, cands_dyn, emb, k=k, lambda_=l5_lam,
            freshness=bundle.freshness, beta=float(beta),
            min_score_ratio=min_ratio,
            cluster_ids=c_ids, cluster_penalty=gamma,
        )
        sel_full = filter_low_quality(sel_full, q, emb, score_floor=0.30, min_keep=2)
        n_returned = len(sel_full)
        res_full, rel_f, ild_f, f1_f = _format_results(sel_full)

        case = {
            "query": query,
            "signals": {
                "intent_score": round(intent_score, 4),
                "temporal_score": round(temporal_score, 4),
                "lambda": round(l5_lam, 4),
                "beta": round(float(beta), 4),
                "selected_backend": backend,
                **search_param,
                "adaptive_topN": int(adaptive_topN),
                "cluster_penalty_gamma": round(gamma, 4),
            },
            "relevance_only": {
                "results": res_base,
                "MeanRel": rel_b, "ILD": ild_b, "F1": f1_b,
            },
            "fixed_mmr": {
                "results": res_mmr,
                "MeanRel": rel_m, "ILD": ild_m, "F1": f1_m,
            },
            "full_pipeline": {
                "results": res_full,
                "n_returned": n_returned,
                "MeanRel": rel_f, "ILD": ild_f, "F1": f1_f,
            },
        }
        cases.append(case)

        print(f"    intent={intent_score:.2f}  temporal={temporal_score:.2f}  "
              f"λ={l5_lam:.2f}  β={beta:.2f}  backend={backend}")
        print(f"    Rel-only:  Rel={rel_b}  ILD={ild_b}  F1={f1_b}")
        print(f"    Fixed-MMR: Rel={rel_m}  ILD={ild_m}  F1={f1_m}")
        print(f"    Full:      Rel={rel_f}  ILD={ild_f}  F1={f1_f}")

    out = out_dir / "case_study_outputs.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Chapter 7: unified experiment runner",
    )
    parser.add_argument("--msmarco-dir", default=None)
    parser.add_argument("--improved-dir", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out-dir", default="outputs/chapter7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default=None,
                        choices=["end_to_end", "ann_sweep", "intent_model",
                                 "case_study"],
                        help="只跑指定部分 (调试用)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Intent classifier (共享) ──
    intent_classifier = IntentClassifier()
    for model_dir in ["models/intent_v3", "models/intent_v2", "models/intent_chatgpt"]:
        model_path = Path(model_dir) / "intent_model.pkl"
        if model_path.exists():
            intent_classifier = IntentClassifier.load(model_dir)
            print(f"Intent classifier loaded from {model_dir}")
            break
    else:
        print("No intent model found, using default classifier")

    run_all = args.only is None

    # ── Part 6: Intent model eval (独立, 不依赖数据集) ──
    if run_all or args.only == "intent_model":
        run_intent_model_eval(out_dir, seed=args.seed)

    # 确定数据集
    data_dir = args.improved_dir or args.msmarco_dir
    if data_dir is None and args.only not in ("intent_model",):
        parser.error("需指定 --improved-dir 或 --msmarco-dir")

    bundle = None
    if data_dir:
        if args.improved_dir and Path(args.improved_dir).exists():
            bundle = load_improved(args.improved_dir, seed=args.seed)
        elif args.msmarco_dir and Path(args.msmarco_dir).exists():
            bundle = load_msmarco(args.msmarco_dir)

    # ── Part 1-4: 端到端 + 消融 + 查询拆分 + 显著性 ──
    if bundle and (run_all or args.only == "end_to_end"):
        rows = run_end_to_end(bundle, intent_classifier, k=args.k, seed=args.seed)
        save_per_query_raw(rows, out_dir)
        save_overall_results(rows, out_dir)
        save_query_type_split(rows, out_dir)
        save_ablation_results(rows, out_dir)
        save_significance_tests(rows, out_dir)

    # ── Part 5: ANN sweep ──
    if data_dir and (run_all or args.only == "ann_sweep"):
        run_ann_sweep(data_dir, out_dir, seed=args.seed)

    # ── Part 7: Case studies ──
    if bundle and (run_all or args.only == "case_study"):
        run_case_studies(bundle, intent_classifier, k=args.k, out_dir=out_dir)

    print(f"\n{'='*70}")
    print(f"  All Chapter 7 experiments complete!")
    print(f"  Output directory: {out_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
