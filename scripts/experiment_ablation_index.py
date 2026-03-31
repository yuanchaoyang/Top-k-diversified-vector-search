#!/usr/bin/env python3
"""
IVF vs HNSW vs 动态切换 — 消融实验
======================================

按 intent_score 将查询分为「模糊组」和「明确组」，对比三种检索策略：
  A) 全部用 IVF   (+cluster_balanced +cluster_penalty)
  B) 全部用 HNSW
  C) 动态切换（模糊→IVF, 明确→HNSW，即现有方案）

对每种策略测量：
  - Relevance  (平均余弦相似度)
  - Diversity  (ILD = 1 - 平均 pairwise cosine)
  - F1         (relevance 和 diversity 的调和平均)

同时扫描阈值 (0.2, 0.3, 0.4, 0.5, 0.6) 寻找最优切分点。

使用方法:
    PYTHONPATH=src python scripts/experiment_ablation_index.py
    PYTHONPATH=src python scripts/experiment_ablation_index.py --data-dir data/improved --out-dir outputs/ablation
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diverse_search.index import build_retriever
from diverse_search.ivf import compute_adaptive_nprobe, compute_adaptive_topN
from diverse_search.hnsw import compute_adaptive_ef
from diverse_search.mmr import mmr_rerank_temporal
from diverse_search.metrics import avg_pairwise_cosine
from diverse_search.intent_model import IntentClassifier
from diverse_search.temporal import query_aware_lambda_and_beta, TemporalConfig


# ═══════════════════════════════════════════════════════════════════
#  单条查询评估
# ═══════════════════════════════════════════════════════════════════

def evaluate_single(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    retriever,
    *,
    k: int,
    top_n: int,
    lam: float,
    intent_score: float,
    beta: float = 0.0,
    freshness: np.ndarray | None = None,
    cluster_balanced: bool = False,
    cluster_ids: np.ndarray | None = None,
    cluster_penalty: float = 0.0,
) -> dict:
    """用指定 retriever 检索 + MMR 重排，返回 relevance/diversity/F1。"""
    q_2d = query_vec.reshape(1, -1)

    # 检索
    t0 = time.perf_counter()
    if cluster_balanced and hasattr(retriever, "search"):
        # IVF 的 cluster_balanced 模式
        try:
            res = retriever.search(q_2d, top_n, cluster_balanced=True)
        except TypeError:
            res = retriever.search(q_2d, top_n)
    else:
        res = retriever.search(q_2d, top_n)
    search_ms = (time.perf_counter() - t0) * 1000

    candidates = res.indices[0]
    scores = res.scores[0]

    # quality boost（复用 demo 逻辑）
    top_score = float(scores[0]) if len(scores) > 0 else 0.0
    adjusted_lam = lam
    if top_score < 0.5:
        quality_boost = (0.5 - top_score) * 1.0
        adjusted_lam = min(0.9, adjusted_lam + quality_boost)

    min_ratio = 0.60 + 0.25 * intent_score

    # MMR rerank
    selected = mmr_rerank_temporal(
        query_vec, candidates, embeddings, k=k, lambda_=float(adjusted_lam),
        freshness=freshness, beta=float(beta),
        min_score_ratio=min_ratio,
        cluster_ids=cluster_ids, cluster_penalty=cluster_penalty,
    )

    # 计算指标
    if len(selected) < 2:
        return {"relevance": 0.0, "diversity": 0.0, "f1": 0.0, "n_results": len(selected),
                "search_ms": search_ms}

    selected_vecs = embeddings[selected]
    sims = np.dot(selected_vecs, query_vec)
    relevance = float(np.mean(sims))
    diversity = float(1.0 - avg_pairwise_cosine(selected_vecs))
    f1 = 2 * relevance * diversity / (relevance + diversity) if (relevance + diversity) > 0 else 0.0

    return {
        "relevance": relevance,
        "diversity": diversity,
        "f1": f1,
        "n_results": len(selected),
        "search_ms": search_ms,
    }


# ═══════════════════════════════════════════════════════════════════
#  主实验
# ═══════════════════════════════════════════════════════════════════

def run_ablation(
    queries: list[str],
    query_vecs: np.ndarray,
    embeddings: np.ndarray,
    ivf_retriever,
    hnsw_retriever,
    intent_classifier: IntentClassifier,
    *,
    k: int = 10,
    freshness: np.ndarray | None = None,
    thresholds: list[float] = None,
) -> dict:
    """运行完整消融实验。"""
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]

    nq = len(queries)
    cluster_ids = getattr(ivf_retriever, "assignments", None)

    # 1) 计算每条查询的 intent 信号
    print("Computing intent signals for all queries...")
    intent_data = []
    for i, query in enumerate(queries):
        ta = query_aware_lambda_and_beta(
            query, intent_classifier,
            temporal_config=TemporalConfig(),
            lambda_min=0.30, lambda_max=0.9,
        )
        intent_data.append(ta)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{nq} queries analyzed")

    intent_scores = np.array([d["intent_score"] for d in intent_data])
    print(f"  Intent score distribution: mean={intent_scores.mean():.3f}, "
          f"median={np.median(intent_scores):.3f}, "
          f"std={intent_scores.std():.3f}")
    print(f"  Histogram: <0.2: {(intent_scores < 0.2).sum()}, "
          f"0.2-0.4: {((intent_scores >= 0.2) & (intent_scores < 0.4)).sum()}, "
          f"0.4-0.6: {((intent_scores >= 0.4) & (intent_scores < 0.6)).sum()}, "
          f"0.6-0.8: {((intent_scores >= 0.6) & (intent_scores < 0.8)).sum()}, "
          f"≥0.8: {(intent_scores >= 0.8).sum()}")

    # 2) 三种策略评估
    strategies = ["all_ivf", "all_hnsw"]
    # 每个阈值对应一个 dynamic 策略
    for t in thresholds:
        strategies.append(f"dynamic_{t:.1f}")

    # 存储每条查询的结果: results[strategy][query_idx] = {relevance, diversity, f1, ...}
    results: dict[str, list[dict]] = {s: [] for s in strategies}

    print(f"\nEvaluating {nq} queries × {len(strategies)} strategies...")

    for i in range(nq):
        query_vec = query_vecs[i]
        ta = intent_data[i]
        intent_score = ta["intent_score"]
        lam = ta["lambda"]
        beta = ta["beta"]
        temporal_score = ta["temporal_score"]

        top_n_ivf = compute_adaptive_topN(intent_score, temporal_score)
        nprobe = compute_adaptive_nprobe(intent_score, temporal_score)
        ivf_retriever.set_nprobe(nprobe)

        top_n_hnsw = compute_adaptive_topN(intent_score, temporal_score)
        ef = compute_adaptive_ef(intent_score, temporal_score)
        hnsw_retriever.set_ef_search(ef)

        # A) 全部用 IVF（带 cluster_balanced + cluster_penalty）
        r = evaluate_single(
            query_vec, embeddings, ivf_retriever,
            k=k, top_n=top_n_ivf, lam=lam, intent_score=intent_score,
            beta=beta, freshness=freshness,
            cluster_balanced=True,
            cluster_ids=cluster_ids, cluster_penalty=0.15,
        )
        r["intent_score"] = intent_score
        results["all_ivf"].append(r)

        # B) 全部用 HNSW（无 cluster 信息）
        r = evaluate_single(
            query_vec, embeddings, hnsw_retriever,
            k=k, top_n=top_n_hnsw, lam=lam, intent_score=intent_score,
            beta=beta, freshness=freshness,
            cluster_balanced=False,
            cluster_ids=None, cluster_penalty=0.0,
        )
        r["intent_score"] = intent_score
        results["all_hnsw"].append(r)

        # C) 动态切换（按不同阈值）
        for t in thresholds:
            key = f"dynamic_{t:.1f}"
            if intent_score < t:
                # 模糊 → IVF
                r = evaluate_single(
                    query_vec, embeddings, ivf_retriever,
                    k=k, top_n=top_n_ivf, lam=lam, intent_score=intent_score,
                    beta=beta, freshness=freshness,
                    cluster_balanced=True,
                    cluster_ids=cluster_ids, cluster_penalty=0.15,
                )
            else:
                # 明确 → HNSW
                r = evaluate_single(
                    query_vec, embeddings, hnsw_retriever,
                    k=k, top_n=top_n_hnsw, lam=lam, intent_score=intent_score,
                    beta=beta, freshness=freshness,
                    cluster_balanced=False,
                    cluster_ids=None, cluster_penalty=0.0,
                )
            r["intent_score"] = intent_score
            results[key].append(r)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{nq} queries evaluated")

    print(f"  All {nq} queries evaluated\n")

    return {
        "results": results,
        "intent_scores": intent_scores,
        "thresholds": thresholds,
        "strategies": strategies,
        "nq": nq,
        "k": k,
    }


# ═══════════════════════════════════════════════════════════════════
#  结果分析与输出
# ═══════════════════════════════════════════════════════════════════

def analyze_and_print(experiment: dict) -> list[dict]:
    """分组统计 + 打印结果表格，返回汇总行用于 CSV。"""
    results = experiment["results"]
    intent_scores = experiment["intent_scores"]
    thresholds = experiment["thresholds"]
    strategies = experiment["strategies"]
    nq = experiment["nq"]

    summary_rows = []

    def agg(rows: list[dict]) -> dict:
        """聚合一组查询的指标。"""
        if not rows:
            return {"relevance": 0, "diversity": 0, "f1": 0, "n": 0, "ms": 0}
        return {
            "relevance": np.mean([r["relevance"] for r in rows]),
            "diversity": np.mean([r["diversity"] for r in rows]),
            "f1": np.mean([r["f1"] for r in rows]),
            "n": len(rows),
            "ms": np.mean([r["search_ms"] for r in rows]),
        }

    # ── 总体对比 ──
    print("=" * 90)
    print("Overall Results (all queries)")
    print("=" * 90)
    header = f"{'Strategy':<20} {'N':>5} {'Relevance':>10} {'Diversity':>10} {'F1':>10} {'ms/q':>8}"
    print(header)
    print("-" * 90)

    for s in strategies:
        a = agg(results[s])
        print(f"{s:<20} {a['n']:>5} {a['relevance']:>10.4f} {a['diversity']:>10.4f} "
              f"{a['f1']:>10.4f} {a['ms']:>8.2f}")
        summary_rows.append({
            "group": "overall", "strategy": s,
            "n_queries": a["n"],
            "relevance": round(a["relevance"], 4),
            "diversity": round(a["diversity"], 4),
            "f1": round(a["f1"], 4),
            "ms_per_query": round(a["ms"], 2),
        })

    # ── 按阈值分组对比 ──
    # 用 0.4 作为主要分组（同时也展示其他阈值的效果）
    for split_t in thresholds:
        mask_ambig = intent_scores < split_t
        mask_clear = ~mask_ambig
        n_ambig = mask_ambig.sum()
        n_clear = mask_clear.sum()

        print(f"\n{'=' * 90}")
        print(f"Split at intent_score = {split_t:.1f}  "
              f"(ambiguous: {n_ambig}, clear: {n_clear})")
        print("=" * 90)

        for group_name, mask in [("ambiguous", mask_ambig), ("clear", mask_clear)]:
            if mask.sum() == 0:
                continue
            print(f"\n  {group_name.upper()} queries (n={mask.sum()}):")
            print(f"  {'Strategy':<20} {'Relevance':>10} {'Diversity':>10} {'F1':>10} {'ms/q':>8}")
            print(f"  {'-' * 70}")

            for s in strategies:
                group_rows = [results[s][j] for j in range(nq) if mask[j]]
                a = agg(group_rows)
                print(f"  {s:<20} {a['relevance']:>10.4f} {a['diversity']:>10.4f} "
                      f"{a['f1']:>10.4f} {a['ms']:>8.2f}")
                summary_rows.append({
                    "group": f"{group_name}_split{split_t:.1f}",
                    "strategy": s,
                    "n_queries": a["n"],
                    "relevance": round(a["relevance"], 4),
                    "diversity": round(a["diversity"], 4),
                    "f1": round(a["f1"], 4),
                    "ms_per_query": round(a["ms"], 2),
                })

    # ── 最优阈值总结 ──
    print(f"\n{'=' * 90}")
    print("Threshold Sensitivity (overall F1)")
    print("=" * 90)
    print(f"{'Strategy':<20} {'F1':>10}")
    print("-" * 35)

    best_t, best_f1 = None, -1
    for s in strategies:
        a = agg(results[s])
        print(f"{s:<20} {a['f1']:>10.4f}")
        if s.startswith("dynamic_") and a["f1"] > best_f1:
            best_f1 = a["f1"]
            best_t = s

    ivf_f1 = agg(results["all_ivf"])["f1"]
    hnsw_f1 = agg(results["all_hnsw"])["f1"]
    print(f"\nBest dynamic threshold: {best_t}  (F1={best_f1:.4f})")
    print(f"  vs all_ivf:  F1 delta = {best_f1 - ivf_f1:+.4f}")
    print(f"  vs all_hnsw: F1 delta = {best_f1 - hnsw_f1:+.4f}")

    return summary_rows


def save_results(summary_rows: list[dict], out_dir: Path) -> None:
    """保存 CSV 和可视化图。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "ablation_results.csv"
    fieldnames = ["group", "strategy", "n_queries", "relevance", "diversity", "f1", "ms_per_query"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSaved: {csv_path}")

    # 可视化
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    # 提取 overall 数据
    overall = [r for r in summary_rows if r["group"] == "overall"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    strategies = [r["strategy"] for r in overall]
    relevances = [r["relevance"] for r in overall]
    diversities = [r["diversity"] for r in overall]
    f1s = [r["f1"] for r in overall]

    # 短标签
    labels = []
    for s in strategies:
        if s == "all_ivf":
            labels.append("All IVF")
        elif s == "all_hnsw":
            labels.append("All HNSW")
        else:
            t = s.replace("dynamic_", "Dyn t=")
            labels.append(t)

    colors = []
    for s in strategies:
        if s == "all_ivf":
            colors.append("tab:blue")
        elif s == "all_hnsw":
            colors.append("tab:orange")
        else:
            colors.append("tab:green")

    # Relevance
    ax = axes[0]
    ax.bar(range(len(labels)), relevances, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Relevance")
    ax.set_title("Mean Relevance")
    ax.grid(True, alpha=0.3, axis="y")

    # Diversity
    ax = axes[1]
    ax.bar(range(len(labels)), diversities, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Diversity (ILD)")
    ax.set_title("Mean Diversity")
    ax.grid(True, alpha=0.3, axis="y")

    # F1
    ax = axes[2]
    ax.bar(range(len(labels)), f1s, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1")
    ax.set_title("F1 (Harmonic Mean)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    png_path = out_dir / "ablation_overall.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")
    plt.close()

    # ── 分组对比图：模糊 vs 明确 (split=0.4) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, group_label in enumerate(["ambiguous", "clear"]):
        ax = axes[idx]
        group_data = [r for r in summary_rows
                      if r["group"] == f"{group_label}_split0.4"
                      and r["strategy"] in ("all_ivf", "all_hnsw", "dynamic_0.4")]

        if not group_data:
            continue

        x = np.arange(3)
        width = 0.25
        strats = [r["strategy"] for r in group_data]

        ax.bar(x - width, [r["relevance"] for r in group_data], width, label="Relevance", color="tab:blue", alpha=0.8)
        ax.bar(x, [r["diversity"] for r in group_data], width, label="Diversity", color="tab:orange", alpha=0.8)
        ax.bar(x + width, [r["f1"] for r in group_data], width, label="F1", color="tab:green", alpha=0.8)

        short = ["All IVF", "All HNSW", "Dynamic"]
        ax.set_xticks(x)
        ax.set_xticklabels(short)
        n_q = group_data[0]["n_queries"] if group_data else 0
        ax.set_title(f"{group_label.capitalize()} queries (n={n_q}, split=0.4)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.0)

    plt.tight_layout()
    png_path = out_dir / "ablation_by_group.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")
    plt.close()

    # ── 阈值敏感性曲线 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    threshold_data = [r for r in summary_rows
                      if r["group"] == "overall" and r["strategy"].startswith("dynamic_")]
    if threshold_data:
        ts = [float(r["strategy"].replace("dynamic_", "")) for r in threshold_data]
        f1s_t = [r["f1"] for r in threshold_data]
        rel_t = [r["relevance"] for r in threshold_data]
        div_t = [r["diversity"] for r in threshold_data]

        ax.plot(ts, f1s_t, "o-", color="tab:green", label="F1", linewidth=2)
        ax.plot(ts, rel_t, "s--", color="tab:blue", label="Relevance", alpha=0.7)
        ax.plot(ts, div_t, "^--", color="tab:orange", label="Diversity", alpha=0.7)

        # all_ivf / all_hnsw 基线
        ivf_row = next(r for r in summary_rows if r["group"] == "overall" and r["strategy"] == "all_ivf")
        hnsw_row = next(r for r in summary_rows if r["group"] == "overall" and r["strategy"] == "all_hnsw")
        ax.axhline(y=ivf_row["f1"], color="tab:blue", linestyle=":", alpha=0.5, label=f"All IVF F1={ivf_row['f1']:.4f}")
        ax.axhline(y=hnsw_row["f1"], color="tab:orange", linestyle=":", alpha=0.5, label=f"All HNSW F1={hnsw_row['f1']:.4f}")

    ax.set_xlabel("Threshold (intent_score split point)")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sensitivity for Dynamic Switching")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = out_dir / "ablation_threshold_sensitivity.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
#  查询加载
# ═══════════════════════════════════════════════════════════════════

# 手工设计的模糊查询：单个多义词或短泛化词组
AMBIGUOUS_WORDS = [
    # 多义词（名人/地名/普通词义）
    "apple", "amazon", "mercury", "jaguar", "python", "crane", "bass",
    "bat", "bank", "spring", "palm", "mouse", "chip", "cell", "bug",
    "plant", "rock", "seal", "drill", "bridge", "key", "match", "table",
    "rose", "date", "fan", "light", "ring", "court", "net", "pitch",
    "staff", "tank", "mine", "organ", "stock", "current", "volume",
    # 抽象/宽泛概念
    "power", "energy", "force", "culture", "system", "design", "model",
    "theory", "pattern", "structure", "process", "value", "growth",
    "intelligence", "network", "evolution", "revolution", "movement",
    # 短泛化短语
    "best practices", "how things work", "health tips", "travel guide",
    "what is art", "meaning of life", "history of science",
    "types of music", "world religions", "climate change effects",
    "machine learning", "space exploration", "ancient civilizations",
    "modern architecture", "digital transformation", "renewable energy",
    "genetic engineering", "artificial intelligence", "quantum computing",
    "deep learning applications",
]


def _encode_queries(queries: list[str]) -> np.ndarray:
    """用 sentence-transformers 编码查询。"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(queries, show_progress_bar=True, convert_to_numpy=True)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / (norms + 1e-8)).astype(np.float32)


def _build_mixed_queries(
    data_dir: Path, args, embeddings: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """构建混合查询集：模糊单词 + MS MARCO 明确查询。"""
    rng = np.random.default_rng(args.seed)

    # ── 模糊查询 ──
    n_ambig = min(args.n_ambiguous, len(AMBIGUOUS_WORDS))
    ambig_idx = rng.choice(len(AMBIGUOUS_WORDS), size=n_ambig, replace=False)
    ambig_queries = [AMBIGUOUS_WORDS[i] for i in ambig_idx]
    print(f"  Mixed mode: {n_ambig} ambiguous queries (single words/short phrases)")

    # ── 明确查询（从 MS MARCO 取）──
    clear_queries = []
    msmarco_query_file = data_dir / "msmarco_queries.txt"
    if not msmarco_query_file.exists():
        # 尝试其他目录
        for d in [Path("data/msmarco"), data_dir]:
            if (d / "msmarco_queries.txt").exists():
                msmarco_query_file = d / "msmarco_queries.txt"
                break

    if msmarco_query_file.exists():
        with open(msmarco_query_file, encoding="utf-8") as f:
            all_clear = [line.strip() for line in f if line.strip()]
        n_clear = min(args.n_clear, len(all_clear))
        clear_idx = rng.choice(len(all_clear), size=n_clear, replace=False)
        clear_queries = [all_clear[i] for i in clear_idx]
        print(f"  Mixed mode: {n_clear} clear queries from MS MARCO")
    else:
        print("  Warning: no MS MARCO queries found, using only ambiguous queries")

    queries = ambig_queries + clear_queries
    print(f"  Total: {len(queries)} mixed queries")

    # 编码
    print("  Encoding mixed queries...")
    query_vecs = _encode_queries(queries)

    return queries, query_vecs


def _load_queries(
    data_dir: Path, args,
) -> tuple[list[str], np.ndarray]:
    """从文件加载查询或自动生成。"""
    query_file = args.query_file
    query_emb_file = args.query_emb_file
    if query_file is None:
        for name in ["msmarco_queries.txt", "queries.txt"]:
            if (data_dir / name).exists():
                query_file = str(data_dir / name)
                break
    if query_emb_file is None:
        if (data_dir / "query_embeddings.npy").exists():
            query_emb_file = str(data_dir / "query_embeddings.npy")

    if query_file and query_emb_file:
        with open(query_file, encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
        query_vecs = np.load(query_emb_file).astype(np.float32)
        qn = np.linalg.norm(query_vecs, axis=1, keepdims=True)
        query_vecs = query_vecs / (qn + 1e-8)
        print(f"  Loaded {len(queries)} queries from {query_file}")
    else:
        print("No pre-computed query embeddings found, encoding queries on the fly...")
        if query_file:
            with open(query_file, encoding="utf-8") as f:
                queries = [line.strip() for line in f if line.strip()]
        else:
            passages_file = data_dir / "passages.txt"
            if not passages_file.exists():
                print(f"Error: no query file or passages.txt in {data_dir}")
                sys.exit(1)
            with open(passages_file, encoding="utf-8") as f:
                all_passages = [line.strip() for line in f if line.strip()]
            rng = np.random.default_rng(args.seed)
            sample_n = min(200, len(all_passages))
            sample_idx = rng.choice(len(all_passages), size=sample_n, replace=False)
            queries = []
            import re
            for idx in sample_idx:
                p = all_passages[idx]
                sentences = re.split(r'(?<=[.?!])\s+', p.strip(), maxsplit=1)
                queries.append(sentences[0][:100] if sentences else p[:100])
            print(f"  Generated {len(queries)} pseudo-queries from passages")

        query_vecs = _encode_queries(queries)

    # 子采样
    if args.n_queries > 0 and args.n_queries < len(queries):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(queries), size=args.n_queries, replace=False)
        queries = [queries[i] for i in idx]
        query_vecs = query_vecs[idx]
        print(f"  Subsampled to {len(queries)} queries")

    return queries, query_vecs


# ═══════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="IVF vs HNSW vs 动态切换 消融实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/msmarco",
                        help="语料库目录 (含 passage_embeddings.npy)")
    parser.add_argument("--query-file", default=None,
                        help="查询文件路径 (默认从 data-dir 自动检测)")
    parser.add_argument("--query-emb-file", default=None,
                        help="查询嵌入文件 (默认从 data-dir 自动检测)")
    parser.add_argument("--k", type=int, default=10, help="返回结果数")
    parser.add_argument("--n-queries", type=int, default=0,
                        help="使用的查询数 (0=全部)")
    parser.add_argument("--nlist", type=int, default=128, help="IVF nlist")
    parser.add_argument("--hnsw-m", type=int, default=16, help="HNSW M")
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6",
                        help="逗号分隔的阈值列表")
    parser.add_argument("--mixed-queries", action="store_true",
                        help="生成混合查询集：模糊单词 + 明确 MS MARCO 查询")
    parser.add_argument("--n-ambiguous", type=int, default=100,
                        help="混合模式中的模糊查询数")
    parser.add_argument("--n-clear", type=int, default=100,
                        help="混合模式中的明确查询数")
    parser.add_argument("--out-dir", default="outputs/ablation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # ── 加载语料库嵌入 ──
    emb_path = data_dir / "passage_embeddings.npy"
    if not emb_path.exists():
        print(f"Error: {emb_path} not found")
        sys.exit(1)

    print(f"Loading passage embeddings from {emb_path}...")
    embeddings = np.load(emb_path).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    print(f"  {embeddings.shape[0]} passages, dim={embeddings.shape[1]}")

    # ── 加载查询 ──
    if args.mixed_queries:
        queries, query_vecs = _build_mixed_queries(data_dir, args, embeddings)
    else:
        queries, query_vecs = _load_queries(data_dir, args)

    # ── 加载 freshness ──
    freshness_file = data_dir / "passage_freshness.npy"
    freshness = np.load(freshness_file) if freshness_file.exists() else None

    # ── 构建索引 ──
    print("\nBuilding IVF index...")
    ivf_retriever = build_retriever(
        embeddings, backend="numpy_ivf",
        nlist=args.nlist, nprobe=16,
    )
    print(f"  IVF built: nlist={args.nlist}")

    print("Building HNSW index...")
    hnsw_retriever = build_retriever(
        embeddings, backend="numpy_hnsw",
        hnsw_m=args.hnsw_m, ef_construction=args.ef_construction, ef_search=64,
    )
    print(f"  HNSW built: M={args.hnsw_m}, ef_construction={args.ef_construction}")

    # ── 加载意图分类器 ──
    intent_classifier = IntentClassifier()
    for model_dir in ["models/intent_v3", "models/intent_v2", "models/intent_chatgpt"]:
        model_path = Path(model_dir) / "intent_model.pkl"
        if model_path.exists():
            intent_classifier = IntentClassifier.load(model_dir)
            print(f"  Intent classifier loaded from {model_dir}")
            break
    else:
        print("  No intent model found, using default classifier")

    # ── 运行实验 ──
    thresholds = [float(t) for t in args.thresholds.split(",")]

    experiment = run_ablation(
        queries, query_vecs, embeddings,
        ivf_retriever, hnsw_retriever, intent_classifier,
        k=args.k,
        freshness=freshness,
        thresholds=thresholds,
    )

    # ── 分析输出 ──
    summary_rows = analyze_and_print(experiment)
    save_results(summary_rows, Path(args.out_dir))


if __name__ == "__main__":
    main()
