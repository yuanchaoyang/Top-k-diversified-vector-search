#!/usr/bin/env python3
"""
严格评估 Intent Model vs 传统方法

生成论文需要的:
  - Table 1: 总体指标对比 (含 oracle upper bound)
  - Figure 1: Relevance-Diversity tradeoff 散点图
  - Table 2: 分层分析 (模糊/中等/明确 queries)
  - Table 3: 统计显著性检验
  - Figure 2: Intent Score → λ mapping 可视化
  - Case studies: 具体例子

用法:
  PYTHONPATH=src python scripts/evaluate_intent_model.py \
    --words-file data/corpus_words.txt \
    --embeddings-file data/corpus_words_embeddings.npy \
    --out-dir outputs/evaluation
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from diverse_search.adaptive_lambda import AdaptiveLambdaConfig, adaptive_lambda_from_scores
from diverse_search.index import build_retriever
from diverse_search.metrics import avg_pairwise_cosine, mean_cosine_to_query
from diverse_search.mmr import mmr_rerank_incremental
from diverse_search.intent_model import IntentClassifier


# ======================================================================
# Data loading
# ======================================================================


def load_word_embeddings(
    words_file: str, embeddings_file: str
) -> Tuple[List[str], np.ndarray]:
    with open(words_file) as f:
        words = [line.strip() for line in f if line.strip()]
    embeddings = np.load(embeddings_file).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    return words, embeddings


# ======================================================================
# Per-query evaluation
# ======================================================================


def evaluate_single_query(
    query_vec: np.ndarray,
    selected_indices: np.ndarray,
    all_vecs: np.ndarray,
) -> Dict[str, float]:
    """Compute relevance and diversity for one query."""
    sel_vecs = all_vecs[selected_indices]
    rel = mean_cosine_to_query(query_vec, sel_vecs)
    avg_cos = avg_pairwise_cosine(sel_vecs)
    ild = 1.0 - avg_cos
    return {"relevance": rel, "diversity": ild, "avg_cos": avg_cos}


def run_method_on_query(
    query_vec: np.ndarray,
    candidates: np.ndarray,
    all_vecs: np.ndarray,
    k: int,
    lambda_: float,
) -> np.ndarray:
    """Run MMR with given lambda on one query, return selected indices."""
    return mmr_rerank_incremental(
        query_vec, candidates, all_vecs, k=k, lambda_=float(lambda_)
    )


# ======================================================================
# Oracle: per-query optimal lambda via grid search
# ======================================================================


def oracle_lambda(
    query_vec: np.ndarray,
    candidates: np.ndarray,
    all_vecs: np.ndarray,
    k: int,
    alpha: float = 0.5,
    grid: np.ndarray = None,
) -> Tuple[float, float]:
    """Find lambda that maximizes alpha*relevance + (1-alpha)*diversity.

    Args:
        alpha: weight for relevance in combined score (0.5 = equal weight).
        grid: lambda values to try.

    Returns:
        (best_lambda, best_combined_score)
    """
    if grid is None:
        grid = np.arange(0.1, 1.0, 0.05)

    best_lam = 0.5
    best_score = -1.0

    for lam in grid:
        sel = run_method_on_query(query_vec, candidates, all_vecs, k, float(lam))
        m = evaluate_single_query(query_vec, sel, all_vecs)
        combined = alpha * m["relevance"] + (1.0 - alpha) * m["diversity"]
        if combined > best_score:
            best_score = combined
            best_lam = float(lam)

    return best_lam, best_score


# ======================================================================
# Main evaluation
# ======================================================================


def run_full_evaluation(
    words: List[str],
    embeddings: np.ndarray,
    intent_classifier: IntentClassifier,
    k: int = 5,
    topN: int = 50,
    lambda_min: float = 0.4,
    lambda_max: float = 0.85,
    oracle_alpha: float = 0.5,
) -> pd.DataFrame:
    """Run all methods on all queries, return per-query results."""

    retriever = build_retriever(embeddings, backend="numpy")

    # Method configs
    fixed_lambdas = [0.5, 0.6, 0.7, 0.8]
    adaptive_strategies = ["entropy", "gap_piecewise", "hybrid"]

    rows: List[Dict] = []

    for qi, (word, qvec) in enumerate(zip(words, embeddings)):
        if (qi + 1) % 100 == 0:
            print(f"  Processing query {qi + 1}/{len(words)}...")

        # Retrieve candidates (shared across methods)
        res = retriever.search(qvec.reshape(1, -1), topN)
        candidates = res.indices[0]
        scores = res.scores[0]

        base_row = {"query_idx": qi, "query_word": word}

        # --- Baseline (top-k, no reranking) ---
        sel_baseline = candidates[:k]
        m = evaluate_single_query(qvec, sel_baseline, embeddings)
        rows.append({
            **base_row, "method": "baseline", "lambda": 1.0,
            "intent_score": None, **m,
        })

        # --- Fixed lambda ---
        for lam in fixed_lambdas:
            sel = run_method_on_query(qvec, candidates, embeddings, k, lam)
            m = evaluate_single_query(qvec, sel, embeddings)
            rows.append({
                **base_row, "method": f"fixed_{lam}", "lambda": lam,
                "intent_score": None, **m,
            })

        # --- Adaptive (score-based) ---
        for strategy in adaptive_strategies:
            cfg = AdaptiveLambdaConfig(
                strategy=strategy, lambda_min=lambda_min, lambda_max=lambda_max,
            )
            lam = adaptive_lambda_from_scores(scores, cfg)
            sel = run_method_on_query(qvec, candidates, embeddings, k, lam)
            m = evaluate_single_query(qvec, sel, embeddings)
            rows.append({
                **base_row, "method": f"adaptive_{strategy}", "lambda": lam,
                "intent_score": None, **m,
            })

        # --- Intent model ---
        lam_intent, intent_score = intent_classifier.predict_lambda(
            word, lambda_min, lambda_max
        )
        sel = run_method_on_query(qvec, candidates, embeddings, k, lam_intent)
        m = evaluate_single_query(qvec, sel, embeddings)
        rows.append({
            **base_row, "method": "intent_model", "lambda": lam_intent,
            "intent_score": intent_score, **m,
        })

        # --- Oracle (upper bound) ---
        best_lam, _ = oracle_lambda(
            qvec, candidates, embeddings, k, alpha=oracle_alpha,
        )
        sel = run_method_on_query(qvec, candidates, embeddings, k, best_lam)
        m = evaluate_single_query(qvec, sel, embeddings)
        rows.append({
            **base_row, "method": "oracle", "lambda": best_lam,
            "intent_score": None, **m,
        })

    return pd.DataFrame(rows)


# ======================================================================
# Analysis and tables
# ======================================================================


def table_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1: Overall method comparison."""
    agg = df.groupby("method").agg(
        relevance_mean=("relevance", "mean"),
        relevance_std=("relevance", "std"),
        diversity_mean=("diversity", "mean"),
        diversity_std=("diversity", "std"),
        lambda_mean=("lambda", "mean"),
        lambda_std=("lambda", "std"),
    ).reset_index()

    # Combined score (harmonic mean as secondary, but report rel/div separately)
    agg["combined_05"] = 0.5 * agg["relevance_mean"] + 0.5 * agg["diversity_mean"]

    # Sort: oracle first, then by combined score
    method_order = {
        "oracle": 0, "intent_model": 1,
        "adaptive_hybrid": 2, "adaptive_entropy": 3, "adaptive_gap_piecewise": 4,
    }
    agg["sort_key"] = agg["method"].map(
        lambda m: method_order.get(m, 10)
    )
    agg = agg.sort_values("sort_key").drop(columns="sort_key")

    return agg


def table_stratified(df: pd.DataFrame) -> pd.DataFrame:
    """Table 2: Stratified by intent score (ambiguous / mid / specific)."""
    intent_df = df[df["method"] == "intent_model"][["query_word", "intent_score"]].copy()
    intent_df = intent_df.rename(columns={"intent_score": "intent"})

    # Assign group
    def _group(s):
        if s is None or np.isnan(s):
            return "unknown"
        if s < 0.3:
            return "ambiguous"
        if s < 0.7:
            return "moderate"
        return "specific"

    intent_df["intent_group"] = intent_df["intent"].apply(_group)

    # Merge group back to all methods
    merged = df.merge(intent_df[["query_word", "intent_group"]], on="query_word", how="left")

    # Filter to key methods for the stratified table
    key_methods = [
        "baseline", "fixed_0.7", "adaptive_hybrid", "intent_model", "oracle",
    ]
    merged = merged[merged["method"].isin(key_methods)]

    strat = merged.groupby(["intent_group", "method"]).agg(
        relevance=("relevance", "mean"),
        diversity=("diversity", "mean"),
        n_queries=("query_word", "count"),
    ).reset_index()

    return strat


def statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Table 3: Paired Wilcoxon signed-rank tests (intent_model vs each baseline)."""
    from scipy.stats import wilcoxon

    intent_data = df[df["method"] == "intent_model"].sort_values("query_idx")
    combined_intent = (
        0.5 * intent_data["relevance"].values + 0.5 * intent_data["diversity"].values
    )

    results = []
    baselines = [
        "baseline", "fixed_0.5", "fixed_0.6", "fixed_0.7", "fixed_0.8",
        "adaptive_entropy", "adaptive_gap_piecewise", "adaptive_hybrid",
    ]

    for bm in baselines:
        bm_data = df[df["method"] == bm].sort_values("query_idx")
        if len(bm_data) != len(intent_data):
            continue

        combined_bm = (
            0.5 * bm_data["relevance"].values + 0.5 * bm_data["diversity"].values
        )
        diff = combined_intent - combined_bm
        n_better = int(np.sum(diff > 0))
        n_worse = int(np.sum(diff < 0))

        try:
            stat, pval = wilcoxon(combined_intent, combined_bm)
        except Exception:
            stat, pval = np.nan, np.nan

        results.append({
            "intent_model vs": bm,
            "mean_diff": float(np.mean(diff)),
            "n_better": n_better,
            "n_worse": n_worse,
            "wilcoxon_stat": float(stat) if np.isfinite(stat) else None,
            "p_value": float(pval) if np.isfinite(pval) else None,
            "significant_005": pval < 0.05 if np.isfinite(pval) else None,
        })

    return pd.DataFrame(results)


def gap_to_oracle(df: pd.DataFrame) -> pd.DataFrame:
    """How close is each method to oracle? (percentage of oracle combined score)"""
    oracle_scores = df[df["method"] == "oracle"].sort_values("query_idx")
    oracle_combined = (
        0.5 * oracle_scores["relevance"].values + 0.5 * oracle_scores["diversity"].values
    )
    mean_oracle = float(np.mean(oracle_combined))

    rows = []
    for method in df["method"].unique():
        if method == "oracle":
            continue
        m_data = df[df["method"] == method].sort_values("query_idx")
        m_combined = 0.5 * m_data["relevance"].values + 0.5 * m_data["diversity"].values
        mean_m = float(np.mean(m_combined))
        pct_of_oracle = mean_m / mean_oracle * 100 if mean_oracle > 0 else 0
        rows.append({
            "method": method,
            "combined_score": mean_m,
            "oracle_score": mean_oracle,
            "pct_of_oracle": pct_of_oracle,
        })

    return pd.DataFrame(rows).sort_values("pct_of_oracle", ascending=False)


# ======================================================================
# Case studies
# ======================================================================


def case_studies(
    words: List[str],
    embeddings: np.ndarray,
    intent_classifier: IntentClassifier,
    k: int = 5,
    topN: int = 50,
    lambda_min: float = 0.4,
    lambda_max: float = 0.85,
) -> List[Dict]:
    """Generate detailed case studies for selected words."""
    # Pick representative words: ambiguous, moderate, specific
    case_words = [
        "apple", "python", "java", "bank", "spring",   # ambiguous
        "music", "cooking", "travel",                    # moderate
        "machine learning", "iphone 15", "eiffel tower", "photosynthesis",  # specific
    ]

    retriever = build_retriever(embeddings, backend="numpy")
    word_to_idx = {w: i for i, w in enumerate(words)}

    cases = []
    for cw in case_words:
        if cw not in word_to_idx:
            continue

        qi = word_to_idx[cw]
        qvec = embeddings[qi]
        res = retriever.search(qvec.reshape(1, -1), topN)
        candidates = res.indices[0]
        scores = res.scores[0]

        lam_intent, intent_score = intent_classifier.predict_lambda(
            cw, lambda_min, lambda_max
        )

        case = {"word": cw, "intent_score": intent_score, "lambda_intent": lam_intent}

        # Compare: fixed λ=0.7 vs intent model
        for label, lam in [("fixed_0.7", 0.7), ("intent", float(lam_intent))]:
            sel = run_method_on_query(qvec, candidates, embeddings, k, lam)
            sel_words = [words[i] for i in sel]
            m = evaluate_single_query(qvec, sel, embeddings)
            case[f"{label}_results"] = sel_words
            case[f"{label}_relevance"] = round(m["relevance"], 4)
            case[f"{label}_diversity"] = round(m["diversity"], 4)

        cases.append(case)

    return cases


# ======================================================================
# Plotting
# ======================================================================


def plot_tradeoff(df: pd.DataFrame, out_path: Path):
    """Figure 1: Relevance-Diversity tradeoff scatter."""
    import matplotlib.pyplot as plt

    agg = df.groupby("method").agg(
        rel=("relevance", "mean"), div=("diversity", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color/marker mapping
    style = {
        "baseline":                 {"c": "#999999", "m": "s", "s": 100},
        "fixed_0.5":                {"c": "#3498db", "m": "o", "s": 60},
        "fixed_0.6":                {"c": "#2980b9", "m": "o", "s": 60},
        "fixed_0.7":                {"c": "#2471a3", "m": "o", "s": 60},
        "fixed_0.8":                {"c": "#1a5276", "m": "o", "s": 60},
        "adaptive_entropy":         {"c": "#27ae60", "m": "^", "s": 100},
        "adaptive_gap_piecewise":   {"c": "#2ecc71", "m": "^", "s": 100},
        "adaptive_hybrid":          {"c": "#16a085", "m": "^", "s": 100},
        "intent_model":             {"c": "#e74c3c", "m": "*", "s": 250},
        "oracle":                   {"c": "#f1c40f", "m": "D", "s": 120},
    }

    for _, row in agg.iterrows():
        method = row["method"]
        st = style.get(method, {"c": "#777", "m": "x", "s": 60})
        ax.scatter(
            row["div"], row["rel"],
            c=st["c"], marker=st["m"], s=st["s"],
            label=method, edgecolors="black", linewidths=0.5, zorder=3,
        )

    ax.set_xlabel("Diversity (ILD = 1 - avg pairwise cosine)", fontsize=12)
    ax.set_ylabel("Relevance (mean cosine to query)", fontsize=12)
    ax.set_title("Relevance-Diversity Trade-off", fontsize=14)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def plot_intent_lambda_scatter(df: pd.DataFrame, out_path: Path):
    """Figure 2: Intent score vs lambda, colored by diversity gain over fixed."""
    import matplotlib.pyplot as plt

    intent_df = df[df["method"] == "intent_model"].copy()
    fixed_df = df[df["method"] == "fixed_0.7"].sort_values("query_idx")

    intent_df = intent_df.sort_values("query_idx")

    div_gain = intent_df["diversity"].values - fixed_df["diversity"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        intent_df["intent_score"], intent_df["lambda"],
        c=div_gain, cmap="RdYlGn", s=30, alpha=0.7, edgecolors="black", linewidths=0.3,
    )
    ax.plot(
        [0, 1],
        [intent_df["lambda"].min(), intent_df["lambda"].max()],
        "k--", alpha=0.4, label="linear trend",
    )
    ax.set_xlabel("Intent Score (0 = ambiguous, 1 = specific)", fontsize=12)
    ax.set_ylabel("Adaptive λ", fontsize=12)
    ax.set_title("Intent Score → λ Mapping", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Diversity gain vs fixed λ=0.7")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def plot_stratified_bars(strat_df: pd.DataFrame, out_path: Path):
    """Figure 3: Stratified bar chart — diversity by intent group."""
    import matplotlib.pyplot as plt

    groups = ["ambiguous", "moderate", "specific"]
    methods = ["fixed_0.7", "adaptive_hybrid", "intent_model"]
    colors = {"fixed_0.7": "#3498db", "adaptive_hybrid": "#2ecc71", "intent_model": "#e74c3c"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in [
        (axes[0], "diversity", "Diversity by Intent Group"),
        (axes[1], "relevance", "Relevance by Intent Group"),
    ]:
        x = np.arange(len(groups))
        width = 0.25
        for i, m in enumerate(methods):
            vals = []
            for g in groups:
                sub = strat_df[(strat_df["intent_group"] == g) & (strat_df["method"] == m)]
                vals.append(float(sub[metric].values[0]) if len(sub) > 0 else 0)
            ax.bar(x + i * width, vals, width, label=m, color=colors[m], edgecolor="black", linewidth=0.5)

        ax.set_xticks(x + width)
        ax.set_xticklabels(groups)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description="Evaluate intent model vs baselines")
    parser.add_argument("--words-file", default="data/corpus_words.txt")
    parser.add_argument("--embeddings-file", default="data/corpus_words_embeddings.npy")
    parser.add_argument("--model-dir", default="models/intent_chatgpt")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--topN", type=int, default=50)
    parser.add_argument("--lambda-min", type=float, default=0.4)
    parser.add_argument("--lambda-max", type=float, default=0.85)
    parser.add_argument("--oracle-alpha", type=float, default=0.5,
                        help="Weight for relevance in oracle combined score")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("INTENT MODEL EVALUATION")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading data...")
    words, embeddings = load_word_embeddings(args.words_file, args.embeddings_file)
    print(f"  {len(words)} words, dim={embeddings.shape[1]}")

    # Load intent model
    print("\n[2/6] Loading intent model...")
    classifier = IntentClassifier.load(args.model_dir)
    print(f"  Model dir: {args.model_dir}")
    print(f"  Label cache: {len(classifier.label_cache)} entries")

    # Run evaluation
    print(f"\n[3/6] Running evaluation (k={args.k}, topN={args.topN})...")
    print(f"  This evaluates {len(words)} queries × 10 methods (incl. oracle grid search)...")
    df = run_full_evaluation(
        words, embeddings, classifier,
        k=args.k, topN=args.topN,
        lambda_min=args.lambda_min, lambda_max=args.lambda_max,
        oracle_alpha=args.oracle_alpha,
    )
    df.to_csv(args.out_dir / "per_query_results.csv", index=False)
    print(f"  Saved: {args.out_dir / 'per_query_results.csv'}")

    # ---- Table 1: Overall ----
    print(f"\n[4/6] Generating tables...")

    t1 = table_overall(df)
    t1.to_csv(args.out_dir / "table1_overall.csv", index=False)
    print("\n  === Table 1: Overall Method Comparison ===")
    print(t1.to_string(index=False, float_format="%.4f"))

    # Gap to oracle
    gap_df = gap_to_oracle(df)
    gap_df.to_csv(args.out_dir / "gap_to_oracle.csv", index=False)
    print("\n  === Gap to Oracle ===")
    print(gap_df.to_string(index=False, float_format="%.2f"))

    # ---- Table 2: Stratified ----
    t2 = table_stratified(df)
    t2.to_csv(args.out_dir / "table2_stratified.csv", index=False)
    print("\n  === Table 2: Stratified by Intent Group ===")
    print(t2.to_string(index=False, float_format="%.4f"))

    # ---- Table 3: Statistical tests ----
    try:
        t3 = statistical_tests(df)
        t3.to_csv(args.out_dir / "table3_significance.csv", index=False)
        print("\n  === Table 3: Statistical Significance (Wilcoxon) ===")
        print(t3.to_string(index=False, float_format="%.4f"))
    except ImportError:
        print("\n  [SKIP] scipy not installed, skipping statistical tests")
        print("  Install with: pip install scipy")

    # ---- Case studies ----
    print(f"\n[5/6] Case studies...")
    cases = case_studies(
        words, embeddings, classifier,
        k=args.k, topN=args.topN,
        lambda_min=args.lambda_min, lambda_max=args.lambda_max,
    )
    with open(args.out_dir / "case_studies.json", "w") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    print("\n  === Case Studies ===")
    for c in cases:
        intent_tag = "ambiguous" if c["intent_score"] < 0.3 else (
            "specific" if c["intent_score"] > 0.7 else "moderate"
        )
        print(f"\n  [{intent_tag}] \"{c['word']}\" (intent={c['intent_score']:.2f}, λ={c['lambda_intent']:.2f})")
        print(f"    Fixed λ=0.7:  {c['fixed_0.7_results']}")
        print(f"      rel={c['fixed_0.7_relevance']:.4f}  div={c['fixed_0.7_diversity']:.4f}")
        print(f"    Intent model: {c['intent_results']}")
        print(f"      rel={c['intent_relevance']:.4f}  div={c['intent_diversity']:.4f}")

    # ---- Plots ----
    print(f"\n[6/6] Generating plots...")
    try:
        plot_tradeoff(df, args.out_dir / "fig1_tradeoff.png")
        plot_intent_lambda_scatter(df, args.out_dir / "fig2_intent_lambda.png")
        t2_for_plot = table_stratified(df)
        plot_stratified_bars(t2_for_plot, args.out_dir / "fig3_stratified.png")
    except ImportError:
        print("  matplotlib not available, skipping plots")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    intent_row = t1[t1["method"] == "intent_model"].iloc[0]
    oracle_row = t1[t1["method"] == "oracle"].iloc[0]
    best_fixed = t1[t1["method"].str.startswith("fixed_")]["combined_05"].max()
    best_adaptive = t1[t1["method"].str.startswith("adaptive_")]["combined_05"].max()
    intent_combined = intent_row["combined_05"]

    print(f"\n  Intent Model combined score:  {intent_combined:.4f}")
    print(f"  Best fixed λ combined score:  {best_fixed:.4f}")
    print(f"  Best adaptive combined score: {best_adaptive:.4f}")
    print(f"  Oracle combined score:        {oracle_row['combined_05']:.4f}")
    print()

    if intent_combined > best_fixed:
        pct = (intent_combined - best_fixed) / best_fixed * 100
        print(f"  Intent Model beats best fixed λ by {pct:.2f}%")
    else:
        pct = (best_fixed - intent_combined) / best_fixed * 100
        print(f"  Intent Model is {pct:.2f}% below best fixed λ")

    if intent_combined > best_adaptive:
        pct = (intent_combined - best_adaptive) / best_adaptive * 100
        print(f"  Intent Model beats best adaptive by {pct:.2f}%")

    oracle_gap_intent = gap_df[gap_df["method"] == "intent_model"]["pct_of_oracle"].values[0]
    oracle_gap_hybrid = gap_df[gap_df["method"] == "adaptive_hybrid"]["pct_of_oracle"].values
    if len(oracle_gap_hybrid) > 0:
        print(f"\n  Closeness to oracle: intent={oracle_gap_intent:.1f}%  adaptive_hybrid={oracle_gap_hybrid[0]:.1f}%")

    print(f"\n  All outputs saved to: {args.out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
