#!/usr/bin/env python3
"""
Backend Switch Threshold Sensitivity Experiment
================================================

Sweep intent_score threshold t ∈ {0.2, 0.3, 0.4, 0.5, 0.6},
plus two endpoint baselines (always_ivf, always_hnsw), to evaluate the
impact on L5 full-pipeline metrics.

All settings except the threshold are identical to experiment_chapter7.py L5:
  - Three-layer signal fusion (ML intent + factual rules + temporal)
  - Adaptive nprobe/ef, topN
  - Cluster-aware MMR + temporal freshness
  - Quality boost, min_score_ratio

Query subset split:
  ambiguous_ref / specific_ref are defined based on the **deployed intent_score
  and the default 0.4 boundary**, fixed once and reused across all thresholds
  to ensure cross-condition comparability.

Usage:
    source .venv/bin/activate
    PYTHONPATH=src python scripts/sweep_backend_switch_threshold.py \\
      --improved-dir data/improved --k 10 --out-dir outputs/backend_switch_threshold

    # Optionally specify MS MARCO as well:
    PYTHONPATH=src python scripts/sweep_backend_switch_threshold.py \\
      --msmarco-dir data/msmarco --improved-dir data/improved --k 10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from experiment_utils import (
    DatasetBundle,
    load_msmarco,
    load_improved,
    build_all_retrievers,
    compute_metrics,
)
from diverse_search.mmr import mmr_rerank_temporal
from diverse_search.intent_model import IntentClassifier
from diverse_search.temporal import (
    query_aware_lambda_and_beta,
    TemporalConfig,
)
from diverse_search.ivf import compute_adaptive_nprobe, compute_adaptive_topN
from diverse_search.hnsw import compute_adaptive_ef

# ═══════════════════════════════════════════════════════════════
#  Constants — consistent with experiment_chapter7.py L5
# ═══════════════════════════════════════════════════════════════

LAMBDA_MIN = 0.30
LAMBDA_MAX = 0.90
DEFAULT_THRESHOLD = 0.4          # currently deployed value
REF_SPLIT_THRESHOLD = 0.4        # used to define ambiguous_ref / specific_ref

SWEEP_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6]
ENDPOINT_BASELINES = ["always_ivf", "always_hnsw"]


# ═══════════════════════════════════════════════════════════════
#  Single-query L5 pipeline (configurable threshold / forced backend)
# ═══════════════════════════════════════════════════════════════

def run_l5_single_query(
    query: str,
    query_vec: np.ndarray,
    emb: np.ndarray,
    ivf,
    hnsw,
    intent_classifier: IntentClassifier,
    *,
    k: int,
    freshness: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    force_backend: Optional[str] = None,
) -> dict:
    """Run the full L5 pipeline and return a single per-query result row.

    Args:
        threshold: If provided, use this value instead of the default 0.4
                   for IVF/HNSW switching.
        force_backend: "ivf" or "hnsw", overrides threshold-based selection.
    """
    q = query_vec

    # ── Signal analysis (consistent with chapter7 L5) ──
    ta = query_aware_lambda_and_beta(
        query, intent_classifier,
        temporal_config=TemporalConfig(),
        lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
    )
    intent_score: float = ta["intent_score"]
    intent_lam: float = ta["lambda"]
    temporal_score: float = ta["temporal_score"]
    beta: float = ta["beta"]

    # ── Adaptive topN ──
    adaptive_topN = compute_adaptive_topN(
        intent_score, temporal_score, topN_min=100, topN_max=300,
    )

    # ── Backend selection ──
    t0 = time.perf_counter()

    if force_backend == "ivf":
        routed = "ivf"
    elif force_backend == "hnsw":
        routed = "hnsw"
    else:
        effective_threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
        routed = "ivf" if intent_score < effective_threshold else "hnsw"

    if routed == "ivf":
        nprobe = compute_adaptive_nprobe(intent_score, temporal_score)
        ivf.set_nprobe(nprobe)
        res = ivf.search(q.reshape(1, -1), adaptive_topN)
    else:
        ef = compute_adaptive_ef(intent_score, temporal_score)
        hnsw.set_ef_search(ef)
        res = hnsw.search(q.reshape(1, -1), adaptive_topN)

    cands = res.indices[0]

    # ── Cluster-aware settings (consistent with chapter7 L4/L5) ──
    cluster_ids = getattr(ivf, "assignments", None)
    use_cluster = (routed == "ivf")
    gamma = 0.15 if use_cluster else 0.0
    c_ids = cluster_ids if use_cluster else None

    # ── Quality boost (consistent with chapter7 L5) ──
    top_score = float(res.scores[0][0]) if len(res.scores[0]) > 0 else 0.0
    l5_lam = float(intent_lam)
    if top_score < 0.5:
        quality_boost = (0.5 - top_score) * 1.0
        l5_lam = min(0.9, l5_lam + quality_boost)

    min_ratio = 0.60 + 0.25 * intent_score

    # ── MMR temporal reranking (consistent with chapter7 L5) ──
    selected = mmr_rerank_temporal(
        q, cands, emb, k=k, lambda_=l5_lam,
        freshness=freshness, beta=float(beta),
        min_score_ratio=min_ratio,
        cluster_ids=c_ids, cluster_penalty=gamma,
    )

    latency_ms = (time.perf_counter() - t0) * 1000

    # ── Metric computation ──
    rel, ild, f1, ws = compute_metrics(q, selected, emb)

    return {
        "query_text": query,
        "intent_score": round(intent_score, 4),
        "temporal_score": round(temporal_score, 4),
        "routed_backend": routed,
        "latency_ms": round(latency_ms, 3),
        "MeanRel": round(rel, 4),
        "ILD": round(ild, 4),
        "AvgF1": round(f1, 4),
        "lambda": round(l5_lam, 4),
        "beta": round(beta, 4),
        "topN": adaptive_topN,
        "gamma": gamma,
        "min_score_ratio": round(min_ratio, 4),
    }


# ═══════════════════════════════════════════════════════════════
#  Main sweep loop
# ═══════════════════════════════════════════════════════════════

def run_sweep(
    bundle: DatasetBundle,
    intent_classifier: IntentClassifier,
    *,
    k: int = 10,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Run full sweep across all threshold conditions + endpoint baselines."""
    if thresholds is None:
        thresholds = SWEEP_THRESHOLDS

    emb = bundle.passage_emb
    queries = bundle.queries
    query_embs = bundle.query_emb

    print(f"\n{'='*70}")
    print(f"  Backend Switch Threshold Sweep")
    print(f"  Dataset: {bundle.dataset_name} ({len(queries)} queries, k={k})")
    print(f"  Thresholds: {thresholds}")
    print(f"  Endpoints: {ENDPOINT_BASELINES}")
    print(f"{'='*70}")

    print("Building retrievers...")
    _brute, ivf, hnsw = build_all_retrievers(emb)

    # ── Compute all intent_scores with default threshold → define fixed ref split ──
    print("Computing intent scores for reference split...")
    intent_scores: list[float] = []
    for i, q_text in enumerate(queries):
        ta = query_aware_lambda_and_beta(
            q_text, intent_classifier,
            temporal_config=TemporalConfig(),
            lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
        )
        intent_scores.append(ta["intent_score"])
    intent_scores_arr = np.array(intent_scores)

    ambig_mask = intent_scores_arr < REF_SPLIT_THRESHOLD
    spec_mask = ~ambig_mask
    n_ambig = int(ambig_mask.sum())
    n_spec = int(spec_mask.sum())
    print(f"  Reference split (threshold={REF_SPLIT_THRESHOLD}): "
          f"{n_ambig} ambiguous, {n_spec} specific")

    # ── Build condition list ──
    conditions: list[dict] = []
    for t in thresholds:
        conditions.append({"label": f"t={t:.1f}", "threshold": t, "force": None})
    conditions.append({"label": "always_ivf", "threshold": None, "force": "ivf"})
    conditions.append({"label": "always_hnsw", "threshold": None, "force": "hnsw"})

    all_rows: list[dict] = []

    for cond in conditions:
        label = cond["label"]
        print(f"\n── Condition: {label} ──")
        t0_cond = time.perf_counter()

        for i in range(len(queries)):
            row = run_l5_single_query(
                queries[i], query_embs[i], emb,
                ivf, hnsw, intent_classifier,
                k=k,
                freshness=bundle.freshness,
                threshold=cond["threshold"],
                force_backend=cond["force"],
            )
            row["query_id"] = i
            row["threshold_setting"] = label
            row["ref_split"] = "ambiguous_ref" if ambig_mask[i] else "specific_ref"
            all_rows.append(row)

            if (i + 1) % 50 == 0 or i == len(queries) - 1:
                elapsed = time.perf_counter() - t0_cond
                print(f"  [{i+1}/{len(queries)}] {elapsed:.1f}s")

    return all_rows


# ═══════════════════════════════════════════════════════════════
#  Aggregation & output
# ═══════════════════════════════════════════════════════════════

def aggregate(rows: list[dict]) -> list[dict]:
    """Aggregate metrics by threshold_setting."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r["threshold_setting"]].append(r)

    summary: list[dict] = []
    for label, recs in buckets.items():
        n = len(recs)
        ivf_count = sum(1 for r in recs if r["routed_backend"] == "ivf")
        hnsw_count = n - ivf_count
        summary.append({
            "threshold_setting": label,
            "n_queries": n,
            "MeanRel": round(float(np.mean([r["MeanRel"] for r in recs])), 4),
            "ILD": round(float(np.mean([r["ILD"] for r in recs])), 4),
            "AvgF1": round(float(np.mean([r["AvgF1"] for r in recs])), 4),
            "latency_ms": round(float(np.mean([r["latency_ms"] for r in recs])), 3),
            "ivf_count": ivf_count,
            "ivf_frac": round(ivf_count / n, 4) if n else 0.0,
            "hnsw_count": hnsw_count,
            "hnsw_frac": round(hnsw_count / n, 4) if n else 0.0,
        })

    # Split by subset
    for label, recs in buckets.items():
        for split in ("ambiguous_ref", "specific_ref"):
            sub = [r for r in recs if r["ref_split"] == split]
            if not sub:
                continue
            summary.append({
                "threshold_setting": f"{label} [{split}]",
                "n_queries": len(sub),
                "MeanRel": round(float(np.mean([r["MeanRel"] for r in sub])), 4),
                "ILD": round(float(np.mean([r["ILD"] for r in sub])), 4),
                "AvgF1": round(float(np.mean([r["AvgF1"] for r in sub])), 4),
                "latency_ms": round(float(np.mean([r["latency_ms"] for r in sub])), 3),
                "ivf_count": sum(1 for r in sub if r["routed_backend"] == "ivf"),
                "ivf_frac": round(
                    sum(1 for r in sub if r["routed_backend"] == "ivf") / len(sub), 4,
                ),
                "hnsw_count": sum(1 for r in sub if r["routed_backend"] == "hnsw"),
                "hnsw_frac": round(
                    sum(1 for r in sub if r["routed_backend"] == "hnsw") / len(sub), 4,
                ),
            })

    return summary


def save_summary_csv(summary: list[dict], out_dir: Path) -> None:
    fields = [
        "threshold_setting", "n_queries",
        "MeanRel", "ILD", "AvgF1", "latency_ms",
        "ivf_count", "ivf_frac", "hnsw_count", "hnsw_frac",
    ]
    out = out_dir / "threshold_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    print(f"Saved: {out}")


def save_per_query_csv(rows: list[dict], out_dir: Path) -> None:
    fields = [
        "query_id", "query_text", "intent_score", "temporal_score",
        "threshold_setting", "routed_backend", "latency_ms",
        "MeanRel", "ILD", "AvgF1", "ref_split",
    ]
    out = out_dir / "threshold_per_query.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {out}")


def save_summary_tex(summary: list[dict], out_dir: Path) -> None:
    """Generate LaTeX table (overall rows only, excluding subset splits)."""
    overall = [s for s in summary if "[" not in s["threshold_setting"]]
    out = out_dir / "threshold_summary.tex"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Backend switch threshold sensitivity}\n")
        f.write("\\label{tab:threshold-sensitivity}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Setting & MeanRel & ILD & Avg.\\ F1 & "
                "Latency (ms) & IVF\\% & HNSW\\% \\\\\n")
        f.write("\\midrule\n")
        for s in overall:
            f.write(
                f"{s['threshold_setting']} & "
                f"{s['MeanRel']:.4f} & {s['ILD']:.4f} & {s['AvgF1']:.4f} & "
                f"{s['latency_ms']:.1f} & "
                f"{s['ivf_frac']*100:.1f} & {s['hnsw_frac']*100:.1f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════

def plot_results(summary: list[dict], out_dir: Path) -> None:
    """Generate three plots: AvgF1, Latency, and Routing distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Overall rows only (excluding subset splits)
    overall = [s for s in summary if "[" not in s["threshold_setting"]]
    labels = [s["threshold_setting"] for s in overall]
    avgf1 = [s["AvgF1"] for s in overall]
    latency = [s["latency_ms"] for s in overall]
    ivf_frac = [s["ivf_frac"] for s in overall]
    hnsw_frac = [s["hnsw_frac"] for s in overall]

    # ── AvgF1 ──
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#2196F3" if "t=" in l else "#FF9800" for l in labels]
    bars = ax.bar(labels, avgf1, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Avg F1")
    ax.set_title("Backend Switch Threshold vs. Avg F1")
    ax.set_ylim(min(avgf1) - 0.01, max(avgf1) + 0.01)
    for bar, val in zip(bars, avgf1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    # Highlight the default threshold
    for i, l in enumerate(labels):
        if l == f"t={DEFAULT_THRESHOLD:.1f}":
            bars[i].set_edgecolor("red")
            bars[i].set_linewidth(2)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_vs_avgf1.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'threshold_vs_avgf1.png'}")

    # ── Latency ──
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, latency, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title("Backend Switch Threshold vs. Latency")
    for bar, val in zip(bars, latency):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_vs_latency.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'threshold_vs_latency.png'}")

    # ── Routing 分布 (stacked bar) ──
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(labels))
    ax.bar(x, ivf_frac, label="IVF", color="#4CAF50", edgecolor="black", linewidth=0.5)
    ax.bar(x, hnsw_frac, bottom=ivf_frac, label="HNSW",
           color="#9C27B0", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction")
    ax.set_title("Backend Routing Distribution")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_vs_routing.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'threshold_vs_routing.png'}")


# ═══════════════════════════════════════════════════════════════
#  Significance tests
# ═══════════════════════════════════════════════════════════════

def run_significance_tests(rows: list[dict], out_dir: Path) -> None:
    """Wilcoxon signed-rank: t=0.4 vs best non-0.4 threshold by AvgF1。"""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("  scipy not installed, skipping significance tests")
        return

    # Bucket by threshold_setting, threshold conditions only (excluding always_*)
    by_cond: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_cond[r["threshold_setting"]][r["query_id"]] = r

    # Find the best non-0.4 condition
    default_label = f"t={DEFAULT_THRESHOLD:.1f}"
    threshold_labels = [l for l in by_cond if l.startswith("t=") and l != default_label]

    if not threshold_labels or default_label not in by_cond:
        print("  Cannot run significance tests (missing conditions)")
        return

    # Select best by overall AvgF1
    def _mean_f1(label: str) -> float:
        recs = list(by_cond[label].values())
        return float(np.mean([r["AvgF1"] for r in recs])) if recs else 0.0

    best_label = max(threshold_labels, key=_mean_f1)
    print(f"\n  Significance test: {default_label} vs {best_label} "
          f"(best non-default by AvgF1)")

    data_default = by_cond[default_label]
    data_best = by_cond[best_label]
    common = sorted(set(data_default.keys()) & set(data_best.keys()))

    if len(common) < 10:
        print(f"  Only {len(common)} paired queries, skipping")
        return

    metrics = ["MeanRel", "ILD", "AvgF1"]
    test_rows: list[dict] = []
    n_tests = len(metrics)

    for metric in metrics:
        vals_def = np.array([data_default[qid][metric] for qid in common])
        vals_best = np.array([data_best[qid][metric] for qid in common])
        diff = vals_def - vals_best

        if np.all(diff == 0):
            stat, p = 0.0, 1.0
        else:
            try:
                stat, p = wilcoxon(vals_def, vals_best, alternative="two-sided")
            except ValueError:
                stat, p = 0.0, 1.0

        p_corrected = min(p * n_tests, 1.0)

        test_rows.append({
            "Comparison": f"{default_label} vs {best_label}",
            "Metric": metric,
            "n_pairs": len(common),
            "mean_default": round(float(np.mean(vals_def)), 4),
            "mean_best": round(float(np.mean(vals_best)), 4),
            "delta": round(float(np.mean(diff)), 4),
            "p_value_raw": round(float(p), 6),
            "p_value_corrected": round(float(p_corrected), 6),
            "significant_0.05": bool(p_corrected < 0.05),
        })

    out = out_dir / "significance_tests.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
        w.writeheader()
        w.writerows(test_rows)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════
#  Auto summary
# ═══════════════════════════════════════════════════════════════

def save_notes(summary: list[dict], out_dir: Path) -> None:
    """Generate threshold_notes.md brief summary."""
    overall = [s for s in summary if "[" not in s["threshold_setting"]]
    if not overall:
        return

    best = max(overall, key=lambda s: s["AvgF1"])
    worst = min(overall, key=lambda s: s["AvgF1"])
    default_row = next(
        (s for s in overall if s["threshold_setting"] == f"t={DEFAULT_THRESHOLD:.1f}"),
        None,
    )

    out = out_dir / "threshold_notes.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Backend Switch Threshold Sensitivity — Summary\n\n")
        f.write(f"- **Best overall AvgF1**: {best['threshold_setting']} "
                f"({best['AvgF1']:.4f})\n")
        f.write(f"- **Worst overall AvgF1**: {worst['threshold_setting']} "
                f"({worst['AvgF1']:.4f})\n")
        if default_row:
            f.write(f"- **Default (t=0.4) AvgF1**: {default_row['AvgF1']:.4f}\n")
            f.write(f"- **Default latency**: {default_row['latency_ms']:.1f} ms\n")
        f.write(f"- **AvgF1 range across conditions**: "
                f"{worst['AvgF1']:.4f} – {best['AvgF1']:.4f} "
                f"(Δ = {best['AvgF1'] - worst['AvgF1']:.4f})\n\n")

        # Subset split
        for split in ("ambiguous_ref", "specific_ref"):
            sub_rows = [s for s in summary
                        if f"[{split}]" in s["threshold_setting"]]
            if not sub_rows:
                continue
            best_sub = max(sub_rows, key=lambda s: s["AvgF1"])
            f.write(f"### {split}\n")
            f.write(f"- Best AvgF1: {best_sub['threshold_setting']} "
                    f"({best_sub['AvgF1']:.4f})\n\n")

    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════
#  Main entry
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backend switch threshold sensitivity experiment",
    )
    parser.add_argument("--msmarco-dir", type=str, default=None,
                        help="Path to MS MARCO data directory")
    parser.add_argument("--improved-dir", type=str, default=None,
                        help="Path to improved corpus directory")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="outputs/backend_switch_threshold")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.msmarco_dir is None and args.improved_dir is None:
        parser.error("At least one of --msmarco-dir or --improved-dir is required")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load intent model
    print("Loading intent classifier...")
    intent_classifier = IntentClassifier.load("models/intent_v3")

    all_rows: list[dict] = []

    # Load datasets and run
    if args.improved_dir:
        bundle = load_improved(args.improved_dir, seed=args.seed)
        rows = run_sweep(bundle, intent_classifier, k=args.k)
        all_rows.extend(rows)

    if args.msmarco_dir:
        bundle = load_msmarco(args.msmarco_dir)
        rows = run_sweep(bundle, intent_classifier, k=args.k)
        all_rows.extend(rows)

    if not all_rows:
        print("No results. Check data directories.")
        return

    # Output
    summary = aggregate(all_rows)
    save_summary_csv(summary, out_dir)
    save_per_query_csv(all_rows, out_dir)
    save_summary_tex(summary, out_dir)
    save_notes(summary, out_dir)

    try:
        plot_results(summary, out_dir)
    except Exception as e:
        print(f"  Plotting failed: {e}")

    run_significance_tests(all_rows, out_dir)

    print(f"\n{'='*70}")
    print(f"  All outputs saved to: {out_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
