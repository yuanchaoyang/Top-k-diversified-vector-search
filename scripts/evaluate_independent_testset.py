#!/usr/bin/env python3
"""
Independent test-set evaluation for the deployed intent model.

Usage:
    source .venv/bin/activate
    PYTHONPATH=src python scripts/evaluate_independent_testset.py \
        --test-set data/intent_test_set.json \
        --model-dir models/intent_v3 \
        --out-dir outputs/intent_eval_independent
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from diverse_search.intent_model import IntentClassifier

BAND_EDGES = {"ambiguous": (0.0, 0.40), "mid": (0.40, 0.70), "specific": (0.70, 1.01)}


def score_to_band(score: float) -> str:
    for band, (lo, hi) in BAND_EDGES.items():
        if lo <= score < hi:
            return band
    return "specific"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set",   default="data/intent_test_set.json")
    parser.add_argument("--model-dir",  default="models/intent_v3")
    parser.add_argument("--out-dir",    default="outputs/intent_eval_independent")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load test set ──────────────────────────────────────────────────────
    with open(args.test_set, encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test queries from {args.test_set}")

    # ── load deployed model ────────────────────────────────────────────────
    clf = IntentClassifier.load(args.model_dir)
    print(f"Model loaded from {args.model_dir}")
    print(f"  use_embeddings={clf.use_embeddings}  "
          f"label_cache_size={len(clf.label_cache)}")

    # ── run predictions ────────────────────────────────────────────────────
    rows = []
    for item in test_data:
        query      = item["query"]
        gold       = float(item["gold_score"])
        gold_band  = item["band"]

        pred       = clf.predict(query)
        pred_band  = score_to_band(pred)
        abs_err    = abs(pred - gold)

        rows.append({
            "query":           query,
            "gold_score":      gold,
            "predicted_score": round(pred, 4),
            "abs_error":       round(abs_err, 4),
            "gold_band":       gold_band,
            "predicted_band":  pred_band,
        })

    golds = np.array([r["gold_score"]      for r in rows])
    preds = np.array([r["predicted_score"] for r in rows])
    errs  = np.abs(golds - preds)

    # ── regression metrics ─────────────────────────────────────────────────
    mae  = float(np.mean(errs))
    rmse = float(np.sqrt(np.mean(errs ** 2)))

    ss_res = np.sum((golds - preds) ** 2)
    ss_tot = np.sum((golds - np.mean(golds)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Pearson
    corr_matrix = np.corrcoef(golds, preds)
    pearson = float(corr_matrix[0, 1])

    # Spearman
    def _rank(arr):
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1)
        return ranks
    spearman = float(np.corrcoef(_rank(golds), _rank(preds))[0, 1])

    # ── band accuracy ──────────────────────────────────────────────────────
    bands = ["ambiguous", "mid", "specific"]
    band_correct = {b: 0 for b in bands}
    band_total   = {b: 0 for b in bands}
    confusion    = {g: {p: 0 for p in bands} for g in bands}

    for r in rows:
        gb, pb = r["gold_band"], r["predicted_band"]
        band_total[gb]  += 1
        confusion[gb][pb] += 1
        if gb == pb:
            band_correct[gb] += 1

    overall_band_acc = sum(band_correct.values()) / len(rows)

    # ── top mismatches ─────────────────────────────────────────────────────
    rows_sorted = sorted(rows, key=lambda r: r["abs_error"], reverse=True)
    top10_mismatches = rows_sorted[:10]

    # ── save CSV ───────────────────────────────────────────────────────────
    csv_path = out_dir / "predictions.csv"
    fields = ["query", "gold_score", "predicted_score", "abs_error",
              "gold_band", "predicted_band"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {csv_path}")

    # ── print + save markdown report ───────────────────────────────────────
    report_lines = []
    def ln(s=""): report_lines.append(s)

    ln("# Intent Model — Independent Test Set Evaluation")
    ln()
    ln(f"**Model checkpoint:** `{args.model_dir}`  ")
    ln(f"**Test set:** `{args.test_set}` ({len(rows)} queries)  ")
    ln()

    ln("## 1. Overall Regression Metrics")
    ln()
    ln(f"| Metric | Value |")
    ln(f"|--------|-------|")
    ln(f"| MAE    | {mae:.4f} |")
    ln(f"| RMSE   | {rmse:.4f} |")
    ln(f"| R²     | {r2:.4f} |")
    ln(f"| Pearson r | {pearson:.4f} |")
    ln(f"| Spearman ρ | {spearman:.4f} |")
    ln()

    ln("## 2. Coarse Band Accuracy")
    ln()
    ln(f"Overall band accuracy: **{overall_band_acc:.1%}** ({sum(band_correct.values())}/{len(rows)})")
    ln()
    ln("| Band | Total | Correct | Accuracy |")
    ln("|------|-------|---------|----------|")
    for b in bands:
        acc = band_correct[b] / band_total[b] if band_total[b] > 0 else 0.0
        ln(f"| {b} | {band_total[b]} | {band_correct[b]} | {acc:.1%} |")
    ln()

    ln("## 3. Confusion Summary (Gold → Predicted)")
    ln()
    header = "| Gold \\ Pred | " + " | ".join(bands) + " |"
    sep    = "|" + "---|" * (len(bands) + 1)
    ln(header)
    ln(sep)
    for gb in bands:
        row_str = f"| **{gb}** | " + " | ".join(str(confusion[gb][pb]) for pb in bands) + " |"
        ln(row_str)
    ln()

    ln("## 4. Top-10 Largest Mismatches")
    ln()
    ln("| Query | Gold | Pred | Error | Gold Band | Pred Band |")
    ln("|-------|------|------|-------|-----------|-----------|")
    for r in top10_mismatches:
        ln(f"| {r['query']} | {r['gold_score']:.2f} | {r['predicted_score']:.2f} "
           f"| {r['abs_error']:.3f} | {r['gold_band']} | {r['predicted_band']} |")
    ln()

    ln("## 5. Interpretation")
    ln()

    # auto-generate brief interpretation
    if mae < 0.08:
        mae_qual = "excellent (MAE < 0.08)"
    elif mae < 0.12:
        mae_qual = "good (MAE < 0.12)"
    elif mae < 0.18:
        mae_qual = "moderate (MAE < 0.18)"
    else:
        mae_qual = "weak (MAE ≥ 0.18)"

    if pearson > 0.90:
        corr_qual = "very strong linear correlation"
    elif pearson > 0.75:
        corr_qual = "strong linear correlation"
    elif pearson > 0.55:
        corr_qual = "moderate linear correlation"
    else:
        corr_qual = "weak linear correlation"

    ln(f"- **Regression accuracy is {mae_qual}** — mean absolute error across {len(rows)} "
       f"independent queries is {mae:.3f} score units.")
    ln(f"- **{corr_qual}** between gold and predicted scores (Pearson r = {pearson:.3f}, "
       f"Spearman ρ = {spearman:.3f}), indicating the model captures the "
       f"ambiguous-to-specific gradient.")
    ln(f"- **Coarse band accuracy is {overall_band_acc:.1%}**, with the largest band-level "
       f"errors in: " + ", ".join(
           b for b in bands if band_total[b] > 0
           and (band_correct[b] / band_total[b]) < 0.75
       ) or "none — all bands ≥ 75%")
    ln(f"- The label cache explains exact matches for short single-word queries; "
       f"multi-word queries exercise the full embedding → regressor path.")
    ln(f"- R² = {r2:.3f} {'indicates the model explains most variance in the gold scores.' if r2 > 0.7 else 'suggests some systematic bias remains.'}")

    report = "\n".join(report_lines)

    report_path = out_dir / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved: {report_path}")

    # ── print to stdout ────────────────────────────────────────────────────
    print()
    print(report)


if __name__ == "__main__":
    main()
