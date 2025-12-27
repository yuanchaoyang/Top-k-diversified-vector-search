#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

# Make src/ importable when running as a script
import sys

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJ_ROOT / "src"))

from diverse_search.datasets import load_hdf5
from diverse_search.experiment import run_sweep
from diverse_search.index import build_retriever


def parse_floats(values: List[str]) -> List[float]:
    return [float(v) for v in values]


def make_plot(df, out_path: Path) -> None:
    """Plot Recall@k vs ILD (or RelMeanCos vs ILD if recall is unavailable)."""

    # Baseline row
    base = df[df["method"] == "baseline"].iloc[0]

    has_recall = df["recall"].notnull().any()

    plt.figure()
    if has_recall:
        y = df["recall"]
        ylabel = "Recall@k"
    else:
        y = df["rel_mean_cos"]
        ylabel = "Mean cosine(q, selected)"

    x = df["ild"]

    # annotate baseline and mmr points
    for _, row in df.iterrows():
        label = "baseline" if row["method"] == "baseline" else f"mmr λ={row['lambda']:.2f}"
        plt.scatter([row["ild"]], [row["recall"] if has_recall else row["rel_mean_cos"]])
        plt.annotate(label, (row["ild"], row["recall"] if has_recall else row["rel_mean_cos"]), fontsize=8)

    plt.xlabel("ILD = 1 - AvgPairwiseCos (higher = more diverse)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Top-k diversified vector search experiment")
    parser.add_argument("--dataset", required=True, help="Path to .hdf5 dataset")
    parser.add_argument("--backend", default="numpy", choices=["numpy", "faiss"], help="Candidate retrieval backend")
    parser.add_argument("--index", default="hnsw", choices=["flat", "hnsw"], help="FAISS index type (if backend=faiss)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--topN", type=int, default=100)
    parser.add_argument("--lambdas", nargs="+", default=["0.2", "0.4", "0.6", "0.8"], help="MMR lambda list")
    parser.add_argument("--max_train", type=int, default=200000, help="Cap train size for quick runs")
    parser.add_argument("--max_test", type=int, default=1000, help="Cap test size for quick runs")
    parser.add_argument("--out_dir", default="outputs/run", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_hdf5(args.dataset, max_train=args.max_train, max_test=args.max_test, normalize=True)

    # If we sub-sample train, ground truth neighbor ids might exceed max_train.
    # We'll filter them to keep recall meaningful.
    gt = ds.neighbors
    if gt is not None:
        mask = gt < ds.train.shape[0]
        # replace out-of-range ids with -1
        gt = gt.copy()
        gt[~mask] = -1

    retriever = build_retriever(
        ds.train,
        backend=args.backend,
        index_type=args.index,
    )

    lambdas = parse_floats(args.lambdas)
    df = run_sweep(
        retriever,
        ds.train,
        ds.test,
        k=args.k,
        topN=args.topN,
        lambdas=lambdas,
        gt_neighbors=gt,
    )

    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    plot_path = out_dir / "tradeoff.png"
    make_plot(df, plot_path)

    print("\n=== Summary ===")
    print(df[["method", "lambda", "recall", "rel_mean_cos", "redundancy_avg_cos", "ild", "redundancy_max_cos", "search_time_sec"]])
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
