from __future__ import annotations

"""CLI entry point to run experiments and generate the key trade-off plot.

Typical usage :

  python3 -m topk_diverse_vector_search.run_experiments \
    --dataset glove-100-angular \
    --backend faiss \
    --index-type hnsw \
    --k 10 --topN 200 \
    --max-train 50000 --max-test 500 \
    --methods baseline mmr mmr_adaptive threshold maxmin \
    --lambdas 0.8 \
    --adaptive-lambda-range 0.6,0.95 \
    --taus 0.7,0.8,0.9 \
    --clusters 100 \
    --out-dir outputs

What you get:
  - outputs/results.csv  (table you can directly paste into thesis)
  - outputs/tradeoff.png (one slide / one figure summary)
"""

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .adaptive_lambda import AdaptiveLambdaConfig, batch_adaptive_lambdas
from .cluster import KMeansConfig, kmeans_cluster_labels
from .datasets import DATASET_URLS, download_dataset, load_hdf5
from .experiment import run_sweep
from .index import build_retriever


def _parse_csv_floats(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split(",")]


def _parse_methods(ms: Sequence[str]) -> List[str]:
    return [m.strip().lower() for m in ms if m.strip()]


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        type=str,
        default="glove-100-angular",
        help=f"Dataset name in {sorted(DATASET_URLS.keys())} or a local .hdf5 path",
    )
    p.add_argument("--data-dir", type=str, default="data", help="Where to store downloaded datasets")

    p.add_argument("--backend", type=str, default="numpy", choices=["numpy", "faiss"])
    p.add_argument("--index-type", type=str, default="hnsw", choices=["flat", "hnsw"])
    p.add_argument("--hnsw-m", type=int, default=32)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--ef-search", type=int, default=64)

    p.add_argument("--k", type=int, default=10)
    p.add_argument("--topN", type=int, default=200)
    p.add_argument("--max-train", type=int, default=50000)
    p.add_argument("--max-test", type=int, default=500)

    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["baseline", "mmr"],
        help="Any of: baseline mmr mmr_adaptive threshold maxmin",
    )
    p.add_argument(
        "--lambdas",
        type=str,
        default="0,0.2,0.4,0.6,0.8,1.0",
        help="Comma separated list, used for MMR sweep",
    )
    p.add_argument(
        "--adaptive-lambda-range",
        type=str,
        default="0.6,0.95",
        help="min,max for query-adaptive lambda (used when methods includes mmr_adaptive)",
    )
    p.add_argument(
        "--adaptive-lambda-strategy",
        type=str,
        default="hybrid",
        choices=["gap_piecewise", "gap_linear", "entropy", "hybrid", "candidate_diversity"],
        help="Strategy: gap_piecewise, gap_linear, entropy, hybrid (recommended), or candidate_diversity",
    )
    p.add_argument(
        "--adaptive-lambda-temperature",
        type=float,
        default=0.08,
        help="Softmax temperature for ambiguity estimation (lower -> sharper)",
    )
    p.add_argument(
        "--adaptive-lambda-entropy-power",
        type=float,
        default=1.0,
        help="Exponent to re-shape entropy before mapping to lambda (>1 pushes to extremes)",
    )
    p.add_argument(
        "--adaptive-lambda-topM",
        type=int,
        default=50,
        help="Use only the topM candidate scores when estimating entropy (<=0 disables)",
    )
    p.add_argument(
        "--adaptive-gap-k",
        type=int,
        default=10,
        help="gap = s1 - s_k for gap_piecewise strategy",
    )
    p.add_argument(
        "--adaptive-gap-low",
        type=float,
        default=0.02,
        help="If gap <= this, use lambda_min (gap_piecewise)",
    )
    p.add_argument(
        "--adaptive-gap-high",
        type=float,
        default=0.08,
        help="If gap >= this, use lambda_max (gap_piecewise)",
    )
    p.add_argument(
        "--adaptive-gap-lambda-mid",
        type=float,
        default=0.8,
        help="Lambda to use when gap is between low/high (gap_piecewise)",
    )
    p.add_argument(
        "--taus",
        type=str,
        default="",
        help="Comma separated list, used for threshold sweep",
    )

    p.add_argument(
        "--mmr-impl",
        type=str,
        default="incremental",
        choices=["full", "incremental"],
        help="MMR implementation. incremental is usually faster",
    )

    p.add_argument(
        "--clusters",
        type=int,
        default=0,
        help="If >0, run k-means clustering on xb and report coverage metrics",
    )

    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument(
        "--plot-adaptive-lambda",
        action="store_true",
        help="If set and methods includes mmr_adaptive, save a histogram of per-query lambdas",
    )

    args = p.parse_args()

    # 1) load dataset
    ds_arg = args.dataset
    if ds_arg.endswith(".hdf5") or ds_arg.endswith(".h5"):
        path = Path(ds_arg)
    else:
        path = Path(args.data_dir) / f"{ds_arg}.hdf5"
        download_dataset(ds_arg, path)

    ds = load_hdf5(path, max_train=int(args.max_train), max_test=int(args.max_test), normalize=True)
    xb = ds.train
    q = ds.test

    # 2) build retriever
    retriever = build_retriever(
        xb,
        backend=args.backend,
        index_type=args.index_type,
        hnsw_m=int(args.hnsw_m),
        ef_construction=int(args.ef_construction),
        ef_search=int(args.ef_search),
    )

    # 3) optional clustering for coverage metrics
    cluster_labels = None
    if int(args.clusters) > 0:
        cluster_labels = kmeans_cluster_labels(
            xb, KMeansConfig(n_clusters=int(args.clusters), seed=42)
        )

    # 4) run sweep
    methods = _parse_methods(args.methods)
    lambdas = _parse_csv_floats(args.lambdas)
    taus = _parse_csv_floats(args.taus)
    adaptive_range = _parse_csv_floats(args.adaptive_lambda_range)
    if len(adaptive_range) != 2:
        raise ValueError("--adaptive-lambda-range must have two values: min,max")
    adaptive_cfg = AdaptiveLambdaConfig(
        lambda_min=float(adaptive_range[0]),
        lambda_max=float(adaptive_range[1]),
        lambda_mid=float(args.adaptive_gap_lambda_mid),
        strategy=str(args.adaptive_lambda_strategy),
        temperature=float(args.adaptive_lambda_temperature),
        entropy_power=float(args.adaptive_lambda_entropy_power),
        topM=int(args.adaptive_lambda_topM) if int(args.adaptive_lambda_topM) > 0 else None,
        gap_k=int(args.adaptive_gap_k),
        gap_t_low=float(args.adaptive_gap_low),
        gap_t_high=float(args.adaptive_gap_high),
    )

    df = run_sweep(
        retriever,
        xb,
        q,
        k=int(args.k),
        topN=int(args.topN),
        lambdas=lambdas,
        taus=taus if taus else None,
        methods=methods,
        gt_neighbors=ds.neighbors,
        cluster_labels=cluster_labels,
        mmr_impl="full" if args.mmr_impl == "full" else "incremental",
        adaptive_lambda_cfg=adaptive_cfg,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "results.csv"
    df.to_csv(out_csv, index=False)

    # 5) print a quick summary
    with np.printoptions(precision=4, suppress=True):
        print("\n=== Results (sorted by relevance) ===")
        print(df.sort_values("rel_mean_cos", ascending=False).to_string(index=False))
        print(f"\nSaved: {out_csv}")

    # 6) plot trade-off
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print("matplotlib not installed; skipping plot.")
            return

        fig = plt.figure()
        ax = plt.gca()

        for method in sorted(set(df["method"].tolist())):
            sub = df[df["method"] == method]
            ax.scatter(sub["ild"], sub["rel_mean_cos"], label=method)

        ax.set_xlabel("Intra-List Diversity (ILD = 1 - avg pairwise cosine)")
        ax.set_ylabel("Mean cosine similarity to query (relevance)")
        ax.set_title("Relevance–Diversity Trade-off")
        ax.legend()

        out_png = out_dir / "tradeoff.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        print(f"Saved: {out_png}")

        if args.plot_adaptive_lambda and "mmr_adaptive" in methods:
            res = retriever.search(q, int(args.topN))
            lambdas = batch_adaptive_lambdas(res.scores, adaptive_cfg)

            fig = plt.figure()
            ax = plt.gca()
            ax.hist(lambdas, bins=30, color="#2a7f62", alpha=0.85, edgecolor="black", linewidth=0.5)
            ax.axvline(float(np.mean(lambdas)), color="#333333", linestyle="--", linewidth=1.0, label="mean")
            ax.set_xlabel("Adaptive lambda")
            ax.set_ylabel("Query count")
            ax.set_title("Adaptive Lambda Distribution")
            ax.legend()

            out_hist = out_dir / "adaptive_lambda_hist.png"
            plt.tight_layout()
            plt.savefig(out_hist, dpi=200)
            print(f"Saved: {out_hist}")


if __name__ == "__main__":
    main()
