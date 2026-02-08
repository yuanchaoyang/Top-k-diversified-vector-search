from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .adaptive_lambda import AdaptiveLambdaConfig, adaptive_lambda_from_scores
from .datasets import download_dataset, load_hdf5
from .index import build_retriever
from .mmr import mmr_rerank_incremental
from .diversify import threshold_greedy_rerank, maxmin_rerank
from .metrics import mean_cosine_to_query, avg_pairwise_cosine, max_pairwise_cosine


def _print_one(method_name: str, q: np.ndarray, xb: np.ndarray, ids: np.ndarray) -> None:
    vecs = xb[ids]
    rel = mean_cosine_to_query(q, vecs)
    red_avg = avg_pairwise_cosine(vecs)
    red_max = max_pairwise_cosine(vecs)
    ild = 1.0 - red_avg

    print(f"\n=== {method_name} ===")
    for r, idx in enumerate(ids.tolist(), start=1):
        sim = float(xb[idx] @ q)
        print(f"{r:02d}. id={idx:<8d}  sim={sim:.4f}")

    print(f"rel_mean_cos={rel:.4f}  ild={ild:.4f}  redundancy_avg_cos={red_avg:.4f}  redundancy_max_cos={red_max:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="glove-100-angular")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "faiss"])
    ap.add_argument("--index-type", default="hnsw", choices=["hnsw", "flat"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--topN", type=int, default=200)
    ap.add_argument("--max-train", type=int, default=50000)
    ap.add_argument("--max-test", type=int, default=1000)

    ap.add_argument("--query-id", type=int, default=0)
    ap.add_argument("--random-query", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--methods", nargs="+", default=["baseline", "mmr", "mmr_adaptive", "threshold", "maxmin"])
    ap.add_argument("--lambdas", default="0.6")  # comma-separated
    ap.add_argument("--tau", type=float, default=0.8)

    # Adaptive lambda for MMR (used when methods includes mmr_adaptive)
    ap.add_argument("--adaptive-lambda-range", default="0.6,0.95",
                    help="min,max for query-adaptive lambda (comma-separated)")
    ap.add_argument("--adaptive-lambda-strategy", default="gap_piecewise",
                    choices=["gap_piecewise", "entropy"])
    ap.add_argument("--adaptive-gap-k", type=int, default=10)
    ap.add_argument("--adaptive-gap-low", type=float, default=0.02)
    ap.add_argument("--adaptive-gap-high", type=float, default=0.08)
    ap.add_argument("--adaptive-gap-lambda-mid", type=float, default=0.8)
    ap.add_argument("--adaptive-lambda-temperature", type=float, default=0.08)
    ap.add_argument("--adaptive-lambda-entropy-power", type=float, default=1.0)
    ap.add_argument("--adaptive-lambda-topM", type=int, default=50)

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    h5_path = data_dir / f"{args.dataset}.hdf5"

    download_dataset(args.dataset, h5_path)
    ds = load_hdf5(h5_path, max_train=args.max_train, max_test=args.max_test, normalize=True)

    xb = ds.train
    Q = ds.test

    rng = np.random.default_rng(args.seed)
    if args.random_query:
        qid = int(rng.integers(0, Q.shape[0]))
    else:
        qid = int(args.query_id)
        if not (0 <= qid < Q.shape[0]):
            raise ValueError(f"query-id out of range: {qid} (0..{Q.shape[0]-1})")

    q = Q[qid]
    print(f"Using query_id={qid}, dim={q.shape[0]}, dataset={args.dataset}")

    retriever = build_retriever(xb, backend=args.backend, index_type=args.index_type)
    res = retriever.search(q, args.topN)
    cand = res.indices[0].astype(np.int64)
    cand_scores = res.scores[0].astype(np.float32)

    k = int(args.k)
    topN = int(args.topN)

    methods = [m.lower() for m in args.methods]
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip() != ""]

    adaptive_cfg = None
    if "mmr_adaptive" in methods:
        adaptive_range = [float(x) for x in args.adaptive_lambda_range.split(",") if x.strip() != ""]
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

    if "baseline" in methods:
        _print_one("baseline", q, xb, cand[:k].astype(np.int32))

    if "mmr" in methods:
        for lam in lambdas:
            ids = mmr_rerank_incremental(q, cand, xb, k=k, lambda_=lam, use_max_redundancy=True)
            _print_one(f"mmr (lambda={lam})", q, xb, ids)

    if "mmr_adaptive" in methods:
        if adaptive_cfg is None:
            raise ValueError("mmr_adaptive requested but adaptive config is missing")
        lam = adaptive_lambda_from_scores(cand_scores, adaptive_cfg)
        ids = mmr_rerank_incremental(q, cand, xb, k=k, lambda_=lam, use_max_redundancy=True)
        _print_one(f"mmr_adaptive (lambda={lam:.3f}, strategy={adaptive_cfg.strategy})", q, xb, ids)

    if "threshold" in methods:
        ids = threshold_greedy_rerank(q, cand, xb, k=k, tau=float(args.tau))
        _print_one(f"threshold (tau={args.tau})", q, xb, ids)

    if "maxmin" in methods:
        ids = maxmin_rerank(q, cand, xb, k=k)
        _print_one("maxmin", q, xb, ids)


if __name__ == "__main__":
    main()
