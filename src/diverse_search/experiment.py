from __future__ import annotations

import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .adaptive_lambda import AdaptiveLambdaConfig, batch_adaptive_lambdas
from .index import BaseRetriever
from .metrics import (
    AggMetrics,
    avg_pairwise_cosine,
    max_pairwise_cosine,
    mean_cosine_to_query,
    recall_at_k,
)
from .diversify import maxmin_rerank, threshold_greedy_rerank
from .mmr import mmr_rerank, mmr_rerank_incremental


def _compute_agg_metrics(
    *,
    xb: np.ndarray,
    queries: np.ndarray,
    selected_ids: np.ndarray,
    gt_neighbors: Optional[np.ndarray],
    cluster_labels: Optional[np.ndarray],
    k: int,
) -> AggMetrics:
    """Aggregate relevance/diversity (and recall if gt is provided)."""

    nq = int(queries.shape[0])
    rel_list: List[float] = []
    avg_cos_list: List[float] = []
    max_cos_list: List[float] = []
    cov_unique_list: List[float] = []
    cov_ratio_list: List[float] = []

    for qi in range(nq):
        q = queries[qi]
        ids = selected_ids[qi]
        vecs = xb[ids]
        rel_list.append(mean_cosine_to_query(q, vecs))
        avg_cos_list.append(avg_pairwise_cosine(vecs))
        max_cos_list.append(max_pairwise_cosine(vecs))

        if cluster_labels is not None:
            labs = np.asarray(cluster_labels, dtype=np.int64)[ids]
            # count unique labels; treat -1 as "unknown" and ignore
            labs = labs[labs >= 0]
            u = float(np.unique(labs).size)
            cov_unique_list.append(u)
            cov_ratio_list.append(u / float(len(ids)) if len(ids) > 0 else 0.0)

    recall = None
    if gt_neighbors is not None:
        # If the database was sub-sampled, ignore GT ids outside [0, nb)
        recall = recall_at_k(selected_ids, gt_neighbors, k, max_index=int(xb.shape[0]))

    cov_unique = float(np.mean(cov_unique_list)) if cov_unique_list else None
    cov_ratio = float(np.mean(cov_ratio_list)) if cov_ratio_list else None

    return AggMetrics(
        recall=recall,
        rel_mean_cos=float(np.mean(rel_list)),
        redundancy_avg_cos=float(np.mean(avg_cos_list)),
        redundancy_max_cos=float(np.mean(max_cos_list)),
        coverage_unique=cov_unique,
        coverage_ratio=cov_ratio,
    )


def evaluate_baseline_topk(
    retriever: BaseRetriever,
    xb: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    topN: int,
    gt_neighbors: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
) -> Dict:
    """Baseline: return the first k items from candidate topN."""

    t0 = time.time()
    res = retriever.search(queries, topN)
    cand = res.indices  # (nq, topN)
    selected = cand[:, :k]
    t1 = time.time()

    agg = _compute_agg_metrics(
        xb=xb,
        queries=queries,
        selected_ids=selected,
        gt_neighbors=gt_neighbors,
        cluster_labels=cluster_labels,
        k=k,
    )

    out = {
        "method": "baseline",
        "lambda": 1.0,
        "k": k,
        "topN": topN,
        "search_time_sec": float(t1 - t0),
        **asdict(agg),
        "ild": float(agg.ild),
    }
    return out


def evaluate_mmr(
    retriever: BaseRetriever,
    xb: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    topN: int,
    lambda_: float,
    gt_neighbors: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
    use_max_redundancy: bool = True,
    impl: str = "full",
) -> Dict:
    """MMR rerank on candidate topN."""

    t0 = time.time()
    res = retriever.search(queries, topN)
    cand = res.indices  # (nq, topN)
    t_retrieve = time.time()

    nq = cand.shape[0]
    selected = np.empty((nq, k), dtype=np.int32)

    rerank_fn = mmr_rerank if impl == "full" else mmr_rerank_incremental

    for qi in range(nq):
        selected[qi] = rerank_fn(
            queries[qi],
            cand[qi],
            xb,
            k=k,
            lambda_=lambda_,
            use_max_redundancy=use_max_redundancy,
        )

    t1 = time.time()

    agg = _compute_agg_metrics(
        xb=xb,
        queries=queries,
        selected_ids=selected,
        gt_neighbors=gt_neighbors,
        cluster_labels=cluster_labels,
        k=k,
    )

    out = {
        "method": "mmr",
        "lambda": float(lambda_),
        "tau": None,
        "k": k,
        "topN": topN,
        "retrieve_time_sec": float(t_retrieve - t0),
        "rerank_time_sec": float(t1 - t_retrieve),
        "search_time_sec": float(t1 - t0),
        "impl": impl,
        **asdict(agg),
        "ild": float(agg.ild),
    }
    return out


def evaluate_mmr_adaptive(
    retriever: BaseRetriever,
    xb: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    topN: int,
    lambda_cfg: AdaptiveLambdaConfig,
    gt_neighbors: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
    use_max_redundancy: bool = True,
    impl: str = "full",
) -> Dict:
    """MMR with a query-adaptive lambda estimated from candidate scores."""

    t0 = time.time()
    res = retriever.search(queries, topN)
    cand = res.indices  # (nq, topN)
    cand_scores = res.scores  # (nq, topN)
    t_retrieve = time.time()

    nq = cand.shape[0]
    selected = np.empty((nq, k), dtype=np.int32)

    lambdas = batch_adaptive_lambdas(cand_scores, lambda_cfg)
    rerank_fn = mmr_rerank if impl == "full" else mmr_rerank_incremental

    for qi in range(nq):
        selected[qi] = rerank_fn(
            queries[qi],
            cand[qi],
            xb,
            k=k,
            lambda_=float(lambdas[qi]),
            use_max_redundancy=use_max_redundancy,
        )

    t1 = time.time()

    agg = _compute_agg_metrics(
        xb=xb,
        queries=queries,
        selected_ids=selected,
        gt_neighbors=gt_neighbors,
        cluster_labels=cluster_labels,
        k=k,
    )

    out = {
        "method": "mmr_adaptive",
        "lambda": float(lambdas.mean()),
        "lambda_min": float(lambdas.min()),
        "lambda_max": float(lambdas.max()),
        "lambda_std": float(lambdas.std()),
        "tau": None,
        "k": k,
        "topN": topN,
        "retrieve_time_sec": float(t_retrieve - t0),
        "rerank_time_sec": float(t1 - t_retrieve),
        "search_time_sec": float(t1 - t0),
        "impl": impl,
        **asdict(agg),
        "ild": float(agg.ild),
        "adaptive_lambda_min": float(lambda_cfg.lambda_min),
        "adaptive_lambda_max": float(lambda_cfg.lambda_max),
        "adaptive_lambda_mid": float(lambda_cfg.lambda_mid),
        "adaptive_lambda_strategy": str(lambda_cfg.strategy),
        "adaptive_lambda_temperature": float(lambda_cfg.temperature),
        "adaptive_lambda_entropy_power": float(lambda_cfg.entropy_power),
        "adaptive_lambda_topM": (
            int(lambda_cfg.topM) if lambda_cfg.topM is not None else None
        ),
        "adaptive_gap_k": int(lambda_cfg.gap_k),
        "adaptive_gap_low": float(lambda_cfg.gap_t_low),
        "adaptive_gap_high": float(lambda_cfg.gap_t_high),
    }
    return out


def evaluate_threshold(
    retriever: BaseRetriever,
    xb: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    topN: int,
    tau: float,
    gt_neighbors: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
) -> Dict:
    """Threshold-based greedy diversification."""

    t0 = time.time()
    res = retriever.search(queries, topN)
    cand = res.indices
    t_retrieve = time.time()

    nq = cand.shape[0]
    selected = np.empty((nq, k), dtype=np.int32)

    for qi in range(nq):
        selected[qi] = threshold_greedy_rerank(
            queries[qi],
            cand[qi],
            xb,
            k=k,
            tau=float(tau),
        )

    t1 = time.time()

    agg = _compute_agg_metrics(
        xb=xb,
        queries=queries,
        selected_ids=selected,
        gt_neighbors=gt_neighbors,
        cluster_labels=cluster_labels,
        k=k,
    )

    out = {
        "method": "threshold",
        "lambda": None,
        "tau": float(tau),
        "k": k,
        "topN": topN,
        "retrieve_time_sec": float(t_retrieve - t0),
        "rerank_time_sec": float(t1 - t_retrieve),
        "search_time_sec": float(t1 - t0),
        "impl": None,
        **asdict(agg),
        "ild": float(agg.ild),
    }
    return out


def evaluate_maxmin(
    retriever: BaseRetriever,
    xb: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    topN: int,
    gt_neighbors: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
) -> Dict:
    """MaxMin (farthest-first) diversification baseline."""

    t0 = time.time()
    res = retriever.search(queries, topN)
    cand = res.indices
    t_retrieve = time.time()

    nq = cand.shape[0]
    selected = np.empty((nq, k), dtype=np.int32)

    for qi in range(nq):
        selected[qi] = maxmin_rerank(
            queries[qi],
            cand[qi],
            xb,
            k=k,
        )

    t1 = time.time()

    agg = _compute_agg_metrics(
        xb=xb,
        queries=queries,
        selected_ids=selected,
        gt_neighbors=gt_neighbors,
        cluster_labels=cluster_labels,
        k=k,
    )

    out = {
        "method": "maxmin",
        "lambda": None,
        "tau": None,
        "k": k,
        "topN": topN,
        "retrieve_time_sec": float(t_retrieve - t0),
        "rerank_time_sec": float(t1 - t_retrieve),
        "search_time_sec": float(t1 - t0),
        "impl": None,
        **asdict(agg),
        "ild": float(agg.ild),
    }
    return out


def run_sweep(
    retriever: BaseRetriever,
    xb: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    topN: int,
    lambdas: Sequence[float],
    adaptive_lambda_cfg: Optional[AdaptiveLambdaConfig] = None,
    gt_neighbors: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
    use_max_redundancy: bool = True,
    mmr_impl: str = "full",
    methods: Optional[Sequence[str]] = None,
    taus: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Run baseline + diversified rerank methods.

    By default it runs: baseline + MMR(lambda sweep).
    You can add query-adaptive MMR by including "mmr_adaptive" in methods.
    You can add additional baselines by setting `methods` and providing `taus`.
    """

    if methods is None:
        methods = ["baseline", "mmr"]

    methods_set = {m.lower() for m in methods}
    rows: List[Dict] = []

    if "baseline" in methods_set:
        rows.append(
            evaluate_baseline_topk(
                retriever,
                xb,
                queries,
                k=k,
                topN=topN,
                gt_neighbors=gt_neighbors,
                cluster_labels=cluster_labels,
            )
        )

    if "mmr" in methods_set:
        for lam in lambdas:
            rows.append(
                evaluate_mmr(
                    retriever,
                    xb,
                    queries,
                    k=k,
                    topN=topN,
                    lambda_=float(lam),
                    gt_neighbors=gt_neighbors,
                    cluster_labels=cluster_labels,
                    use_max_redundancy=use_max_redundancy,
                    impl=mmr_impl,
                )
            )

    if "mmr_adaptive" in methods_set:
        cfg = adaptive_lambda_cfg or AdaptiveLambdaConfig()
        rows.append(
            evaluate_mmr_adaptive(
                retriever,
                xb,
                queries,
                k=k,
                topN=topN,
                lambda_cfg=cfg,
                gt_neighbors=gt_neighbors,
                cluster_labels=cluster_labels,
                use_max_redundancy=use_max_redundancy,
                impl=mmr_impl,
            )
        )

    if "threshold" in methods_set:
        if not taus:
            raise ValueError("methods includes 'threshold' but no taus were provided")
        for tau in taus:
            rows.append(
                evaluate_threshold(
                    retriever,
                    xb,
                    queries,
                    k=k,
                    topN=topN,
                    tau=float(tau),
                    gt_neighbors=gt_neighbors,
                    cluster_labels=cluster_labels,
                )
            )

    if "maxmin" in methods_set:
        rows.append(
            evaluate_maxmin(
                retriever,
                xb,
                queries,
                k=k,
                topN=topN,
                gt_neighbors=gt_neighbors,
                cluster_labels=cluster_labels,
            )
        )

    return pd.DataFrame(rows)
