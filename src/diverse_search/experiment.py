from __future__ import annotations

import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .index import BaseRetriever
from .metrics import (
    AggMetrics,
    avg_pairwise_cosine,
    max_pairwise_cosine,
    mean_cosine_to_query,
    recall_at_k,
)
from .mmr import mmr_rerank


def _compute_agg_metrics(
    *,
    xb: np.ndarray,
    queries: np.ndarray,
    selected_ids: np.ndarray,
    gt_neighbors: Optional[np.ndarray],
    k: int,
) -> AggMetrics:
    """Aggregate relevance/diversity (and recall if gt is provided)."""

    nq = int(queries.shape[0])
    rel_list: List[float] = []
    avg_cos_list: List[float] = []
    max_cos_list: List[float] = []

    for qi in range(nq):
        q = queries[qi]
        ids = selected_ids[qi]
        vecs = xb[ids]
        rel_list.append(mean_cosine_to_query(q, vecs))
        avg_cos_list.append(avg_pairwise_cosine(vecs))
        max_cos_list.append(max_pairwise_cosine(vecs))

    recall = None
    if gt_neighbors is not None:
        recall = recall_at_k(selected_ids, gt_neighbors, k)

    return AggMetrics(
        recall=recall,
        rel_mean_cos=float(np.mean(rel_list)),
        redundancy_avg_cos=float(np.mean(avg_cos_list)),
        redundancy_max_cos=float(np.mean(max_cos_list)),
    )


def evaluate_baseline_topk(
    retriever: BaseRetriever,
    xb: np.ndarray,
    queries: np.ndarray,
    *,
    k: int,
    topN: int,
    gt_neighbors: Optional[np.ndarray] = None,
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
    use_max_redundancy: bool = True,
) -> Dict:
    """MMR rerank on candidate topN."""

    t0 = time.time()
    res = retriever.search(queries, topN)
    cand = res.indices  # (nq, topN)

    nq = cand.shape[0]
    selected = np.empty((nq, k), dtype=np.int32)

    for qi in range(nq):
        selected[qi] = mmr_rerank(
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
        k=k,
    )

    out = {
        "method": "mmr",
        "lambda": float(lambda_),
        "k": k,
        "topN": topN,
        "search_time_sec": float(t1 - t0),
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
    gt_neighbors: Optional[np.ndarray] = None,
    use_max_redundancy: bool = True,
) -> pd.DataFrame:
    """Run baseline + MMR(lambda sweep), return aggregated results."""

    rows: List[Dict] = []
    rows.append(
        evaluate_baseline_topk(
            retriever,
            xb,
            queries,
            k=k,
            topN=topN,
            gt_neighbors=gt_neighbors,
        )
    )

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
                use_max_redundancy=use_max_redundancy,
            )
        )

    return pd.DataFrame(rows)
