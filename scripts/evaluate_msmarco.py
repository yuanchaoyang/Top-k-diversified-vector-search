#!/usr/bin/env python3
"""
在 MS MARCO 数据上评估 intent model vs 传统方法.

与 evaluate_intent_model.py 类似，但使用真实的文本查询.

用法:
  # 先准备数据
  python scripts/prepare_msmarco.py --n-passages 50000 --n-queries 1000

  # 再运行评估
  PYTHONPATH=src python scripts/evaluate_msmarco.py \
    --data-dir data/msmarco --k 10 --topN 100 \
    --out-dir outputs/msmarco_eval
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


def load_msmarco_data(data_dir: Path):
    """加载预处理好的 MS MARCO 数据."""
    with open(data_dir / "msmarco_passages.txt") as f:
        passages = [line.strip() for line in f if line.strip()]
    with open(data_dir / "msmarco_queries.txt") as f:
        queries = [line.strip() for line in f if line.strip()]

    passage_emb = np.load(data_dir / "passage_embeddings.npy")
    query_emb = np.load(data_dir / "query_embeddings.npy")

    # 确保归一化
    def normalize(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return (x / (norms + 1e-8)).astype(np.float32)

    passage_emb = normalize(passage_emb)
    query_emb = normalize(query_emb)

    return passages, queries, passage_emb, query_emb


def evaluate_query(
    query_vec: np.ndarray,
    selected: np.ndarray,
    passage_emb: np.ndarray,
) -> Dict[str, float]:
    """评估一个查询的搜索结果."""
    sel_vecs = passage_emb[selected]
    rel = mean_cosine_to_query(query_vec, sel_vecs)
    avg_cos = avg_pairwise_cosine(sel_vecs)
    return {"relevance": rel, "diversity": 1.0 - avg_cos}


def run_evaluation(
    queries: List[str],
    query_emb: np.ndarray,
    passages: List[str],
    passage_emb: np.ndarray,
    intent_classifier: IntentClassifier,
    k: int = 10,
    topN: int = 100,
    lambda_min: float = 0.5,
    lambda_max: float = 0.85,
    max_queries: int = 500,
) -> pd.DataFrame:
    """在 MS MARCO 查询上评估所有方法."""

    retriever = build_retriever(passage_emb, backend="numpy")
    n_queries = min(len(queries), max_queries)

    rows = []

    for qi in range(n_queries):
        if (qi + 1) % 50 == 0:
            print(f"  Query {qi + 1}/{n_queries}...")

        query_text = queries[qi]
        qvec = query_emb[qi]

        # 检索候选
        res = retriever.search(qvec.reshape(1, -1), topN)
        candidates = res.indices[0]
        scores = res.scores[0]

        base = {"query_idx": qi, "query_text": query_text}

        # --- Baseline ---
        m = evaluate_query(qvec, candidates[:k], passage_emb)
        rows.append({**base, "method": "baseline", "lambda": 1.0, **m})

        # --- Fixed lambda ---
        for lam in [0.5, 0.6, 0.7, 0.8]:
            sel = mmr_rerank_incremental(qvec, candidates, passage_emb, k=k, lambda_=lam)
            m = evaluate_query(qvec, sel, passage_emb)
            rows.append({**base, "method": f"fixed_{lam}", "lambda": lam, **m})

        # --- Adaptive hybrid ---
        cfg = AdaptiveLambdaConfig(
            strategy="hybrid", lambda_min=lambda_min, lambda_max=lambda_max,
        )
        lam = adaptive_lambda_from_scores(scores, cfg)
        sel = mmr_rerank_incremental(qvec, candidates, passage_emb, k=k, lambda_=lam)
        m = evaluate_query(qvec, sel, passage_emb)
        rows.append({**base, "method": "adaptive_hybrid", "lambda": lam, **m})

        # --- Intent model ---
        # 用完整查询文本分析意图
        lam_intent, intent_score = intent_classifier.predict_lambda(
            query_text, lambda_min, lambda_max,
        )
        sel = mmr_rerank_incremental(
            qvec, candidates, passage_emb, k=k, lambda_=lam_intent,
        )
        m = evaluate_query(qvec, sel, passage_emb)
        rows.append({
            **base, "method": "intent_model",
            "lambda": lam_intent, "intent_score": intent_score, **m,
        })

    return pd.DataFrame(rows)


def print_case_studies(
    df: pd.DataFrame,
    passages: List[str],
    passage_emb: np.ndarray,
    query_emb: np.ndarray,
    queries: List[str],
    retriever,
    intent_classifier: IntentClassifier,
    k: int = 10,
    topN: int = 100,
    n_cases: int = 10,
):
    """打印具体案例."""
    # 选一些有代表性的 query (intent 分数不同的)
    intent_df = df[df["method"] == "intent_model"].copy()
    if "intent_score" not in intent_df.columns:
        return

    # 取最模糊的5个 + 最明确的5个
    ambiguous = intent_df.nsmallest(n_cases // 2, "intent_score")
    specific = intent_df.nlargest(n_cases // 2, "intent_score")
    cases = pd.concat([ambiguous, specific])

    print("\n  === Case Studies ===")
    for _, row in cases.iterrows():
        qi = int(row["query_idx"])
        query_text = row["query_text"]
        intent_score = row.get("intent_score", 0.5)

        # 获取搜索结果
        qvec = query_emb[qi]
        res = retriever.search(qvec.reshape(1, -1), topN)
        candidates = res.indices[0]

        # Fixed λ=0.7
        sel_fixed = mmr_rerank_incremental(
            qvec, candidates, passage_emb, k=min(k, 3), lambda_=0.7,
        )
        # Intent model
        lam, _ = intent_classifier.predict_lambda(query_text, 0.5, 0.85)
        sel_intent = mmr_rerank_incremental(
            qvec, candidates, passage_emb, k=min(k, 3), lambda_=lam,
        )

        tag = "ambiguous" if intent_score < 0.4 else "specific"
        print(f"\n  [{tag}] \"{query_text}\" (intent={intent_score:.2f}, λ={lam:.2f})")
        print(f"    Fixed λ=0.7 results:")
        for idx in sel_fixed:
            print(f"      - {passages[idx][:80]}...")
        print(f"    Intent model results:")
        for idx in sel_intent:
            print(f"      - {passages[idx][:80]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/msmarco"))
    parser.add_argument("--model-dir", default="models/intent_chatgpt")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--topN", type=int, default=100)
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--lambda-min", type=float, default=0.5)
    parser.add_argument("--lambda-max", type=float, default=0.85)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/msmarco_eval"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MS MARCO EVALUATION")
    print("=" * 70)

    # 加载数据
    print("\n[1/4] Loading data...")
    passages, queries, passage_emb, query_emb = load_msmarco_data(args.data_dir)
    print(f"  Passages: {len(passages)}, Queries: {len(queries)}, Dim: {passage_emb.shape[1]}")

    # 加载模型
    print("\n[2/4] Loading intent model...")
    classifier = IntentClassifier.load(args.model_dir)

    # 运行评估
    print(f"\n[3/4] Running evaluation (k={args.k}, topN={args.topN}, max_queries={args.max_queries})...")
    df = run_evaluation(
        queries, query_emb, passages, passage_emb, classifier,
        k=args.k, topN=args.topN,
        lambda_min=args.lambda_min, lambda_max=args.lambda_max,
        max_queries=args.max_queries,
    )

    # 保存
    df.to_csv(args.out_dir / "msmarco_results.csv", index=False)

    # 汇总
    print(f"\n[4/4] Results...")
    agg = df.groupby("method").agg(
        relevance=("relevance", "mean"),
        diversity=("diversity", "mean"),
        lambda_mean=("lambda", "mean"),
    ).reset_index()
    agg["combined"] = 0.5 * agg["relevance"] + 0.5 * agg["diversity"]
    agg = agg.sort_values("combined", ascending=False)

    print("\n  === Overall Results ===")
    print(agg.to_string(index=False, float_format="%.4f"))
    agg.to_csv(args.out_dir / "msmarco_summary.csv", index=False)

    # 案例
    retriever = build_retriever(passage_emb, backend="numpy")
    print_case_studies(
        df, passages, passage_emb, query_emb, queries,
        retriever, classifier, k=args.k, topN=args.topN,
    )

    # 画图
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {
            "baseline": "#999", "fixed_0.5": "#3498db", "fixed_0.6": "#2980b9",
            "fixed_0.7": "#2471a3", "fixed_0.8": "#1a5276",
            "adaptive_hybrid": "#2ecc71", "intent_model": "#e74c3c",
        }
        markers = {
            "baseline": "s", "adaptive_hybrid": "^", "intent_model": "*",
        }
        for _, row in agg.iterrows():
            m = row["method"]
            ax.scatter(
                row["diversity"], row["relevance"],
                c=colors.get(m, "#777"),
                marker=markers.get(m, "o"),
                s=200 if m in ("intent_model", "adaptive_hybrid") else 80,
                label=m, edgecolors="black", linewidths=0.5, zorder=3,
            )
        ax.set_xlabel("Diversity (ILD)")
        ax.set_ylabel("Relevance")
        ax.set_title("MS MARCO: Relevance-Diversity Trade-off")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_dir / "msmarco_tradeoff.png", dpi=200)
        print(f"\n  Saved: {args.out_dir / 'msmarco_tradeoff.png'}")
    except ImportError:
        pass

    print(f"\n  All outputs: {args.out_dir}/")


if __name__ == "__main__":
    main()
