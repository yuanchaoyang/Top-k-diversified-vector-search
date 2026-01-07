from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer

from .index import build_retriever
from .mmr import mmr_rerank_incremental
from .diversify import threshold_greedy_rerank, maxmin_rerank
from .metrics import mean_cosine_to_query, avg_pairwise_cosine
from .cluster import KMeansConfig, kmeans_cluster_labels


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize."""
    x = x.astype(np.float32, copy=False)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / (n + eps)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive text query -> diversified top-k demo.")
    ap.add_argument("--corpus", default="data/corpus_words.txt",
                    help="Text file, one item per line (the retrieval corpus).")
    ap.add_argument("--embeddings", default="data/corpus_words_embeddings.npy",
                    help="Numpy cache for corpus embeddings (created if missing).")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small",
                    help="SentenceTransformer model name.")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "faiss"],
                    help="Retriever backend (numpy or faiss).")

    ap.add_argument("--k", type=int, default=10, help="Final diversified top-k size.")
    ap.add_argument("--topN", type=int, default=200, help="Candidate pool size before reranking.")

    ap.add_argument("--methods", nargs="+",
                    default=["baseline", "mmr", "threshold", "maxmin"],
                    help="Any of: baseline, mmr, threshold, maxmin")

    # IMPORTANT: can't use args.lambda in Python because 'lambda' is a keyword.
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.6,
                    help="MMR trade-off lambda in [0,1]. (higher -> more relevance)")
    ap.add_argument("--tau", type=float, default=0.8,
                    help="Threshold for greedy diversity filter (higher -> stricter)")

    # Coverage (pseudo-topics via clustering)
    ap.add_argument("--clusters", type=int, default=50,
                    help="Number of KMeans clusters for coverage metric. Set 0 to disable.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for clustering.")

    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    emb_path = Path(args.embeddings)

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}\n"
            f"Create it first (one item per line), e.g. data/corpus_words.txt"
        )

    texts = [line.strip() for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(texts) == 0:
        raise ValueError("Corpus is empty.")

    # Clamp sizes to corpus length
    topN = min(int(args.topN), len(texts))
    k = min(int(args.k), topN)

    model = SentenceTransformer(args.model)

    # Build / load corpus embeddings
    if emb_path.exists():
        xb = np.load(emb_path)
        xb = np.asarray(xb, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[0] != len(texts):
            raise ValueError(
                f"Embedding cache shape mismatch: {xb.shape}, corpus lines={len(texts)}.\n"
                f"Delete {emb_path} and rerun to rebuild."
            )
    else:
        is_e5 = "e5" in args.model.lower()
        passages = [("passage: " + t) if is_e5 else t for t in texts]
        xb = model.encode(passages, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        xb = np.asarray(xb, dtype=np.float32)
        xb = l2_normalize(xb)
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, xb)
        print(f"[cache] Saved embeddings: {emb_path} (shape={xb.shape})")

    xb = l2_normalize(xb)
    retriever = build_retriever(xb, backend=args.backend)

    methods = [m.lower() for m in args.methods]
    is_e5 = "e5" in args.model.lower()

    # Build cluster labels once (for coverage)
    labels = None
    if int(args.clusters) > 0:
        try:
            cfg = KMeansConfig(n_clusters=int(args.clusters), seed=int(args.seed))
            labels = kmeans_cluster_labels(xb, cfg)
            print(f"[coverage] KMeans clusters={args.clusters} ready.")
        except Exception as e:
            labels = None
            print(f"[coverage] Disabled (clustering failed): {e}")

    print("\nInteractive demo:")
    print("  - Type a query like: book, library, novel, machine learning, travel\n"
          "  - Type 'exit' to quit\n")

    def show_with_summary(name: str, q: np.ndarray, ids: np.ndarray) -> None:
        vecs = xb[ids]
        rel_mean = float(mean_cosine_to_query(q, vecs))
        redundancy_avg = float(avg_pairwise_cosine(vecs))
        ild = float(1.0 - redundancy_avg)

        if labels is not None:
            cov_unique = int(np.unique(labels[ids]).size)
            cov_ratio = float(cov_unique / len(ids))
            cov_str = f"coverage={cov_unique}/{len(ids)} ({cov_ratio:.2f})"
        else:
            cov_str = "coverage=NA"

        print(f"\n--- {name} ---")
        for r, idx in enumerate(ids.tolist(), start=1):
            sim = float(xb[idx] @ q)
            print(f"{r:02d}. {texts[idx]}  (sim={sim:.4f})")

        # one-line summary at the end
        print(f"Summary: rel_mean={rel_mean:.4f} | ILD={ild:.4f} | {cov_str}")

    while True:
        q_text = input("query> ").strip()
        if q_text.lower() in {"exit", "quit"}:
            break
        if not q_text:
            continue

        q_in = ("query: " + q_text) if is_e5 else q_text
        q = model.encode([q_in], convert_to_numpy=True).astype(np.float32)[0]
        q = l2_normalize(q)

        res = retriever.search(q, topN)
        cand = res.indices[0].astype(np.int64)

        if "baseline" in methods:
            show_with_summary("baseline", q, cand[:k].astype(np.int32))

        if "mmr" in methods:
            lam = float(args.lambda_)
            ids = mmr_rerank_incremental(q, cand, xb, k=k, lambda_=lam)
            show_with_summary(f"mmr (lambda={lam})", q, ids)

        if "threshold" in methods:
            tau = float(args.tau)
            ids = threshold_greedy_rerank(q, cand, xb, k=k, tau=tau)
            show_with_summary(f"threshold (tau={tau})", q, ids)

        if "maxmin" in methods:
            ids = maxmin_rerank(q, cand, xb, k=k)
            show_with_summary("maxmin", q, ids)

        print("")


if __name__ == "__main__":
    main()
