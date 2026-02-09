#!/usr/bin/env python3
"""
MS MARCO Passage Search Demo - FastAPI backend

Searches over 50k MS MARCO passages using intent-adaptive diversified reranking.
"""

import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

from diverse_search.index import build_retriever
from diverse_search.mmr import mmr_rerank_temporal
from diverse_search.metrics import avg_pairwise_cosine
from diverse_search.intent_model import IntentClassifier
from diverse_search.temporal import query_aware_lambda_and_beta, TemporalConfig

app = FastAPI(title="MS MARCO Passage Search Demo")

# Global resources
retriever = None
passages = None
embeddings = None
passage_sources = None  # list of "wiki"/"msmarco" per passage, or None
passage_freshness = None  # (nb,) float32 freshness scores, or None
intent_classifier = None
embedding_model = None


def extract_title(passage: str) -> str:
    """Extract a title from the first sentence of a passage."""
    # Split on sentence-ending punctuation followed by a space
    sentences = re.split(r'(?<=[.?!])\s+', passage.strip(), maxsplit=1)
    title = sentences[0] if sentences else passage
    # If first sentence is too short, use more text
    if len(title) < 10:
        title = passage[:80]
    # Truncate long titles
    if len(title) > 80:
        title = title[:77] + "..."
    return title


def extract_snippet(passage: str, query: str) -> str:
    """Extract a query-relevant snippet (2-3 best sentences) from the passage."""
    sentences = re.split(r'(?<=[.?!])\s+', passage.strip())
    if len(sentences) <= 2:
        snippet = passage
    else:
        # Score each sentence by word overlap with query
        query_words = set(query.lower().split())
        scored = []
        for i, sent in enumerate(sentences):
            sent_words = set(sent.lower().split())
            overlap = len(query_words & sent_words)
            # Small bonus for position (prefer earlier sentences)
            scored.append((overlap - i * 0.1, i, sent))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Take top 2-3 sentences, re-order by original position
        top = sorted(scored[:3], key=lambda x: x[1])
        snippet = " ".join(t[2] for t in top)
    # Truncate
    if len(snippet) > 250:
        snippet = snippet[:247] + "..."
    return snippet


class SearchResult(BaseModel):
    title: str
    snippet: str
    passage: str
    score: float
    rank: int
    passage_idx: int
    source: str = "web"
    freshness: float = 0.5


class SearchResponse(BaseModel):
    query: str
    intent_score: float
    lambda_value: float
    method: str
    results: List[SearchResult]
    relevance: float
    diversity: float
    f1: float
    temporal_score: float = 0.0
    temporal_explanation: str = ""
    beta: float = 0.0


def load_resources():
    """Load passages, embeddings, intent model, and sentence transformer.

    Checks data/mixed/ first (Wikipedia + MS MARCO), falls back to data/msmarco/.
    """
    global retriever, passages, embeddings, passage_sources, passage_freshness
    global intent_classifier, embedding_model

    print("Loading resources...")

    # Check data directories: prefer mixed corpus, fall back to msmarco
    mixed_dir = Path("data/mixed")
    msmarco_dir = Path("data/msmarco")

    if (mixed_dir / "passages.txt").exists() and (mixed_dir / "passage_embeddings.npy").exists():
        data_dir = mixed_dir
        passages_file = data_dir / "passages.txt"
        embeddings_file = data_dir / "passage_embeddings.npy"
        print(f"  Using mixed corpus from {data_dir}/")
    elif (msmarco_dir / "msmarco_passages.txt").exists() and (msmarco_dir / "passage_embeddings.npy").exists():
        data_dir = msmarco_dir
        passages_file = data_dir / "msmarco_passages.txt"
        embeddings_file = data_dir / "passage_embeddings.npy"
        print(f"  Using MS MARCO corpus from {data_dir}/")
    else:
        raise FileNotFoundError(
            "No corpus data found. Run one of:\n"
            "  python scripts/prepare_mixed_corpus.py   (recommended)\n"
            "  python scripts/prepare_msmarco.py"
        )

    with open(passages_file) as f:
        passages = [line.strip() for line in f]
    embeddings = np.load(embeddings_file)
    print(f"  Loaded {len(passages)} passages, dim={embeddings.shape[1]}")

    # Load source labels if available
    sources_file = data_dir / "passage_sources.txt"
    if sources_file.exists():
        with open(sources_file) as f:
            passage_sources = [line.strip() for line in f]
        n_wiki = sum(1 for s in passage_sources if s == "wiki")
        n_msmarco = sum(1 for s in passage_sources if s == "msmarco")
        print(f"  Sources: {n_wiki} wiki, {n_msmarco} msmarco")
    else:
        passage_sources = None

    # Load freshness scores if available
    freshness_file = data_dir / "passage_freshness.npy"
    if freshness_file.exists():
        passage_freshness = np.load(freshness_file)
        print(f"  Freshness scores loaded: shape={passage_freshness.shape}")
    else:
        passage_freshness = None
        print("  No freshness scores found (temporal reranking disabled)")

    # L2-normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    # Build retriever (brute force is fast enough for 50k)
    retriever = build_retriever(embeddings, backend="numpy")

    # Load intent classifier (try intent_v2 first, fall back to intent_chatgpt)
    for model_dir in ["models/intent_v3"]:
        model_path = Path(model_dir) / "intent_model.pkl"
        if model_path.exists():
            intent_classifier = IntentClassifier.load(model_dir)
            print(f"  Intent classifier loaded from {model_dir}")
            break
    else:
        intent_classifier = IntentClassifier()
        print("  No intent model found, using default classifier")

    # Load sentence transformer for query encoding
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("  Embedding model loaded")

    print("Resources loaded!")


def search(query: str, k: int = 10, method: str = "intent") -> SearchResponse:
    """Execute a search query with the specified method."""
    # Encode query
    query_vec = embedding_model.encode([query])[0]
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

    # Retrieve top-N candidates
    top_n = 200
    res = retriever.search(query_vec.reshape(1, -1), top_n)
    candidates = res.indices[0]
    scores = res.scores[0]

    # Determine lambda based on method
    temporal_score = 0.0
    temporal_explanation = ""
    beta = 0.0

    if method == "intent":
        # Combined intent + temporal analysis
        ta = query_aware_lambda_and_beta(
            query, intent_classifier,
            temporal_config=TemporalConfig(),
            lambda_min=0.5, lambda_max=0.9,
        )
        lam = ta["lambda"]
        intent_score = ta["intent_score"]
        temporal_score = ta["temporal_score"]
        temporal_explanation = ta["temporal_explanation"]
        beta = ta["beta"]
    elif method == "fixed_low":
        lam, intent_score = 0.5, 0.5
    elif method == "fixed_high":
        lam, intent_score = 0.8, 0.5
    elif method == "baseline":
        lam, intent_score = 1.0, 0.5
    else:
        lam, intent_score = 0.65, 0.5

    # 当候选整体相关度低时, 自动提高λ保住相关性
    # (语料库对该查询覆盖差时, 多样性只会拉入更多垃圾)
    top_candidate_score = float(scores[0]) if len(scores) > 0 else 0.0
    if top_candidate_score < 0.5 and method == "intent":
        # 候选质量差 → 将λ向0.85靠拢, 越差越靠
        quality_boost = (0.5 - top_candidate_score) * 1.0  # 最高 +0.5
        lam = min(0.9, lam + quality_boost)

    # MMR rerank with optional freshness boost
    selected = mmr_rerank_temporal(
        query_vec, candidates, embeddings, k=k, lambda_=float(lam),
        freshness=passage_freshness, beta=float(beta),
        min_score_ratio=0.85,
    )

    # Compute metrics
    selected_vecs = embeddings[selected]
    sims = np.dot(selected_vecs, query_vec)
    relevance = float(np.mean(sims))
    diversity = float(1.0 - avg_pairwise_cosine(selected_vecs))
    f1 = (
        2 * relevance * diversity / (relevance + diversity)
        if (relevance + diversity) > 0
        else 0
    )

    # Build results
    results = []
    for rank, idx in enumerate(selected):
        passage_text = passages[idx]
        # Determine source label
        if passage_sources is not None and idx < len(passage_sources):
            src = "wiki" if passage_sources[idx] == "wiki" else "web"
        else:
            src = "web"
        # Freshness for this passage
        fr = float(passage_freshness[idx]) if passage_freshness is not None else 0.5
        results.append(SearchResult(
            title=extract_title(passage_text),
            snippet=extract_snippet(passage_text, query),
            passage=passage_text,
            score=float(sims[rank]),
            rank=rank + 1,
            passage_idx=int(idx),
            source=src,
            freshness=fr,
        ))

    return SearchResponse(
        query=query,
        intent_score=float(intent_score),
        lambda_value=float(lam),
        method=method,
        results=results,
        relevance=relevance,
        diversity=diversity,
        f1=f1,
        temporal_score=float(temporal_score),
        temporal_explanation=temporal_explanation,
        beta=float(beta),
    )


@app.on_event("startup")
async def startup():
    load_resources()


@app.get("/")
async def root():
    return FileResponse("demo/index.html")


@app.get("/api/search")
async def api_search(
    q: str = Query(..., description="Search query"),
    k: int = Query(10, description="Number of results"),
    method: str = Query("intent", description="Method: intent, fixed_low, fixed_high, baseline"),
):
    return search(q, k, method)


@app.get("/api/compare")
async def api_compare(
    q: str = Query(..., description="Search query"),
    k: int = Query(10, description="Number of results"),
):
    """Compare three methods side by side."""
    return {
        "intent_adaptive": search(q, k, "intent"),
        "fixed_low": search(q, k, "fixed_low"),
        "fixed_high": search(q, k, "fixed_high"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
