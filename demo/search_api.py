#!/usr/bin/env python3
"""
MS MARCO Passage Search Demo - FastAPI backend

Searches over 50k MS MARCO passages using intent-adaptive diversified reranking.
"""

import re
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional

from diverse_search.index import build_retriever, compute_adaptive_nprobe, compute_adaptive_topN
from diverse_search.mmr import mmr_rerank_temporal
from diverse_search.metrics import avg_pairwise_cosine
from diverse_search.intent_model import IntentClassifier
from diverse_search.temporal import query_aware_lambda_and_beta, TemporalConfig

app = FastAPI(title="MS MARCO Passage Search Demo")

# Global resources
retriever = None
brute_retriever = None  # 暴力搜索 retriever，用于对比实验
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
    nprobe: int = 0
    topN: int = 0


def load_resources():
    """Load passages, embeddings, intent model, and sentence transformer.

    Checks data/improved/ first (best quality), then data/mixed/, then data/msmarco/.
    """
    global retriever, brute_retriever, passages, embeddings, passage_sources, passage_freshness
    global intent_classifier, embedding_model

    print("Loading resources...")

    # Check data directories: prefer improved corpus, then mixed, then msmarco
    improved_dir = Path("data/improved")
    mixed_dir = Path("data/mixed")
    msmarco_dir = Path("data/msmarco")

    if (improved_dir / "passages.txt").exists() and (improved_dir / "passage_embeddings.npy").exists():
        data_dir = improved_dir
        passages_file = data_dir / "passages.txt"
        embeddings_file = data_dir / "passage_embeddings.npy"
        print(f"  Using improved corpus from {data_dir}/")
    elif (mixed_dir / "passages.txt").exists() and (mixed_dir / "passage_embeddings.npy").exists():
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
            "  python scripts/build_improved_corpus.py --output data/improved  (recommended)\n"
            "  python scripts/prepare_mixed_corpus.py\n"
            "  python scripts/prepare_msmarco.py"
        )

    with open(passages_file, encoding='utf-8') as f:
        passages = [line.strip() for line in f]
    embeddings = np.load(embeddings_file)
    print(f"  Loaded {len(passages)} passages, dim={embeddings.shape[1]}")

    # Load source labels if available
    sources_file = data_dir / "passage_sources.txt"
    if sources_file.exists():
        with open(sources_file, encoding='utf-8') as f:
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

    # Build retrievers — 纯 NumPy IVF (主检索) + brute-force (对比用)
    brute_retriever = build_retriever(embeddings, backend="numpy")
    print("  Brute-force retriever built (for comparison)")
    retriever = build_retriever(
        embeddings, backend="numpy_ivf",
        nlist=128, nprobe=16,
    )
    print(f"  NumPy IVF index built: nlist=128, default nprobe=16")

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

    # 1) 先计算 intent 信号（用于自适应 nprobe 和 lambda）
    temporal_score = 0.0
    temporal_explanation = ""
    beta = 0.0
    current_nprobe = 0

    if method == "intent":
        # Combined intent + temporal analysis
        ta = query_aware_lambda_and_beta(
            query, intent_classifier,
            temporal_config=TemporalConfig(),
            lambda_min=0.40, lambda_max=0.9,
        )
        lam = ta["lambda"]
        intent_score = ta["intent_score"]
        temporal_score = ta["temporal_score"]
        temporal_explanation = ta["temporal_explanation"]
        beta = ta["beta"]

        # 2) 设置自适应 nprobe 和 topN
        if hasattr(retriever, "set_nprobe"):
            current_nprobe = compute_adaptive_nprobe(intent_score, temporal_score)
            retriever.set_nprobe(current_nprobe)
        top_n = compute_adaptive_topN(intent_score, temporal_score)
    elif method == "fixed_low":
        lam, intent_score = 0.5, 0.5
        top_n = 200
    elif method == "fixed_high":
        lam, intent_score = 0.8, 0.5
        top_n = 200
    elif method == "baseline":
        lam, intent_score = 1.0, 0.5
        top_n = 200
    else:
        lam, intent_score = 0.65, 0.5
        top_n = 200

    # 读取实际 nprobe（非 intent 方法或非 IVF 时为 0）
    if current_nprobe == 0 and hasattr(retriever, "get_nprobe"):
        current_nprobe = retriever.get_nprobe() or 0

    # 3) 检索 top-N 候选
    res = retriever.search(query_vec.reshape(1, -1), top_n)
    candidates = res.indices[0]
    scores = res.scores[0]

    # 当候选整体相关度低时, 自动提高λ保住相关性
    # (语料库对该查询覆盖差时, 多样性只会拉入更多垃圾)
    top_candidate_score = float(scores[0]) if len(scores) > 0 else 0.0
    if top_candidate_score < 0.5 and method == "intent":
        # 候选质量差 → 将λ向0.85靠拢, 越差越靠
        quality_boost = (0.5 - top_candidate_score) * 1.0  # 最高 +0.5
        lam = min(0.9, lam + quality_boost)

    # 歧义查询放宽相关度门槛，保留多样候选；明确查询收紧门槛，过滤噪声
    min_ratio = 0.60 + 0.25 * intent_score  # 0.60 ~ 0.85

    # 4) MMR rerank with optional freshness boost
    selected = mmr_rerank_temporal(
        query_vec, candidates, embeddings, k=k, lambda_=float(lam),
        freshness=passage_freshness, beta=float(beta),
        min_score_ratio=min_ratio,
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
        nprobe=current_nprobe,
        topN=top_n,
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


class IndexCompareResponse(BaseModel):
    """IVF vs Brute-force 对比结果"""
    query: str
    intent_score: float
    temporal_score: float
    nprobe: int

    # 候选集对比 (top-N retrieval)
    topN: int
    brute_time_ms: float
    ivf_time_ms: float
    speedup: float
    candidate_overlap: int          # 两者共有的候选数
    candidate_overlap_pct: float    # 重叠率 %
    brute_only_count: int           # 仅暴力搜索找到的
    ivf_only_count: int             # 仅 IVF 找到的

    # top-N 候选的平均相关度
    brute_candidate_avg_score: float
    ivf_candidate_avg_score: float

    # 最终 rerank 后对比 (top-k)
    k: int
    brute_relevance: float
    ivf_relevance: float
    brute_diversity: float
    ivf_diversity: float
    brute_f1: float
    ivf_f1: float
    rerank_overlap: int             # 最终结果中共有的条目数
    rerank_overlap_pct: float

    # 具体差异示例：IVF 丢失但暴力搜索找到的高分候选
    missed_examples: List[dict]


@app.get("/api/compare_index")
async def api_compare_index(
    q: str = Query(..., description="Search query"),
    k: int = Query(10, description="Number of final results"),
    topN: int = Query(200, description="Candidate pool size"),
):
    """对比 IVF 索引与暴力搜索的检索结果差异。"""
    # 编码查询
    query_vec = embedding_model.encode([q])[0]
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    q_2d = query_vec.reshape(1, -1)

    # 计算 intent 信号
    ta = query_aware_lambda_and_beta(
        q, intent_classifier,
        temporal_config=TemporalConfig(),
        lambda_min=0.40, lambda_max=0.9,
    )
    lam = ta["lambda"]
    intent_score = ta["intent_score"]
    temporal_score = ta["temporal_score"]
    beta = ta["beta"]

    # 自适应 nprobe
    current_nprobe = compute_adaptive_nprobe(intent_score, temporal_score)
    if hasattr(retriever, "set_nprobe"):
        retriever.set_nprobe(current_nprobe)

    # --- 暴力搜索 ---
    t0 = time.perf_counter()
    brute_res = brute_retriever.search(q_2d, topN)
    brute_time = (time.perf_counter() - t0) * 1000
    brute_ids = set(brute_res.indices[0].tolist())
    brute_scores = brute_res.scores[0]

    # --- IVF 搜索 ---
    t0 = time.perf_counter()
    ivf_res = retriever.search(q_2d, topN)
    ivf_time = (time.perf_counter() - t0) * 1000
    ivf_ids = set(ivf_res.indices[0].tolist())
    ivf_scores = ivf_res.scores[0]

    # 候选集对比
    overlap = brute_ids & ivf_ids
    brute_only = brute_ids - ivf_ids
    ivf_only = ivf_ids - brute_ids

    # quality boost (复用主搜索逻辑)
    top_score = float(brute_scores[0]) if len(brute_scores) > 0 else 0.0
    if top_score < 0.5:
        quality_boost = (0.5 - top_score) * 1.0
        lam = min(0.9, lam + quality_boost)
    min_ratio = 0.60 + 0.25 * intent_score

    # --- MMR rerank 两组候选 ---
    brute_selected = mmr_rerank_temporal(
        query_vec, brute_res.indices[0], embeddings, k=k, lambda_=float(lam),
        freshness=passage_freshness, beta=float(beta), min_score_ratio=min_ratio,
    )
    ivf_selected = mmr_rerank_temporal(
        query_vec, ivf_res.indices[0], embeddings, k=k, lambda_=float(lam),
        freshness=passage_freshness, beta=float(beta), min_score_ratio=min_ratio,
    )

    # rerank 后指标
    def _metrics(selected):
        vecs = embeddings[selected]
        sims = np.dot(vecs, query_vec)
        rel = float(np.mean(sims))
        div = float(1.0 - avg_pairwise_cosine(vecs))
        f1 = 2 * rel * div / (rel + div) if (rel + div) > 0 else 0.0
        return rel, div, f1

    b_rel, b_div, b_f1 = _metrics(brute_selected)
    i_rel, i_div, i_f1 = _metrics(ivf_selected)

    rerank_overlap_set = set(brute_selected.tolist()) & set(ivf_selected.tolist())

    # IVF 丢失的高分候选示例
    brute_score_map = {int(brute_res.indices[0][i]): float(brute_scores[i])
                       for i in range(len(brute_scores))}
    missed = sorted(
        [(idx, brute_score_map.get(idx, 0.0)) for idx in brute_only],
        key=lambda x: x[1], reverse=True,
    )[:5]
    missed_examples = []
    for idx, score in missed:
        if idx >= 0 and idx < len(passages):
            missed_examples.append({
                "passage_idx": idx,
                "score": round(score, 4),
                "title": extract_title(passages[idx]),
            })

    return IndexCompareResponse(
        query=q,
        intent_score=round(intent_score, 4),
        temporal_score=round(temporal_score, 4),
        nprobe=current_nprobe,
        topN=topN,
        brute_time_ms=round(brute_time, 3),
        ivf_time_ms=round(ivf_time, 3),
        speedup=round(brute_time / ivf_time, 2) if ivf_time > 0 else 0.0,
        candidate_overlap=len(overlap),
        candidate_overlap_pct=round(100.0 * len(overlap) / topN, 1),
        brute_only_count=len(brute_only),
        ivf_only_count=len(ivf_only),
        brute_candidate_avg_score=round(float(np.mean(brute_scores)), 4),
        ivf_candidate_avg_score=round(float(np.mean(ivf_scores)), 4),
        k=k,
        brute_relevance=round(b_rel, 4),
        ivf_relevance=round(i_rel, 4),
        brute_diversity=round(b_div, 4),
        ivf_diversity=round(i_div, 4),
        brute_f1=round(b_f1, 4),
        ivf_f1=round(i_f1, 4),
        rerank_overlap=len(rerank_overlap_set),
        rerank_overlap_pct=round(100.0 * len(rerank_overlap_set) / k, 1),
        missed_examples=missed_examples,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
