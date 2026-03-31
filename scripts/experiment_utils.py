#!/usr/bin/env python3
"""
实验共享工具 — 数据加载、检索器构建、指标计算
==============================================

供 experiment_case_study.py 和 experiment_quantitative.py 共用。
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diverse_search.index import build_retriever
from diverse_search.metrics import mean_cosine_to_query, avg_pairwise_cosine
from diverse_search.temporal import generate_freshness_scores, TemporalConfig

# 从 experiment_ablation_index.py 复用
AMBIGUOUS_WORDS = [
    # 多义词（名人/地名/普通词义）
    "apple", "amazon", "mercury", "jaguar", "python", "crane", "bass",
    "bat", "bank", "spring", "palm", "mouse", "chip", "cell", "bug",
    "plant", "rock", "seal", "drill", "bridge", "key", "match", "table",
    "rose", "date", "fan", "light", "ring", "court", "net", "pitch",
    "staff", "tank", "mine", "organ", "stock", "current", "volume",
    # 抽象/宽泛概念
    "power", "energy", "force", "culture", "system", "design", "model",
    "theory", "pattern", "structure", "process", "value", "growth",
    "intelligence", "network", "evolution", "revolution", "movement",
    # 短泛化短语
    "best practices", "how things work", "health tips", "travel guide",
    "what is art", "meaning of life", "history of science",
    "types of music", "world religions", "climate change effects",
    "machine learning", "space exploration", "ancient civilizations",
    "modern architecture", "digital transformation", "renewable energy",
    "genetic engineering", "artificial intelligence", "quantum computing",
    "deep learning applications",
]


@dataclass
class DatasetBundle:
    """统一的数据集容器"""
    passages: list[str]
    passage_emb: np.ndarray          # (N, d) L2-normalized
    queries: list[str]
    query_emb: np.ndarray            # (nq, d) L2-normalized
    passage_titles: list[str] = field(default_factory=list)
    passage_sources: list[str] = field(default_factory=list)
    freshness: Optional[np.ndarray] = None   # (N,) or None
    dataset_name: str = ""


# ═══════════════════════════════════════════════════════════════
#  编码 & 归一化
# ═══════════════════════════════════════════════════════════════

def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / (norms + 1e-8)).astype(np.float32)


def encode_queries(texts: list[str]) -> np.ndarray:
    """用 SentenceTransformer 编码查询并 L2 归一化。"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return _l2_normalize(vecs)


# ═══════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════

def load_msmarco(data_dir: str = "data/msmarco") -> DatasetBundle:
    """加载 MS MARCO 数据集（需要已运行 prepare_msmarco.py）。"""
    d = Path(data_dir)

    # passages
    with open(d / "msmarco_passages.txt", encoding="utf-8") as f:
        passages = [line.strip() for line in f]

    # embeddings
    emb = np.load(d / "passage_embeddings.npy").astype(np.float32)
    emb = _l2_normalize(emb)

    # queries
    with open(d / "msmarco_queries.txt", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    query_emb = np.load(d / "query_embeddings.npy").astype(np.float32)
    query_emb = _l2_normalize(query_emb)

    # titles: first sentence of each passage
    titles = [_extract_title(p) for p in passages]

    # sources: MS MARCO 无 passage_sources.txt
    sources_file = d / "passage_sources.txt"
    if sources_file.exists():
        with open(sources_file, encoding="utf-8") as f:
            sources = [line.strip() for line in f]
    else:
        sources = []

    # freshness: MS MARCO 一般无
    freshness = None
    freshness_file = d / "passage_freshness.npy"
    if freshness_file.exists():
        freshness = np.load(freshness_file)

    print(f"[load_msmarco] {len(passages)} passages, {len(queries)} queries, dim={emb.shape[1]}")
    return DatasetBundle(
        passages=passages, passage_emb=emb,
        queries=queries, query_emb=query_emb,
        passage_titles=titles, passage_sources=sources,
        freshness=freshness, dataset_name="msmarco",
    )


def load_improved(
    data_dir: str = "data/improved",
    n_title_queries: int = 100,
    n_ambig_queries: int = 60,
    seed: int = 42,
) -> DatasetBundle:
    """加载 improved corpus，自动构建混合查询集。

    因为 improved corpus 没有预存查询嵌入，我们混合构造查询集：
    - 从 passage_titles 采样 → specific 查询
    - 从 AMBIGUOUS_WORDS 取 → ambiguous 查询
    然后用 SentenceTransformer 编码。
    """
    d = Path(data_dir)

    # passages
    with open(d / "passages.txt", encoding="utf-8") as f:
        passages = [line.strip() for line in f]

    # embeddings
    emb = np.load(d / "passage_embeddings.npy").astype(np.float32)
    emb = _l2_normalize(emb)

    # titles
    titles_file = d / "passage_titles.txt"
    if titles_file.exists():
        with open(titles_file, encoding="utf-8") as f:
            titles = [line.strip() for line in f]
    else:
        titles = [_extract_title(p) for p in passages]

    # sources
    sources_file = d / "passage_sources.txt"
    if sources_file.exists():
        with open(sources_file, encoding="utf-8") as f:
            sources = [line.strip() for line in f]
    else:
        sources = []

    # 构建混合查询集
    rng = np.random.default_rng(seed)

    # specific: 从 titles 采样（去重且长度适中的 title）
    unique_titles = list({t for t in titles if 5 < len(t) < 80})
    n_title = min(n_title_queries, len(unique_titles))
    title_idx = rng.choice(len(unique_titles), size=n_title, replace=False)
    specific_queries = [unique_titles[i] for i in title_idx]

    # ambiguous: 从 AMBIGUOUS_WORDS 取
    n_ambig = min(n_ambig_queries, len(AMBIGUOUS_WORDS))
    ambig_idx = rng.choice(len(AMBIGUOUS_WORDS), size=n_ambig, replace=False)
    ambig_queries = [AMBIGUOUS_WORDS[i] for i in ambig_idx]

    queries = ambig_queries + specific_queries
    print(f"[load_improved] {len(passages)} passages, "
          f"queries: {n_ambig} ambiguous + {n_title} specific = {len(queries)}")

    # 编码查询
    print("  Encoding queries with SentenceTransformer...")
    query_emb = encode_queries(queries)

    # freshness
    freshness = None
    if sources:
        freshness = generate_freshness_scores(sources, TemporalConfig())

    return DatasetBundle(
        passages=passages, passage_emb=emb,
        queries=queries, query_emb=query_emb,
        passage_titles=titles, passage_sources=sources,
        freshness=freshness, dataset_name="improved",
    )


# ═══════════════════════════════════════════════════════════════
#  检索器构建
# ═══════════════════════════════════════════════════════════════

def build_all_retrievers(emb: np.ndarray, *, nlist: int = 128, hnsw_m: int = 16,
                         ef_construction: int = 200):
    """构建三种检索器：brute force, IVF, HNSW。"""
    print("  Building brute-force retriever...")
    brute = build_retriever(emb, backend="numpy")

    print(f"  Building IVF retriever (nlist={nlist})...")
    ivf = build_retriever(emb, backend="numpy_ivf", nlist=nlist, nprobe=16)

    print(f"  Building HNSW retriever (M={hnsw_m}, ef_construction={ef_construction})...")
    hnsw = build_retriever(emb, backend="numpy_hnsw",
                           hnsw_m=hnsw_m, ef_construction=ef_construction, ef_search=64)

    return brute, ivf, hnsw


# ═══════════════════════════════════════════════════════════════
#  指标计算
# ═══════════════════════════════════════════════════════════════

def compute_metrics(
    query_vec: np.ndarray,
    selected: np.ndarray,
    emb: np.ndarray,
) -> tuple[float, float, float, float]:
    """计算 (relevance, ILD, F1, weighted_sum)。

    Returns:
        (rel, ild, f1, ws) 四元组
    """
    if len(selected) < 2:
        return 0.0, 0.0, 0.0, 0.0

    selected_vecs = emb[selected]
    rel = float(mean_cosine_to_query(query_vec, selected_vecs))
    avg_pw = float(avg_pairwise_cosine(selected_vecs))
    ild = 1.0 - avg_pw  # intra-list diversity
    f1 = 2 * rel * ild / (rel + ild) if (rel + ild) > 0 else 0.0
    ws = 0.5 * rel + 0.5 * ild  # weighted sum
    return rel, ild, f1, ws


def filter_low_quality(
    selected: np.ndarray,
    query_vec: np.ndarray,
    emb: np.ndarray,
    *,
    score_floor: float = 0.30,
    min_keep: int = 2,
) -> np.ndarray:
    """绝对分数地板：移除 sim < score_floor 的结果，至少保留 min_keep 个。

    Args:
        selected: (k,) 已选 passage 索引
        query_vec: (d,) 查询向量
        emb: (N, d) 全量嵌入矩阵
        score_floor: 最低相似度阈值
        min_keep: 至少保留的结果数

    Returns:
        过滤后的 passage 索引数组
    """
    if len(selected) == 0:
        return selected
    sims = emb[selected] @ query_vec
    mask = sims >= score_floor
    if mask.sum() >= min_keep:
        return selected[mask]
    # 不足 min_keep 个：取 top min_keep
    top_idx = np.argsort(-sims)[:min_keep]
    return selected[top_idx]


# ═══════════════════════════════════════════════════════════════
#  辅助
# ═══════════════════════════════════════════════════════════════

def _extract_title(passage: str) -> str:
    """从 passage 的第一句提取 title。"""
    sentences = re.split(r'(?<=[.?!])\s+', passage.strip(), maxsplit=1)
    title = sentences[0] if sentences else passage
    if len(title) < 10:
        title = passage[:80]
    if len(title) > 80:
        title = title[:77] + "..."
    return title
