#!/usr/bin/env python3
"""
构建混合语料库: Wikipedia + MS MARCO (Bing).

两个来源各有优势:
  - FineWiki (HuggingFaceFW/finewiki en): 2025年8月 Wikipedia dump
    百科知识, 概念覆盖全面, 比 wikimedia/wikipedia 更新两年
  - MS MARCO (Bing 搜索结果): 真实搜索场景的段落
    已有数据, 直接复用 data/msmarco/msmarco_passages.txt

同时生成 passage_freshness.npy: 每条passage的新鲜度分数 (0=旧, 1=新),
供时间敏感查询的 recency-aware reranking 使用.

Wikipedia 用 streaming 模式, 无需下载全量数据.

依赖:
  pip install datasets sentence-transformers

用法:
  source .venv/bin/activate
  python scripts/prepare_mixed_corpus.py
  python scripts/prepare_mixed_corpus.py --n-wiki 30000 --n-msmarco 20000
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import List

import numpy as np


def _is_good_paragraph(para: str) -> bool:
    """判断段落是否适合用作检索语料."""
    if len(para) < 80 or len(para) > 600:
        return False
    if para.startswith("|") or para.startswith("*") or para.startswith("{"):
        return False
    if "." not in para:
        return False
    if para.count(" ") < 10:
        return False
    return True


def collect_wiki_paragraphs(n_paragraphs: int, seed: int = 42) -> List[str]:
    """从 FineWiki (2025年8月 Wikipedia dump) 收集段落 (streaming 模式).

    FineWiki 是 HuggingFace 维护的高质量 Wikipedia 解析版本,
    基于 2025年8月的 HTML dump, 比 wikimedia/wikipedia (2023.11) 新两年.
    """
    from datasets import load_dataset

    print(f"[Wiki] 从 FineWiki (2025.08) 收集段落 (目标: {n_paragraphs})...")

    candidates: List[str] = []
    seen = set()
    n_articles = 0

    ds = load_dataset(
        "HuggingFaceFW/finewiki", "en",
        split="train", streaming=True,
        trust_remote_code=True,
    )

    for article in ds:
        n_articles += 1
        text = article.get("text", "")
        if not text:
            continue

        # FineWiki 用 markdown 格式, 按双换行分段, 跳过标题行
        for para in text.split("\n\n"):
            para = para.strip()
            # 跳过 markdown 标题行
            if para.startswith("#"):
                continue
            if not _is_good_paragraph(para):
                continue
            h = hashlib.md5(para.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            candidates.append(para)

        if n_articles % 10000 == 0:
            print(f"  已扫描 {n_articles} 篇文章, 候选段落: {len(candidates)}")

        if len(candidates) >= n_paragraphs * 3:
            break

    print(f"  扫描了 {n_articles} 篇文章, 候选段落: {len(candidates)}")

    rng = random.Random(seed)
    if len(candidates) > n_paragraphs:
        candidates = rng.sample(candidates, n_paragraphs)
    print(f"  采样 {len(candidates)} 条 Wikipedia 段落")

    return candidates


def collect_fineweb_paragraphs(
    n_paragraphs: int,
    snapshot: str = "CC-MAIN-2024-51",
    seed: int = 42,
) -> List[str]:
    """从 FineWeb 收集网页段落 (streaming 模式).

    默认使用 CC-MAIN-2024-51 (2024年12月) snapshot.
    """
    from datasets import load_dataset

    print(f"[FineWeb] 从 FineWeb {snapshot} 收集段落 (目标: {n_paragraphs})...")

    candidates: List[str] = []
    seen = set()
    n_docs = 0

    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name=snapshot,
        split="train", streaming=True,
        trust_remote_code=True,
    )

    for doc in ds:
        n_docs += 1
        text = doc.get("text", "")
        if not text:
            continue

        for para in text.split("\n\n"):
            para = para.strip()
            if not _is_good_paragraph(para):
                continue
            h = hashlib.md5(para.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            candidates.append(para)

        if n_docs % 5000 == 0:
            print(f"  已扫描 {n_docs} 篇网页, 候选段落: {len(candidates)}")

        if len(candidates) >= n_paragraphs * 3:
            break

    print(f"  扫描了 {n_docs} 篇网页, 候选段落: {len(candidates)}")

    rng = random.Random(seed)
    if len(candidates) > n_paragraphs:
        candidates = rng.sample(candidates, n_paragraphs)
    print(f"  采样 {len(candidates)} 条 FineWeb 段落")

    return candidates


def load_msmarco_passages(
    data_dir: Path, n_passages: int, seed: int = 42,
) -> List[str]:
    """从已有 MS MARCO (Bing) 数据中加载并采样."""
    passages_file = data_dir / "msmarco_passages.txt"
    if not passages_file.exists():
        print(f"[MSMARCO] 跳过 (文件不存在: {passages_file})")
        return []

    with open(passages_file) as f:
        all_passages = [line.strip() for line in f if line.strip()]

    print(f"[MSMARCO] 加载了 {len(all_passages)} 条 Bing 搜索结果")

    rng = random.Random(seed)
    if len(all_passages) > n_passages:
        passages = rng.sample(all_passages, n_passages)
    else:
        passages = all_passages

    print(f"  采样 {len(passages)} 条 MS MARCO passages")
    return passages


def encode_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> np.ndarray:
    """用 sentence-transformers 编码文本为向量."""
    from sentence_transformers import SentenceTransformer

    print(f"  加载模型 {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"  编码 {len(texts)} 条文本 (batch_size={batch_size})...")
    embeddings = model.encode(
        texts, show_progress_bar=True, batch_size=batch_size,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="构建 Wikipedia + MS MARCO 混合语料库"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/mixed"))
    parser.add_argument("--msmarco-dir", type=Path, default=Path("data/msmarco"))
    parser.add_argument("--n-wiki", type=int, default=30000,
                        help="Wikipedia 段落数量")
    parser.add_argument("--n-msmarco", type=int, default=20000,
                        help="MS MARCO (Bing) passage 数量")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Wikipedia (百科知识)
    print("\n" + "=" * 60)
    wiki_paragraphs = collect_wiki_paragraphs(args.n_wiki, seed=args.seed)

    # Step 2: MS MARCO (Bing 搜索结果)
    print("\n" + "=" * 60)
    msmarco_passages = load_msmarco_passages(
        args.msmarco_dir, args.n_msmarco, seed=args.seed,
    )

    # Step 3: 合并
    print("\n" + "=" * 60)
    all_passages = wiki_paragraphs + msmarco_passages
    sources = (
        ["wiki"] * len(wiki_paragraphs)
        + ["msmarco"] * len(msmarco_passages)
    )
    print(f"[合并] 总计 {len(all_passages)} 条")
    print(f"  Wikipedia: {len(wiki_paragraphs)}")
    print(f"  MS MARCO:  {len(msmarco_passages)}")

    # Step 4: 保存文本
    with open(out_dir / "passages.txt", "w") as f:
        for p in all_passages:
            f.write(p.replace("\n", " ") + "\n")
    with open(out_dir / "passage_sources.txt", "w") as f:
        for s in sources:
            f.write(s + "\n")
    print(f"  保存: {out_dir}/passages.txt, passage_sources.txt")

    # Step 5: 生成新鲜度分数
    from diverse_search.temporal import generate_freshness_scores
    freshness = generate_freshness_scores(sources)
    np.save(out_dir / "passage_freshness.npy", freshness)
    print(f"  保存: {out_dir}/passage_freshness.npy shape={freshness.shape}")
    print(f"    wiki 新鲜度: {freshness[sources.index('wiki')]:.2f}" if "wiki" in sources else "")
    print(f"    msmarco 新鲜度: {freshness[sources.index('msmarco')]:.2f}" if "msmarco" in sources else "")

    # Step 6: 编码
    print(f"\n[编码] 编码 {len(all_passages)} 条文本...")
    embeddings = encode_texts(all_passages, args.model, args.batch_size)
    np.save(out_dir / "passage_embeddings.npy", embeddings)
    print(f"  保存: {out_dir}/passage_embeddings.npy shape={embeddings.shape}")

    # 完成
    print(f"\n{'=' * 60}")
    print("混合语料库构建完成!")
    print(f"{'=' * 60}")
    print(f"  Wikipedia: {len(wiki_paragraphs)} 条 (百科知识)")
    print(f"  MS MARCO:  {len(msmarco_passages)} 条 (Bing搜索)")
    print(f"  总计:      {len(all_passages)} 条, dim={embeddings.shape[1]}")
    print(f"  文件位置:  {out_dir}/")
    print(f"""
下一步 - 重启 Demo:

  python demo/search_api.py
  # 打开浏览器: http://localhost:8000
""")


if __name__ == "__main__":
    main()
