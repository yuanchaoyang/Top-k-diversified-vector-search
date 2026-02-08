#!/usr/bin/env python3
"""
下载并预处理 MS MARCO passage ranking 数据集，适配本项目的 pipeline.

步骤:
  1. 下载 MS MARCO queries + passages (如果不存在)
  2. 用 sentence-transformers 编码成向量
  3. 保存为项目兼容格式 (npy + txt)

依赖:
  pip install sentence-transformers datasets

用法:
  python scripts/prepare_msmarco.py --n-passages 50000 --n-queries 1000
  python scripts/prepare_msmarco.py --n-passages 200000 --n-queries 5000  # 更大规模
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def download_msmarco(data_dir: Path, n_passages: int, n_queries: int):
    """下载 MS MARCO 数据 (通过 HuggingFace datasets)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装: pip install datasets")

    print(f"[1] 加载 MS MARCO passages (取前 {n_passages} 条)...")
    # MS MARCO passage corpus
    ds = load_dataset(
        "microsoft/ms_marco", "v2.1", split="train", trust_remote_code=True,
    )

    # 提取 passages (去重)
    passages: List[str] = []
    passage_set = set()
    for row in ds:
        for p in row.get("passages", {}).get("passage_text", []):
            text = p.strip()
            if text and text not in passage_set and len(text) > 20:
                passages.append(text)
                passage_set.add(text)
                if len(passages) >= n_passages:
                    break
        if len(passages) >= n_passages:
            break
    print(f"  收集了 {len(passages)} 条 passages")

    # 提取 queries
    queries: List[str] = []
    query_set = set()
    for row in ds:
        q = row.get("query", "").strip()
        if q and q not in query_set:
            queries.append(q)
            query_set.add(q)
            if len(queries) >= n_queries:
                break
    print(f"  收集了 {len(queries)} 条 queries")

    # 保存文本
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "msmarco_passages.txt", "w") as f:
        for p in passages:
            f.write(p.replace("\n", " ") + "\n")
    with open(data_dir / "msmarco_queries.txt", "w") as f:
        for q in queries:
            f.write(q.replace("\n", " ") + "\n")

    print(f"  保存到 {data_dir}/msmarco_passages.txt, msmarco_queries.txt")
    return passages, queries


def encode_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> np.ndarray:
    """用 sentence-transformers 编码文本为向量."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("请安装: pip install sentence-transformers")

    print(f"  加载模型 {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"  编码 {len(texts)} 条文本 (batch_size={batch_size})...")
    embeddings = model.encode(
        texts, show_progress_bar=True, batch_size=batch_size,
        normalize_embeddings=True,  # L2归一化, cosine sim = dot product
    )
    return embeddings.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="准备 MS MARCO 数据")
    parser.add_argument("--data-dir", type=Path, default=Path("data/msmarco"))
    parser.add_argument("--n-passages", type=int, default=50000,
                        help="使用的 passage 数量 (50k 约 5分钟编码)")
    parser.add_argument("--n-queries", type=int, default=1000,
                        help="使用的 query 数量")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformers 模型")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过下载，直接从已有文件编码")
    args = parser.parse_args()

    data_dir = args.data_dir

    # Step 1: 下载
    if args.skip_download and (data_dir / "msmarco_passages.txt").exists():
        print("[1] 跳过下载，使用已有文件")
        with open(data_dir / "msmarco_passages.txt") as f:
            passages = [line.strip() for line in f if line.strip()]
        with open(data_dir / "msmarco_queries.txt") as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        passages, queries = download_msmarco(data_dir, args.n_passages, args.n_queries)

    # Step 2: 编码 passages
    passage_emb_path = data_dir / "passage_embeddings.npy"
    if passage_emb_path.exists():
        print(f"\n[2] 加载已有 passage embeddings: {passage_emb_path}")
        passage_embeddings = np.load(passage_emb_path)
    else:
        print(f"\n[2] 编码 passages...")
        passage_embeddings = encode_texts(passages, args.model, args.batch_size)
        np.save(passage_emb_path, passage_embeddings)
        print(f"  保存: {passage_emb_path} shape={passage_embeddings.shape}")

    # Step 3: 编码 queries
    query_emb_path = data_dir / "query_embeddings.npy"
    if query_emb_path.exists():
        print(f"\n[3] 加载已有 query embeddings: {query_emb_path}")
        query_embeddings = np.load(query_emb_path)
    else:
        print(f"\n[3] 编码 queries...")
        query_embeddings = encode_texts(queries, args.model, args.batch_size)
        np.save(query_emb_path, query_embeddings)
        print(f"  保存: {query_emb_path} shape={query_embeddings.shape}")

    # Step 4: 打印统计
    print(f"\n{'='*60}")
    print(f"数据准备完成!")
    print(f"{'='*60}")
    print(f"  Passages: {len(passages)} 条, dim={passage_embeddings.shape[1]}")
    print(f"  Queries:  {len(queries)} 条, dim={query_embeddings.shape[1]}")
    print(f"  文件位置: {data_dir}/")
    print(f"""
下一步 - 运行搜索实验:

  PYTHONPATH=src python scripts/evaluate_msmarco.py \\
    --data-dir {data_dir} \\
    --k 10 --topN 100 \\
    --out-dir outputs/msmarco_eval
""")


if __name__ == "__main__":
    main()
