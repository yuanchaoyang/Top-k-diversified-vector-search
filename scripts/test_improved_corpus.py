#!/usr/bin/env python3
"""
测试改进语料库的质量

测试覆盖度、多样性和查询响应质量
"""

import sys
from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer

# 确保能导入项目模块
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diverse_search.index import build_retriever
from diverse_search.mmr import mmr_rerank_incremental
from diverse_search.metrics import avg_pairwise_cosine, mean_cosine_to_query


def load_corpus(corpus_dir: str = "data/improved"):
    """加载语料库"""
    corpus_path = Path(corpus_dir)

    print(f"加载语料库: {corpus_path}")

    # 加载段落
    with open(corpus_path / "passages.txt", encoding="utf-8") as f:
        passages = [line.strip() for line in f]

    # 加载嵌入
    embeddings = np.load(corpus_path / "passage_embeddings.npy")

    # 加载来源
    sources = []
    if (corpus_path / "passage_sources.txt").exists():
        with open(corpus_path / "passage_sources.txt", encoding="utf-8") as f:
            sources = [line.strip() for line in f]

    # 加载元数据
    metadata = {}
    if (corpus_path / "metadata.json").exists():
        with open(corpus_path / "metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)

    print(f"  段落数: {len(passages)}")
    print(f"  嵌入维度: {embeddings.shape[1]}")

    if metadata:
        print("\n来源分布:")
        for src, count in sorted(metadata.get("source_distribution", {}).items()):
            pct = count / len(passages) * 100
            print(f"  {src}: {count} ({pct:.1f}%)")

    return passages, embeddings, sources


def test_coverage(passages, embeddings, model):
    """测试关键查询的覆盖度"""
    print("\n" + "=" * 60)
    print("测试1: 关键查询覆盖度")
    print("=" * 60)

    # 测试查询（包括常见的问题查询）
    test_queries = [
        # 单词概念
        "apple",
        "bank",
        "word",
        "music",
        "light",
        "mercury",

        # 实体
        "Albert Einstein",
        "United States",
        "Python programming",
        "Apple Inc",

        # 歧义查询
        "apple fruit",
        "apple company",
        "bank finance",
        "river bank",

        # 问答式
        "what is machine learning",
        "how does photosynthesis work",
        "who invented the telephone",
    ]

    retriever = build_retriever(embeddings, backend="numpy")

    print("\n查询 -> 最相关段落（前3个）\n")

    for query in test_queries:
        # 编码查询
        query_vec = model.encode([query])[0]
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        # 检索
        res = retriever.search(query_vec.reshape(1, -1), 10)
        top_ids = res.indices[0][:3]
        top_scores = res.scores[0][:3]

        print(f"'{query}':")
        for rank, (idx, score) in enumerate(zip(top_ids, top_scores), 1):
            snippet = passages[idx][:80] + "..." if len(passages[idx]) > 80 else passages[idx]
            print(f"  {rank}. [{score:.3f}] {snippet}")
        print()


def test_diversity(passages, embeddings, model):
    """测试多样性查询的效果"""
    print("\n" + "=" * 60)
    print("测试2: 多样性查询效果")
    print("=" * 60)

    test_cases = [
        ("apple", 10, [0.5, 0.7, 0.9]),  # 歧义查询
        ("bank", 10, [0.5, 0.7, 0.9]),
        ("python", 10, [0.5, 0.7, 0.9]),
    ]

    retriever = build_retriever(embeddings, backend="numpy")

    for query, k, lambdas in test_cases:
        print(f"\n查询: '{query}' (k={k})")

        # 编码查询
        query_vec = model.encode([query])[0]
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        # 检索候选
        res = retriever.search(query_vec.reshape(1, -1), 200)
        candidates = res.indices[0]

        for lam in lambdas:
            # MMR重排
            selected = mmr_rerank_incremental(
                query_vec, candidates, embeddings,
                k=k, lambda_=lam, use_max_redundancy=True
            )

            # 计算指标
            selected_vecs = embeddings[selected]
            relevance = mean_cosine_to_query(query_vec, selected_vecs)
            diversity = 1.0 - avg_pairwise_cosine(selected_vecs)

            print(f"\n  λ={lam:.1f} | 相关性={relevance:.3f} | 多样性={diversity:.3f}")
            print("  结果:")
            for rank, idx in enumerate(selected[:5], 1):
                snippet = passages[idx][:60] + "..." if len(passages[idx]) > 60 else passages[idx]
                print(f"    {rank}. {snippet}")


def test_disambiguation(passages, embeddings, sources, model):
    """测试消歧义能力"""
    print("\n" + "=" * 60)
    print("测试3: 消歧义能力")
    print("=" * 60)

    # 检查是否有消歧义页面
    if sources:
        disamb_count = sum(1 for s in sources if "disamb" in s)
        print(f"\n消歧义页面数量: {disamb_count}")

        if disamb_count > 0:
            print("\n示例消歧义页面:")
            count = 0
            for i, (passage, source) in enumerate(zip(passages, sources)):
                if "disamb" in source:
                    print(f"  - {passage[:100]}...")
                    count += 1
                    if count >= 3:
                        break
    else:
        print("  未找到来源信息")


def test_word_definitions(passages, sources):
    """测试词典定义覆盖"""
    print("\n" + "=" * 60)
    print("测试4: 词典定义覆盖")
    print("=" * 60)

    if sources:
        wordnet_count = sum(1 for s in sources if "wordnet" in s)
        print(f"\nWordNet定义数量: {wordnet_count}")

        if wordnet_count > 0:
            print("\n示例WordNet定义:")
            count = 0
            for passage, source in zip(passages, sources):
                if "wordnet" in source:
                    print(f"  - {passage}")
                    count += 1
                    if count >= 5:
                        break
    else:
        print("  未找到来源信息")


def test_entity_coverage(passages, sources):
    """测试实体覆盖"""
    print("\n" + "=" * 60)
    print("测试5: 实体卡片覆盖")
    print("=" * 60)

    if sources:
        entity_count = sum(1 for s in sources if "wikidata" in s)
        print(f"\nWikidata实体卡片数量: {entity_count}")

        if entity_count > 0:
            print("\n示例实体卡片:")
            count = 0
            for passage, source in zip(passages, sources):
                if "wikidata" in source:
                    print(f"  - {passage}")
                    count += 1
                    if count >= 5:
                        break
    else:
        print("  未找到来源信息")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="测试改进语料库质量")
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/improved",
        help="语料库目录"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="句子编码模型"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["all", "coverage", "diversity", "disamb", "wordnet", "entity"],
        default="all",
        help="运行的测试"
    )

    args = parser.parse_args()

    # 加载语料库
    passages, embeddings, sources = load_corpus(args.corpus)

    # 加载模型
    print(f"\n加载模型: {args.model}")
    model = SentenceTransformer(args.model)

    # 运行测试
    if args.test in ["all", "coverage"]:
        test_coverage(passages, embeddings, model)

    if args.test in ["all", "diversity"]:
        test_diversity(passages, embeddings, model)

    if args.test in ["all", "disamb"]:
        test_disambiguation(passages, embeddings, sources, model)

    if args.test in ["all", "wordnet"]:
        test_word_definitions(passages, sources)

    if args.test in ["all", "entity"]:
        test_entity_coverage(passages, sources)

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
