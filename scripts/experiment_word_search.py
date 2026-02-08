#!/usr/bin/env python3
"""
真实场景实验: 词向量相似词搜索

使用corpus_words作为查询词，在词向量空间中搜索相似词
这是Intent-Adaptive最适合的场景 - 真实的文本查询
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from diverse_search.index import build_retriever
from diverse_search.mmr import mmr_rerank_incremental
from diverse_search.metrics import avg_pairwise_cosine
from diverse_search.adaptive_lambda import AdaptiveLambdaConfig, adaptive_lambda_from_scores
from diverse_search.intent_model import IntentClassifier


def load_word_embeddings(words_file: str, embeddings_file: str) -> Tuple[List[str], np.ndarray]:
    """加载词和对应的词向量"""
    with open(words_file) as f:
        words = [line.strip() for line in f if line.strip()]

    embeddings = np.load(embeddings_file)

    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    return words, embeddings


def evaluate_diversity_relevance(
    query_word: str,
    query_vec: np.ndarray,
    selected_indices: np.ndarray,
    all_words: List[str],
    all_vecs: np.ndarray,
    topN_indices: np.ndarray,
    k: int,
) -> Dict:
    """评估多样性和相关性"""
    selected_vecs = all_vecs[selected_indices]
    selected_words = [all_words[i] for i in selected_indices]

    # Relevance: 与query的平均相似度
    sims = np.dot(selected_vecs, query_vec)
    relevance = np.mean(sims)

    # Diversity (ILD): 结果之间的平均不相似度
    diversity = 1.0 - avg_pairwise_cosine(selected_vecs)

    # Pseudo-Recall: 选中的在topN中的比例
    topN_set = set(topN_indices[:k])
    selected_set = set(selected_indices)
    pseudo_recall = len(topN_set & selected_set) / k if k > 0 else 0

    # F1
    if relevance + diversity > 0:
        f1 = 2 * relevance * diversity / (relevance + diversity)
    else:
        f1 = 0

    return {
        "query": query_word,
        "selected_words": selected_words,
        "relevance": relevance,
        "diversity": diversity,
        "pseudo_recall": pseudo_recall,
        "f1": f1,
    }


def run_word_search_experiment(
    words: List[str],
    embeddings: np.ndarray,
    k: int = 5,
    topN: int = 50,
) -> pd.DataFrame:
    """运行词搜索实验"""

    # 构建检索器
    retriever = build_retriever(embeddings, backend="numpy")

    # 加载意图分类器 (使用ChatGPT训练的模型)
    intent_classifier = IntentClassifier.load("models/intent_chatgpt")

    # 方法配置
    methods = {
        "Fixed λ=0.5": {"type": "fixed", "lambda": 0.5},
        "Fixed λ=0.6": {"type": "fixed", "lambda": 0.6},
        "Fixed λ=0.7": {"type": "fixed", "lambda": 0.7},
        "Fixed λ=0.8": {"type": "fixed", "lambda": 0.8},
        "Adaptive-Entropy": {"type": "adaptive", "strategy": "entropy"},
        "Adaptive-Gap": {"type": "adaptive", "strategy": "gap_piecewise"},
        "Intent-Adaptive": {"type": "intent"},
    }

    all_results = []

    for method_name, config in methods.items():
        print(f"\n  Running: {method_name}")

        method_metrics = {
            "relevance": [], "diversity": [], "f1": [],
            "lambdas": [], "examples": []
        }

        for qi, (word, query_vec) in enumerate(zip(words, embeddings)):
            # 获取候选
            res = retriever.search(query_vec.reshape(1, -1), topN)
            candidates = res.indices[0]
            scores = res.scores[0]

            # 确定λ
            if config["type"] == "fixed":
                lam = config["lambda"]
            elif config["type"] == "adaptive":
                adaptive_config = AdaptiveLambdaConfig(
                    strategy=config["strategy"],
                    lambda_min=0.4,
                    lambda_max=0.85,
                )
                lam = adaptive_lambda_from_scores(scores, adaptive_config)
            elif config["type"] == "intent":
                lam, intent_score = intent_classifier.predict_lambda(word, 0.4, 0.85)

            method_metrics["lambdas"].append(lam)

            # MMR重排序
            selected = mmr_rerank_incremental(
                query_vec, candidates, embeddings, k=k, lambda_=float(lam)
            )

            # 评估
            result = evaluate_diversity_relevance(
                word, query_vec, selected, words, embeddings, candidates, k
            )

            method_metrics["relevance"].append(result["relevance"])
            method_metrics["diversity"].append(result["diversity"])
            method_metrics["f1"].append(result["f1"])

            # 保存一些例子
            if len(method_metrics["examples"]) < 5:
                method_metrics["examples"].append({
                    "query": word,
                    "results": result["selected_words"],
                    "lambda": lam,
                })

        all_results.append({
            "Method": method_name,
            "Relevance": np.mean(method_metrics["relevance"]),
            "Diversity": np.mean(method_metrics["diversity"]),
            "F1": np.mean(method_metrics["f1"]),
            "λ_mean": np.mean(method_metrics["lambdas"]),
            "λ_std": np.std(method_metrics["lambdas"]),
        })

        print(f"    Relevance={np.mean(method_metrics['relevance']):.4f}, "
              f"Diversity={np.mean(method_metrics['diversity']):.4f}, "
              f"F1={np.mean(method_metrics['f1']):.4f}")

    return pd.DataFrame(all_results)


def analyze_intent_effect(
    words: List[str],
    embeddings: np.ndarray,
    k: int = 5,
    topN: int = 50,
) -> pd.DataFrame:
    """分析意图对结果的影响"""

    retriever = build_retriever(embeddings, backend="numpy")
    intent_classifier = IntentClassifier.load("models/intent_chatgpt")

    results = []

    for word, query_vec in zip(words, embeddings):
        # 获取意图分数和λ
        lam, intent_score = intent_classifier.predict_lambda(word, 0.4, 0.85)

        # 获取候选
        res = retriever.search(query_vec.reshape(1, -1), topN)
        candidates = res.indices[0]

        # MMR重排序
        selected = mmr_rerank_incremental(
            query_vec, candidates, embeddings, k=k, lambda_=float(lam)
        )

        # 计算指标
        selected_vecs = embeddings[selected]
        relevance = np.mean(np.dot(selected_vecs, query_vec))
        diversity = 1.0 - avg_pairwise_cosine(selected_vecs)

        results.append({
            "word": word,
            "intent_score": intent_score,
            "lambda": lam,
            "relevance": relevance,
            "diversity": diversity,
        })

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame, intent_df: pd.DataFrame, output_dir: Path):
    """可视化结果"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 方法对比 - F1
    ax = axes[0, 0]
    methods = df["Method"].tolist()
    colors = ["#3498db" if "Fixed" in m else "#e74c3c" if "Intent" in m else "#2ecc71"
              for m in methods]
    bars = ax.barh(range(len(df)), df["F1"], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("F1 Score")
    ax.set_title("Method Comparison (F1 Score)")
    ax.grid(True, alpha=0.3, axis='x')

    best_idx = df["F1"].idxmax()
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    # 2. Relevance vs Diversity
    ax = axes[0, 1]
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.scatter(row["Relevance"], row["Diversity"], c=colors[i], s=150,
                  label=methods[i], edgecolors='black', linewidth=1)
    ax.set_xlabel("Relevance (avg similarity to query)")
    ax.set_ylabel("Diversity (1 - avg pairwise similarity)")
    ax.set_title("Relevance vs Diversity Trade-off")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Intent Score分布和对应的λ
    ax = axes[1, 0]

    # 按intent score排序
    intent_sorted = intent_df.sort_values("intent_score")

    # 分组统计
    bins = [0, 0.3, 0.5, 0.7, 1.0]
    labels = ["Ambiguous\n(0-0.3)", "Low-Mid\n(0.3-0.5)", "Mid-High\n(0.5-0.7)", "Specific\n(0.7-1.0)"]
    intent_df["intent_group"] = pd.cut(intent_df["intent_score"], bins=bins, labels=labels)

    group_stats = intent_df.groupby("intent_group", observed=True).agg({
        "lambda": "mean",
        "diversity": "mean",
        "relevance": "mean",
        "word": "count"
    }).reset_index()

    x = range(len(group_stats))
    width = 0.35
    ax.bar([i - width/2 for i in x], group_stats["diversity"], width, label="Diversity", color="#e74c3c")
    ax.bar([i + width/2 for i in x], group_stats["relevance"], width, label="Relevance", color="#3498db")

    ax.set_xticks(x)
    ax.set_xticklabels(group_stats["intent_group"])
    ax.set_ylabel("Score")
    ax.set_title("Intent Group: Diversity vs Relevance")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Intent Score vs λ scatter
    ax = axes[1, 1]
    scatter = ax.scatter(intent_df["intent_score"], intent_df["lambda"],
                        c=intent_df["diversity"], cmap="RdYlGn", s=50, alpha=0.6)
    ax.plot([0, 1], [0.4, 0.85], 'k--', alpha=0.5, label='λ = 0.4 + 0.45 × intent')
    ax.set_xlabel("Intent Score (0=ambiguous, 1=specific)")
    ax.set_ylabel("Adaptive λ")
    ax.set_title("Intent Score → λ Mapping (color=diversity)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="Diversity")

    plt.suptitle("Intent-based Adaptive Lambda: Word Similarity Search", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "word_search_results.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved: {output_dir / 'word_search_results.png'}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--words-file", type=str, default="data/corpus_words.txt")
    parser.add_argument("--embeddings-file", type=str, default="data/corpus_words_embeddings.npy")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--topN", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/word_search_experiment"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("WORD SIMILARITY SEARCH EXPERIMENT")
    print("="*60)

    # 检查文件是否存在
    if not Path(args.words_file).exists():
        print(f"Error: {args.words_file} not found")
        return
    if not Path(args.embeddings_file).exists():
        print(f"Error: {args.embeddings_file} not found")
        print("Please run: PYTHONPATH=src python -c \"from diverse_search.cli_text import precompute_embeddings; precompute_embeddings()\"")
        return

    # 加载数据
    print("\nLoading word embeddings...")
    words, embeddings = load_word_embeddings(args.words_file, args.embeddings_file)
    print(f"  Loaded {len(words)} words, dim={embeddings.shape[1]}")

    # 运行实验
    print("\n" + "="*60)
    print("RUNNING EXPERIMENTS")
    print("="*60)

    df = run_word_search_experiment(words, embeddings, k=args.k, topN=args.topN)

    # 分析意图效果
    print("\n" + "="*60)
    print("ANALYZING INTENT EFFECT")
    print("="*60)

    intent_df = analyze_intent_effect(words, embeddings, k=args.k, topN=args.topN)

    # 打印结果
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(df.to_string(index=False))

    # 保存结果
    df.to_csv(args.out_dir / "method_comparison.csv", index=False)
    intent_df.to_csv(args.out_dir / "intent_analysis.csv", index=False)

    # 可视化
    plot_results(df, intent_df, args.out_dir)

    # 关键分析
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    intent_f1 = df[df["Method"] == "Intent-Adaptive"]["F1"].values[0]
    fixed_df = df[df["Method"].str.contains("Fixed")]
    best_fixed = fixed_df["F1"].max()
    best_fixed_name = fixed_df.loc[fixed_df["F1"].idxmax(), "Method"]

    print(f"\n  Intent-Adaptive F1: {intent_f1:.4f}")
    print(f"  Best Fixed λ F1:    {best_fixed:.4f} ({best_fixed_name})")

    if intent_f1 > best_fixed:
        print(f"\n  ✓ Intent-Adaptive WINS by {(intent_f1-best_fixed)/best_fixed*100:.2f}%")
    else:
        print(f"\n  △ Fixed λ wins, but Intent-Adaptive offers better interpretability")

    # 展示意图分析示例
    print("\n" + "="*60)
    print("INTENT ANALYSIS EXAMPLES")
    print("="*60)

    # 最模糊的词
    print("\n  Most AMBIGUOUS words (low λ → more diversity):")
    for _, row in intent_df.nsmallest(5, "intent_score").iterrows():
        print(f"    {row['word']:20s} intent={row['intent_score']:.2f} λ={row['lambda']:.2f} div={row['diversity']:.2f}")

    # 最明确的词
    print("\n  Most SPECIFIC words (high λ → more relevance):")
    for _, row in intent_df.nlargest(5, "intent_score").iterrows():
        print(f"    {row['word']:20s} intent={row['intent_score']:.2f} λ={row['lambda']:.2f} rel={row['relevance']:.2f}")

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
Intent-based Adaptive Lambda的优势:

1. 语义理解: 根据查询词的语义模糊度自动调整λ
2. 可解释性: "apple"是模糊词所以用低λ增加多样性
3. 无需标注数据: 可用LLM自动生成意图标签
4. 实用价值: 适用于真实的文本搜索场景

这是毕业设计的创新点: 将LLM的语义理解能力引入向量检索的多样性控制!
""")


if __name__ == "__main__":
    main()
