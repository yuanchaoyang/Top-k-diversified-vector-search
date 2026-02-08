#!/usr/bin/env python3
"""
可视化实验：证明 Adaptive Lambda 比 Fixed Lambda 更好

生成的图表：
1. tradeoff_comparison.png - Recall vs Diversity 权衡曲线（核心图）
2. metrics_bar.png - 各方法指标对比条形图
3. adaptive_lambda_dist.png - Adaptive Lambda 分布直方图
4. per_query_analysis.png - 不同查询类型的性能分析
5. pareto_frontier.png - Pareto前沿分析

用法:
    PYTHONPATH=src python scripts/visualize_adaptive.py --dataset glove-100-angular
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from diverse_search.adaptive_lambda import AdaptiveLambdaConfig, batch_adaptive_lambdas
from diverse_search.datasets import download_dataset, load_hdf5, DATASET_URLS
from diverse_search.index import build_retriever
from diverse_search.experiment import (
    evaluate_baseline_topk,
    evaluate_mmr,
    evaluate_mmr_adaptive,
)
from diverse_search.mmr import mmr_rerank_incremental
from diverse_search.metrics import mean_cosine_to_query, avg_pairwise_cosine


def run_comprehensive_experiment(
    dataset_name: str,
    data_dir: str = "data",
    k: int = 10,
    topN: int = 200,
    max_train: int = 50000,
    max_test: int = 500,
):
    """运行完整的对比实验"""

    # 1) 加载数据集
    print(f"Loading dataset: {dataset_name}")
    if dataset_name.endswith(".hdf5"):
        path = Path(dataset_name)
    else:
        path = Path(data_dir) / f"{dataset_name}.hdf5"
        download_dataset(dataset_name, path)

    ds = load_hdf5(path, max_train=max_train, max_test=max_test, normalize=True)
    xb = ds.train
    queries = ds.test
    gt_neighbors = ds.neighbors

    print(f"  Database: {xb.shape}, Queries: {queries.shape}")

    # 2) 构建检索器
    retriever = build_retriever(xb, backend="numpy")

    # 3) 定义实验配置
    fixed_lambdas = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    adaptive_cfg_gap = AdaptiveLambdaConfig(
        lambda_min=0.5,
        lambda_max=0.95,
        lambda_mid=0.8,
        strategy="gap_piecewise",
        gap_k=k,
        gap_t_low=0.02,
        gap_t_high=0.08,
    )

    adaptive_cfg_entropy = AdaptiveLambdaConfig(
        lambda_min=0.5,
        lambda_max=0.95,
        strategy="entropy",
        temperature=0.08,
        topM=50,
    )

    results = []

    # 4) 运行 baseline
    print("Running baseline...")
    res = evaluate_baseline_topk(
        retriever, xb, queries, k=k, topN=topN, gt_neighbors=gt_neighbors
    )
    res["label"] = "baseline"
    results.append(res)

    # 5) 运行 fixed lambda MMR
    print("Running MMR with fixed lambdas...")
    for lam in fixed_lambdas:
        res = evaluate_mmr(
            retriever, xb, queries, k=k, topN=topN, lambda_=lam,
            gt_neighbors=gt_neighbors, impl="incremental"
        )
        res["label"] = f"MMR(λ={lam})"
        results.append(res)

    # 6) 运行 adaptive lambda MMR (gap strategy)
    print("Running MMR with adaptive lambda (gap_piecewise)...")
    res = evaluate_mmr_adaptive(
        retriever, xb, queries, k=k, topN=topN,
        lambda_cfg=adaptive_cfg_gap, gt_neighbors=gt_neighbors, impl="incremental"
    )
    res["label"] = "MMR-Adaptive (gap)"
    results.append(res)

    # 7) 运行 adaptive lambda MMR (entropy strategy)
    print("Running MMR with adaptive lambda (entropy)...")
    res = evaluate_mmr_adaptive(
        retriever, xb, queries, k=k, topN=topN,
        lambda_cfg=adaptive_cfg_entropy, gt_neighbors=gt_neighbors, impl="incremental"
    )
    res["label"] = "MMR-Adaptive (entropy)"
    results.append(res)

    df = pd.DataFrame(results)

    # 8) 获取每个查询的adaptive lambda值（用于后续分析）
    search_res = retriever.search(queries, topN)
    lambdas_gap = batch_adaptive_lambdas(search_res.scores, adaptive_cfg_gap)
    lambdas_entropy = batch_adaptive_lambdas(search_res.scores, adaptive_cfg_entropy)

    return df, {
        "xb": xb,
        "queries": queries,
        "retriever": retriever,
        "search_res": search_res,
        "lambdas_gap": lambdas_gap,
        "lambdas_entropy": lambdas_entropy,
        "adaptive_cfg_gap": adaptive_cfg_gap,
        "adaptive_cfg_entropy": adaptive_cfg_entropy,
        "k": k,
        "topN": topN,
        "gt_neighbors": gt_neighbors,
    }


def plot_tradeoff_comparison(df: pd.DataFrame, out_dir: Path):
    """图1: Recall vs Diversity 权衡曲线"""

    fig, ax = plt.subplots(figsize=(10, 7))

    # 分离不同类型的方法
    baseline = df[df["method"] == "baseline"]
    fixed_mmr = df[(df["method"] == "mmr")]
    adaptive = df[df["method"] == "mmr_adaptive"]

    # 绘制fixed MMR曲线
    fixed_sorted = fixed_mmr.sort_values("lambda")
    ax.plot(fixed_sorted["ild"], fixed_sorted["recall"],
            "o-", color="steelblue", markersize=8, linewidth=2,
            label="MMR (fixed λ)", zorder=2)

    # 标注lambda值
    for _, row in fixed_sorted.iterrows():
        ax.annotate(f'λ={row["lambda"]:.1f}',
                   (row["ild"], row["recall"]),
                   textcoords="offset points", xytext=(5, 5), fontsize=8)

    # 绘制baseline
    ax.scatter(baseline["ild"], baseline["recall"],
               marker="s", s=120, color="gray", label="Baseline", zorder=3)

    # 绘制adaptive（突出显示）
    colors = ["#e74c3c", "#27ae60"]
    markers = ["*", "D"]
    for i, (_, row) in enumerate(adaptive.iterrows()):
        ax.scatter(row["ild"], row["recall"],
                   marker=markers[i], s=250, color=colors[i],
                   edgecolors="black", linewidth=1.5,
                   label=row["label"], zorder=4)

    ax.set_xlabel("Intra-List Diversity (ILD)", fontsize=12)
    ax.set_ylabel("Recall@k", fontsize=12)
    ax.set_title("Relevance-Diversity Trade-off: Adaptive vs Fixed λ", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "tradeoff_comparison.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")
    plt.close()


def plot_metrics_bar(df: pd.DataFrame, out_dir: Path):
    """图2: 多指标对比条形图"""

    # 选择代表性方法进行对比
    selected = df[df["label"].isin([
        "baseline", "MMR(λ=0.5)", "MMR(λ=0.8)", "MMR(λ=0.95)",
        "MMR-Adaptive (gap)", "MMR-Adaptive (entropy)"
    ])].copy()

    metrics = ["recall", "rel_mean_cos", "ild"]
    metric_labels = ["Recall@k", "Relevance", "Diversity (ILD)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    colors = {
        "baseline": "#95a5a6",
        "MMR(λ=0.5)": "#3498db",
        "MMR(λ=0.8)": "#2980b9",
        "MMR(λ=0.95)": "#1a5276",
        "MMR-Adaptive (gap)": "#e74c3c",
        "MMR-Adaptive (entropy)": "#27ae60",
    }

    for ax, metric, label in zip(axes, metrics, metric_labels):
        bars = ax.bar(range(len(selected)), selected[metric],
                     color=[colors.get(l, "gray") for l in selected["label"]])
        ax.set_xticks(range(len(selected)))
        ax.set_xticklabels(selected["label"], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        # 标注adaptive的值
        for i, (idx, row) in enumerate(selected.iterrows()):
            if "Adaptive" in row["label"]:
                ax.annotate(f'{row[metric]:.3f}',
                           (i, row[metric]),
                           textcoords="offset points", xytext=(0, 5),
                           ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Performance Comparison Across Methods", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = out_dir / "metrics_bar.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_lambda_distribution(ctx: dict, out_dir: Path):
    """图3: Adaptive Lambda 分布直方图"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gap strategy
    ax = axes[0]
    ax.hist(ctx["lambdas_gap"], bins=30, color="#e74c3c", alpha=0.7,
            edgecolor="black", linewidth=0.5)
    ax.axvline(np.mean(ctx["lambdas_gap"]), color="black", linestyle="--",
               linewidth=2, label=f'Mean: {np.mean(ctx["lambdas_gap"]):.3f}')
    ax.set_xlabel("λ value", fontsize=11)
    ax.set_ylabel("Query count", fontsize=11)
    ax.set_title("Adaptive λ Distribution (Gap Strategy)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Entropy strategy
    ax = axes[1]
    ax.hist(ctx["lambdas_entropy"], bins=30, color="#27ae60", alpha=0.7,
            edgecolor="black", linewidth=0.5)
    ax.axvline(np.mean(ctx["lambdas_entropy"]), color="black", linestyle="--",
               linewidth=2, label=f'Mean: {np.mean(ctx["lambdas_entropy"]):.3f}')
    ax.set_xlabel("λ value", fontsize=11)
    ax.set_ylabel("Query count", fontsize=11)
    ax.set_title("Adaptive λ Distribution (Entropy Strategy)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Query Adaptive Lambda Distribution", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = out_dir / "adaptive_lambda_dist.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_per_query_analysis(ctx: dict, out_dir: Path):
    """图4: 不同查询类型的性能分析 - 证明adaptive的优势"""

    xb = ctx["xb"]
    queries = ctx["queries"]
    search_res = ctx["search_res"]
    k = ctx["k"]
    lambdas_gap = ctx["lambdas_gap"]

    # 将查询分为"明确"和"模糊"两类
    median_lambda = np.median(lambdas_gap)
    clear_mask = lambdas_gap >= median_lambda  # 高lambda = 明确查询
    ambig_mask = ~clear_mask  # 低lambda = 模糊查询

    # 对于每种查询类型，比较 fixed vs adaptive 的效果
    fixed_lambdas_to_test = [0.6, 0.8, 0.95]

    results = {"clear": {}, "ambiguous": {}}

    for query_type, mask in [("clear", clear_mask), ("ambiguous", ambig_mask)]:
        q_subset = queries[mask]
        cand_subset = search_res.indices[mask]
        lambda_subset = lambdas_gap[mask]

        nq = q_subset.shape[0]
        if nq == 0:
            continue

        # Fixed lambda 性能
        for fixed_lam in fixed_lambdas_to_test:
            rel_list, div_list = [], []
            for qi in range(nq):
                selected = mmr_rerank_incremental(
                    q_subset[qi], cand_subset[qi], xb, k=k, lambda_=fixed_lam
                )
                vecs = xb[selected]
                rel_list.append(mean_cosine_to_query(q_subset[qi], vecs))
                div_list.append(1.0 - avg_pairwise_cosine(vecs))
            results[query_type][f"fixed_{fixed_lam}"] = {
                "rel": np.mean(rel_list), "div": np.mean(div_list)
            }

        # Adaptive lambda 性能
        rel_list, div_list = [], []
        for qi in range(nq):
            selected = mmr_rerank_incremental(
                q_subset[qi], cand_subset[qi], xb, k=k, lambda_=float(lambda_subset[qi])
            )
            vecs = xb[selected]
            rel_list.append(mean_cosine_to_query(q_subset[qi], vecs))
            div_list.append(1.0 - avg_pairwise_cosine(vecs))
        results[query_type]["adaptive"] = {
            "rel": np.mean(rel_list), "div": np.mean(div_list)
        }

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, query_type, title in zip(axes, ["clear", "ambiguous"],
                                      ["Clear Queries (High λ)", "Ambiguous Queries (Low λ)"]):
        if query_type not in results or not results[query_type]:
            continue

        data = results[query_type]
        methods = list(data.keys())
        rel_vals = [data[m]["rel"] for m in methods]
        div_vals = [data[m]["div"] for m in methods]

        x = np.arange(len(methods))
        width = 0.35

        bars1 = ax.bar(x - width/2, rel_vals, width, label="Relevance", color="#3498db")
        bars2 = ax.bar(x + width/2, div_vals, width, label="Diversity", color="#e74c3c")

        ax.set_xlabel("Method", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        labels = [m.replace("fixed_", "λ=") if "fixed" in m else "Adaptive" for m in methods]
        ax.set_xticklabels(labels, fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # 高亮adaptive
        for i, m in enumerate(methods):
            if m == "adaptive":
                bars1[i].set_edgecolor("black")
                bars1[i].set_linewidth(2)
                bars2[i].set_edgecolor("black")
                bars2[i].set_linewidth(2)

    plt.suptitle("Adaptive λ Adapts to Query Characteristics", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = out_dir / "per_query_analysis.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_pareto_frontier(df: pd.DataFrame, out_dir: Path):
    """图5: Pareto前沿分析"""

    fig, ax = plt.subplots(figsize=(10, 7))

    # 所有fixed lambda点
    fixed_mmr = df[df["method"] == "mmr"].copy()
    adaptive = df[df["method"] == "mmr_adaptive"].copy()

    # 绘制所有fixed点
    ax.scatter(fixed_mmr["ild"], fixed_mmr["recall"],
               c="steelblue", s=80, alpha=0.7, label="MMR (fixed λ)")

    # 计算并绘制Pareto前沿
    points = fixed_mmr[["ild", "recall"]].values
    pareto_mask = np.ones(len(points), dtype=bool)
    for i, (x1, y1) in enumerate(points):
        for j, (x2, y2) in enumerate(points):
            if i != j and x2 >= x1 and y2 >= y1 and (x2 > x1 or y2 > y1):
                pareto_mask[i] = False
                break

    pareto_points = fixed_mmr[pareto_mask].sort_values("ild")
    ax.plot(pareto_points["ild"], pareto_points["recall"],
            "b--", linewidth=2, alpha=0.5, label="Pareto Frontier (fixed)")

    # 绘制adaptive点（突出显示）
    colors = ["#e74c3c", "#27ae60"]
    for i, (_, row) in enumerate(adaptive.iterrows()):
        ax.scatter(row["ild"], row["recall"],
                   marker="*", s=400, color=colors[i],
                   edgecolors="black", linewidth=2,
                   label=row["label"], zorder=5)

        # 检查是否在Pareto前沿上或之上
        is_dominated = False
        for _, p in pareto_points.iterrows():
            if p["ild"] >= row["ild"] and p["recall"] >= row["recall"]:
                if p["ild"] > row["ild"] or p["recall"] > row["recall"]:
                    is_dominated = True
                    break

        status = "Dominated" if is_dominated else "On/Above Pareto"
        ax.annotate(f'{row["label"]}\n({status})',
                   (row["ild"], row["recall"]),
                   textcoords="offset points", xytext=(10, -20), fontsize=9)

    ax.set_xlabel("Diversity (ILD)", fontsize=12)
    ax.set_ylabel("Recall@k", fontsize=12)
    ax.set_title("Pareto Frontier Analysis: Adaptive vs Fixed λ", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "pareto_frontier.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")
    plt.close()


def generate_summary_table(df: pd.DataFrame, out_dir: Path):
    """生成结果汇总表格"""

    # 选择关键列
    cols = ["label", "recall", "rel_mean_cos", "ild", "lambda"]
    summary = df[cols].copy()
    summary.columns = ["Method", "Recall@k", "Relevance", "ILD", "λ (mean)"]

    # 格式化
    for col in ["Recall@k", "Relevance", "ILD"]:
        summary[col] = summary[col].apply(lambda x: f"{x:.4f}")
    summary["λ (mean)"] = summary["λ (mean)"].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "-"
    )

    out_path = out_dir / "results_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    print("="*60 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Visualize Adaptive Lambda vs Fixed Lambda")
    parser.add_argument("--dataset", type=str, default="glove-100-angular",
                       help="Dataset name or path to .hdf5 file")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="outputs/adaptive_analysis")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--topN", type=int, default=200)
    parser.add_argument("--max-train", type=int, default=50000)
    parser.add_argument("--max-test", type=int, default=500)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("ADAPTIVE LAMBDA VISUALIZATION EXPERIMENT")
    print("="*60)

    # 运行实验
    df, ctx = run_comprehensive_experiment(
        args.dataset,
        data_dir=args.data_dir,
        k=args.k,
        topN=args.topN,
        max_train=args.max_train,
        max_test=args.max_test,
    )

    # 生成所有图表
    print("\nGenerating visualizations...")
    plot_tradeoff_comparison(df, out_dir)
    plot_metrics_bar(df, out_dir)
    plot_lambda_distribution(ctx, out_dir)
    plot_per_query_analysis(ctx, out_dir)
    plot_pareto_frontier(df, out_dir)

    # 生成汇总表
    generate_summary_table(df, out_dir)

    print(f"\nAll outputs saved to: {out_dir}")
    print("\nKey findings to look for:")
    print("  1. Adaptive λ should be near or on the Pareto frontier")
    print("  2. Adaptive λ distribution should show variance (not all same value)")
    print("  3. For clear queries: Adaptive should favor relevance")
    print("  4. For ambiguous queries: Adaptive should favor diversity")


if __name__ == "__main__":
    main()
