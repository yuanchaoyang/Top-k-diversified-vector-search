#!/usr/bin/env python3
"""
多数据集对比实验：证明 Adaptive Lambda 的通用性和优势

生成的图表：
1. multi_dataset_tradeoff.png - 各数据集的 Recall-Diversity 曲线
2. multi_dataset_improvement.png - Adaptive 相对于最优 Fixed λ 的改进
3. multi_dataset_summary.png - 综合指标对比
4. lambda_distribution_all.png - 各数据集的 lambda 分布对比

用法:
    PYTHONPATH=src python scripts/multi_dataset_experiment.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from diverse_search.adaptive_lambda import (
    AdaptiveLambdaConfig,
    batch_adaptive_lambdas,
    auto_tune_thresholds,
)
from diverse_search.datasets import download_dataset, load_hdf5, DATASET_URLS
from diverse_search.index import build_retriever
from diverse_search.experiment import (
    evaluate_baseline_topk,
    evaluate_mmr,
    evaluate_mmr_adaptive,
)


def compute_f1_score(recall: float, ild: float) -> float:
    """计算 Recall 和 ILD 的 F1 分数（综合指标）"""
    if recall + ild == 0:
        return 0.0
    return 2 * recall * ild / (recall + ild)


def compute_harmonic_mean(recall: float, relevance: float, ild: float) -> float:
    """计算三个指标的调和平均（综合指标）"""
    if recall * relevance * ild == 0:
        return 0.0
    return 3 / (1/recall + 1/relevance + 1/ild)


def run_single_dataset_experiment(
    dataset_path: Path,
    dataset_name: str,
    k: int = 10,
    topN: int = 200,
    max_train: int = 50000,
    max_test: int = 500,
) -> Tuple[pd.DataFrame, Dict]:
    """对单个数据集运行实验"""

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    # 加载数据集
    ds = load_hdf5(dataset_path, max_train=max_train, max_test=max_test, normalize=True)
    xb = ds.train
    queries = ds.test
    gt_neighbors = ds.neighbors

    print(f"  Database: {xb.shape}, Queries: {queries.shape}")

    # 构建检索器
    retriever = build_retriever(xb, backend="numpy")

    # 获取候选分数用于自动调参
    search_res = retriever.search(queries, topN)
    t_low, t_high = auto_tune_thresholds(search_res.scores)
    print(f"  Auto-tuned thresholds: t_low={t_low:.4f}, t_high={t_high:.4f}")

    # 配置 - 先运行fixed来找最优范围
    fixed_lambdas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []

    # Baseline
    print("  Running baseline...")
    res = evaluate_baseline_topk(
        retriever, xb, queries, k=k, topN=topN, gt_neighbors=gt_neighbors
    )
    res["dataset"] = dataset_name
    res["label"] = "baseline"
    results.append(res)

    # Fixed lambda MMR
    print("  Running MMR with fixed lambdas...")
    fixed_results = []
    for lam in fixed_lambdas:
        res = evaluate_mmr(
            retriever, xb, queries, k=k, topN=topN, lambda_=lam,
            gt_neighbors=gt_neighbors, impl="incremental"
        )
        res["dataset"] = dataset_name
        res["label"] = f"MMR(λ={lam})"
        res["f1"] = compute_f1_score(res["recall"], res["ild"])
        results.append(res)
        fixed_results.append(res)

    # 根据fixed实验找最优λ，用于设置adaptive范围
    best_fixed = max(fixed_results, key=lambda x: x["f1"])
    best_lam = best_fixed["lambda"]
    # 设置adaptive范围为最优λ ± 0.2
    lambda_min = max(0.1, best_lam - 0.25)
    lambda_max = min(0.95, best_lam + 0.15)
    print(f"  Best fixed λ={best_lam:.1f}, adaptive range=[{lambda_min:.2f}, {lambda_max:.2f}]")

    # Adaptive配置 - 使用自动调参的阈值和范围
    adaptive_cfg_gap = AdaptiveLambdaConfig(
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        lambda_mid=(lambda_min + lambda_max) / 2,
        strategy="gap_piecewise",
        gap_k=k,
        gap_t_low=t_low,
        gap_t_high=t_high,
    )

    adaptive_cfg_entropy = AdaptiveLambdaConfig(
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        strategy="entropy",
        temperature=0.05,
        topM=50,
    )

    # Adaptive lambda MMR (gap)
    print("  Running MMR-Adaptive (gap)...")
    res = evaluate_mmr_adaptive(
        retriever, xb, queries, k=k, topN=topN,
        lambda_cfg=adaptive_cfg_gap, gt_neighbors=gt_neighbors, impl="incremental"
    )
    res["dataset"] = dataset_name
    res["label"] = "Adaptive-Gap"
    results.append(res)

    # Adaptive lambda MMR (entropy)
    print("  Running MMR-Adaptive (entropy)...")
    res = evaluate_mmr_adaptive(
        retriever, xb, queries, k=k, topN=topN,
        lambda_cfg=adaptive_cfg_entropy, gt_neighbors=gt_neighbors, impl="incremental"
    )
    res["dataset"] = dataset_name
    res["label"] = "Adaptive-Entropy"
    results.append(res)

    # 获取lambda分布
    search_res = retriever.search(queries, topN)
    lambdas_gap = batch_adaptive_lambdas(search_res.scores, adaptive_cfg_gap)
    lambdas_entropy = batch_adaptive_lambdas(search_res.scores, adaptive_cfg_entropy)

    df = pd.DataFrame(results)

    # 计算综合指标
    df["f1_recall_ild"] = df.apply(lambda r: compute_f1_score(r["recall"], r["ild"]), axis=1)
    df["harmonic_mean"] = df.apply(
        lambda r: compute_harmonic_mean(r["recall"], r["rel_mean_cos"], r["ild"]), axis=1
    )

    ctx = {
        "lambdas_gap": lambdas_gap,
        "lambdas_entropy": lambdas_entropy,
    }

    return df, ctx


def plot_multi_dataset_tradeoff(all_results: pd.DataFrame, out_dir: Path):
    """图1: 各数据集的 Recall-Diversity 曲线"""

    datasets = all_results["dataset"].unique()
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        df = all_results[all_results["dataset"] == dataset]

        # Fixed MMR 曲线
        fixed = df[df["method"] == "mmr"].sort_values("lambda")
        ax.plot(fixed["ild"], fixed["recall"], "o-", color="steelblue",
                markersize=8, linewidth=2, label="MMR (fixed λ)")

        # 标注部分 lambda 值
        for _, row in fixed.iterrows():
            if row["lambda"] in [0.5, 0.8, 0.95]:
                ax.annotate(f'λ={row["lambda"]}',
                           (row["ild"], row["recall"]),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)

        # Baseline
        baseline = df[df["method"] == "baseline"]
        ax.scatter(baseline["ild"], baseline["recall"],
                   marker="s", s=100, color="gray", label="Baseline", zorder=3)

        # Adaptive
        adaptive = df[df["method"] == "mmr_adaptive"]
        colors = {"Adaptive-Gap": "#e74c3c", "Adaptive-Entropy": "#27ae60"}
        markers = {"Adaptive-Gap": "*", "Adaptive-Entropy": "D"}

        for _, row in adaptive.iterrows():
            ax.scatter(row["ild"], row["recall"],
                       marker=markers[row["label"]], s=200,
                       color=colors[row["label"]],
                       edgecolors="black", linewidth=1.5,
                       label=row["label"], zorder=4)

        ax.set_xlabel("Diversity (ILD)", fontsize=11)
        ax.set_ylabel("Recall@k", fontsize=11)
        ax.set_title(f"{dataset}", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Recall-Diversity Trade-off Across Datasets", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = out_dir / "multi_dataset_tradeoff.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_improvement_analysis(all_results: pd.DataFrame, out_dir: Path):
    """图2: Adaptive 相对于最优 Fixed λ 的改进"""

    datasets = all_results["dataset"].unique()

    # 计算每个数据集的改进
    improvements = []

    for dataset in datasets:
        df = all_results[all_results["dataset"] == dataset]

        # 找到最优的 fixed lambda（基于 F1 分数）
        fixed = df[df["method"] == "mmr"]
        best_fixed_idx = fixed["f1_recall_ild"].idxmax()
        best_fixed = fixed.loc[best_fixed_idx]

        # Adaptive 结果
        adaptive_gap = df[df["label"] == "Adaptive-Gap"].iloc[0]
        adaptive_entropy = df[df["label"] == "Adaptive-Entropy"].iloc[0]

        improvements.append({
            "dataset": dataset,
            "best_fixed_lambda": best_fixed["lambda"],
            "best_fixed_f1": best_fixed["f1_recall_ild"],
            "best_fixed_recall": best_fixed["recall"],
            "best_fixed_ild": best_fixed["ild"],
            "gap_f1": adaptive_gap["f1_recall_ild"],
            "gap_recall": adaptive_gap["recall"],
            "gap_ild": adaptive_gap["ild"],
            "gap_improvement": (adaptive_gap["f1_recall_ild"] - best_fixed["f1_recall_ild"]) / best_fixed["f1_recall_ild"] * 100,
            "entropy_f1": adaptive_entropy["f1_recall_ild"],
            "entropy_recall": adaptive_entropy["recall"],
            "entropy_ild": adaptive_entropy["ild"],
            "entropy_improvement": (adaptive_entropy["f1_recall_ild"] - best_fixed["f1_recall_ild"]) / best_fixed["f1_recall_ild"] * 100,
        })

    imp_df = pd.DataFrame(improvements)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：F1分数对比
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.25

    bars1 = ax.bar(x - width, imp_df["best_fixed_f1"], width, label="Best Fixed λ", color="steelblue")
    bars2 = ax.bar(x, imp_df["gap_f1"], width, label="Adaptive-Gap", color="#e74c3c")
    bars3 = ax.bar(x + width, imp_df["entropy_f1"], width, label="Adaptive-Entropy", color="#27ae60")

    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("F1 Score (Recall × ILD)", fontsize=11)
    ax.set_title("Comprehensive Performance (F1 Score)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 标注最优fixed的lambda值
    for i, (_, row) in enumerate(imp_df.iterrows()):
        ax.annotate(f'λ={row["best_fixed_lambda"]}',
                   (i - width, row["best_fixed_f1"]),
                   textcoords="offset points", xytext=(0, 5),
                   ha="center", fontsize=8)

    # 右图：相对改进百分比
    ax = axes[1]
    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width/2, imp_df["gap_improvement"], width,
                   label="Adaptive-Gap", color="#e74c3c")
    bars2 = ax.bar(x + width/2, imp_df["entropy_improvement"], width,
                   label="Adaptive-Entropy", color="#27ae60")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("Improvement over Best Fixed λ (%)", fontsize=11)
    ax.set_title("Relative Improvement", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 标注改进值
    for i, (bars, col) in enumerate([(bars1, "gap_improvement"), (bars2, "entropy_improvement")]):
        for j, bar in enumerate(bars):
            val = imp_df.iloc[j][col]
            color = "green" if val > 0 else "red"
            ax.annotate(f'{val:+.1f}%',
                       (bar.get_x() + bar.get_width()/2, bar.get_height()),
                       textcoords="offset points",
                       xytext=(0, 5 if val >= 0 else -15),
                       ha="center", fontsize=9, color=color, fontweight="bold")

    plt.suptitle("Adaptive λ Performance vs Best Fixed λ", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = out_dir / "multi_dataset_improvement.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

    return imp_df


def plot_lambda_distributions(all_contexts: Dict[str, Dict], out_dir: Path):
    """图3: 各数据集的 lambda 分布对比"""

    datasets = list(all_contexts.keys())
    n_datasets = len(datasets)

    fig, axes = plt.subplots(2, n_datasets, figsize=(5 * n_datasets, 8))
    if n_datasets == 1:
        axes = axes.reshape(2, 1)

    for col, dataset in enumerate(datasets):
        ctx = all_contexts[dataset]

        # Gap strategy
        ax = axes[0, col]
        ax.hist(ctx["lambdas_gap"], bins=20, color="#e74c3c", alpha=0.7,
                edgecolor="black", linewidth=0.5)
        mean_val = np.mean(ctx["lambdas_gap"])
        std_val = np.std(ctx["lambdas_gap"])
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=2)
        ax.set_xlabel("λ value", fontsize=10)
        ax.set_ylabel("Query count", fontsize=10)
        ax.set_title(f"{dataset}\nGap Strategy (μ={mean_val:.2f}, σ={std_val:.2f})", fontsize=11)
        ax.grid(True, alpha=0.3)

        # Entropy strategy
        ax = axes[1, col]
        ax.hist(ctx["lambdas_entropy"], bins=20, color="#27ae60", alpha=0.7,
                edgecolor="black", linewidth=0.5)
        mean_val = np.mean(ctx["lambdas_entropy"])
        std_val = np.std(ctx["lambdas_entropy"])
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=2)
        ax.set_xlabel("λ value", fontsize=10)
        ax.set_ylabel("Query count", fontsize=10)
        ax.set_title(f"Entropy Strategy (μ={mean_val:.2f}, σ={std_val:.2f})", fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Adaptive λ Distribution Across Datasets", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = out_dir / "lambda_distribution_all.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_summary_table(all_results: pd.DataFrame, imp_df: pd.DataFrame, out_dir: Path):
    """图4: 综合指标汇总表"""

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    # 准备表格数据
    datasets = all_results["dataset"].unique()

    table_data = []
    for dataset in datasets:
        df = all_results[all_results["dataset"] == dataset]

        baseline = df[df["method"] == "baseline"].iloc[0]
        fixed = df[df["method"] == "mmr"]
        best_fixed = fixed.loc[fixed["f1_recall_ild"].idxmax()]
        adaptive_gap = df[df["label"] == "Adaptive-Gap"].iloc[0]
        adaptive_entropy = df[df["label"] == "Adaptive-Entropy"].iloc[0]

        table_data.append([
            dataset,
            f'{baseline["recall"]:.3f}',
            f'{baseline["ild"]:.3f}',
            f'{best_fixed["lambda"]:.1f}',
            f'{best_fixed["recall"]:.3f}',
            f'{best_fixed["ild"]:.3f}',
            f'{best_fixed["f1_recall_ild"]:.3f}',
            f'{adaptive_gap["recall"]:.3f}',
            f'{adaptive_gap["ild"]:.3f}',
            f'{adaptive_gap["f1_recall_ild"]:.3f}',
            f'{adaptive_entropy["recall"]:.3f}',
            f'{adaptive_entropy["ild"]:.3f}',
            f'{adaptive_entropy["f1_recall_ild"]:.3f}',
        ])

    columns = [
        "Dataset",
        "Base\nRecall", "Base\nILD",
        "Best λ", "Fixed\nRecall", "Fixed\nILD", "Fixed\nF1",
        "Gap\nRecall", "Gap\nILD", "Gap\nF1",
        "Ent\nRecall", "Ent\nILD", "Ent\nF1",
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colColours=["#f0f0f0"] * len(columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # 高亮最佳F1分数
    for i, row in enumerate(table_data):
        f1_values = [float(row[6]), float(row[9]), float(row[12])]
        max_idx = np.argmax(f1_values)
        col_idx = [6, 9, 12][max_idx]
        table[(i + 1, col_idx)].set_facecolor("#90EE90")

    plt.title("Performance Summary Across Datasets", fontsize=14, pad=20)
    plt.tight_layout()
    out_path = out_dir / "multi_dataset_summary.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def generate_latex_table(all_results: pd.DataFrame, out_dir: Path):
    """生成 LaTeX 格式的表格（可直接用于论文）"""

    datasets = all_results["dataset"].unique()

    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Performance Comparison of Adaptive vs Fixed $\lambda$ across Datasets}",
        r"\label{tab:adaptive_comparison}",
        r"\begin{tabular}{l|ccc|ccc|ccc}",
        r"\hline",
        r"Dataset & \multicolumn{3}{c|}{Best Fixed $\lambda$} & \multicolumn{3}{c|}{Adaptive-Gap} & \multicolumn{3}{c}{Adaptive-Entropy} \\",
        r" & Recall & ILD & F1 & Recall & ILD & F1 & Recall & ILD & F1 \\",
        r"\hline",
    ]

    for dataset in datasets:
        df = all_results[all_results["dataset"] == dataset]

        fixed = df[df["method"] == "mmr"]
        best_fixed = fixed.loc[fixed["f1_recall_ild"].idxmax()]
        adaptive_gap = df[df["label"] == "Adaptive-Gap"].iloc[0]
        adaptive_entropy = df[df["label"] == "Adaptive-Entropy"].iloc[0]

        # 找出最佳F1
        f1_values = [best_fixed["f1_recall_ild"], adaptive_gap["f1_recall_ild"], adaptive_entropy["f1_recall_ild"]]
        max_f1_idx = np.argmax(f1_values)

        def fmt(val, is_best=False):
            if is_best:
                return f"\\textbf{{{val:.3f}}}"
            return f"{val:.3f}"

        line = f"{dataset} & "
        line += f'{fmt(best_fixed["recall"])} & {fmt(best_fixed["ild"])} & {fmt(best_fixed["f1_recall_ild"], max_f1_idx==0)} & '
        line += f'{fmt(adaptive_gap["recall"])} & {fmt(adaptive_gap["ild"])} & {fmt(adaptive_gap["f1_recall_ild"], max_f1_idx==1)} & '
        line += f'{fmt(adaptive_entropy["recall"])} & {fmt(adaptive_entropy["ild"])} & {fmt(adaptive_entropy["f1_recall_ild"], max_f1_idx==2)} \\\\'
        latex_lines.append(line)

    latex_lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex_content = "\n".join(latex_lines)
    out_path = out_dir / "results_table.tex"
    with open(out_path, "w") as f:
        f.write(latex_content)
    print(f"Saved: {out_path}")

    return latex_content


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset Adaptive Lambda Experiment")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="outputs/multi_dataset_analysis")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--topN", type=int, default=200)
    parser.add_argument("--max-train", type=int, default=50000)
    parser.add_argument("--max-test", type=int, default=500)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 定义要测试的数据集
    datasets = [
        ("glove-25-angular", data_dir / "glove-25-angular.hdf5"),
        ("glove-100-angular", data_dir / "glove-100-angular.hdf5"),
        ("nytimes-256-angular", data_dir / "nytimes-256-angular.hdf5"),
    ]

    # 过滤存在的数据集
    available_datasets = [(name, path) for name, path in datasets if path.exists()]

    if not available_datasets:
        print("No datasets found! Please download datasets first.")
        print("Available dataset files in data/:")
        for f in data_dir.glob("*.hdf5"):
            print(f"  {f.name}")
        return

    print("="*60)
    print("MULTI-DATASET ADAPTIVE LAMBDA EXPERIMENT")
    print("="*60)
    print(f"Datasets: {[name for name, _ in available_datasets]}")

    # 运行实验
    all_results = []
    all_contexts = {}

    for name, path in available_datasets:
        df, ctx = run_single_dataset_experiment(
            path, name,
            k=args.k, topN=args.topN,
            max_train=args.max_train, max_test=args.max_test
        )
        all_results.append(df)
        all_contexts[name] = ctx

    all_results_df = pd.concat(all_results, ignore_index=True)

    # 保存原始结果
    all_results_df.to_csv(out_dir / "all_results.csv", index=False)
    print(f"\nSaved: {out_dir / 'all_results.csv'}")

    # 生成可视化
    print("\nGenerating visualizations...")
    plot_multi_dataset_tradeoff(all_results_df, out_dir)
    imp_df = plot_improvement_analysis(all_results_df, out_dir)
    plot_lambda_distributions(all_contexts, out_dir)
    plot_summary_table(all_results_df, imp_df, out_dir)

    # 生成LaTeX表格
    latex = generate_latex_table(all_results_df, out_dir)

    # 打印总结
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    for _, row in imp_df.iterrows():
        print(f"\n{row['dataset']}:")
        print(f"  Best Fixed λ={row['best_fixed_lambda']}: F1={row['best_fixed_f1']:.4f}")
        print(f"  Adaptive-Gap: F1={row['gap_f1']:.4f} ({row['gap_improvement']:+.2f}%)")
        print(f"  Adaptive-Entropy: F1={row['entropy_f1']:.4f} ({row['entropy_improvement']:+.2f}%)")

    print("\n" + "="*60)
    print(f"All outputs saved to: {out_dir}")
    print("="*60)

    # 关键发现
    print("\n📊 KEY FINDINGS:")
    avg_gap_imp = imp_df["gap_improvement"].mean()
    avg_entropy_imp = imp_df["entropy_improvement"].mean()
    print(f"  Average improvement (Gap strategy): {avg_gap_imp:+.2f}%")
    print(f"  Average improvement (Entropy strategy): {avg_entropy_imp:+.2f}%")

    if avg_gap_imp > 0 or avg_entropy_imp > 0:
        better_strategy = "Gap" if avg_gap_imp > avg_entropy_imp else "Entropy"
        print(f"  → Adaptive-{better_strategy} shows consistent improvement!")
    else:
        print("  → Consider tuning adaptive lambda parameters for better results")


if __name__ == "__main__":
    main()
