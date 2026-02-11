#!/usr/bin/env python3
"""
nlist 参数敏感性实验

扫描不同的 nlist 值，对比：
1. 构建耗时（K-Means 训练）
2. 簇大小均匀度（std / mean）
3. 不同 nprobe 下的 recall@topN
4. 检索延迟

使用方法:
    PYTHONPATH=src python scripts/experiment_nlist.py
    PYTHONPATH=src python scripts/experiment_nlist.py --out-dir outputs/nlist_experiment
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diverse_search.index import NumpyIVFRetriever, NumpyBruteForceRetriever


def run_experiment(
    xb: np.ndarray,
    queries: np.ndarray,
    nlist_values: list[int],
    nprobe_values: list[int],
    topN: int = 200,
) -> list[dict]:
    """对每个 nlist 值运行完整实验。"""
    n, d = xb.shape
    nq = queries.shape[0]

    # 先算暴力搜索的 ground truth
    print(f"Computing brute-force ground truth ({nq} queries, topN={topN})...")
    brute = NumpyBruteForceRetriever(xb)
    t0 = time.perf_counter()
    brute_res = brute.search(queries, topN)
    brute_ms_per_q = (time.perf_counter() - t0) * 1000 / nq
    print(f"  Brute-force: {brute_ms_per_q:.3f} ms/query\n")

    gt_ids = [set(brute_res.indices[i].tolist()) for i in range(nq)]

    results = []

    for nlist in nlist_values:
        if nlist >= n:
            print(f"nlist={nlist} >= N={n}, skipping")
            continue

        print(f"=== nlist={nlist} ===")

        # 构建 IVF 索引
        t0 = time.perf_counter()
        ivf = NumpyIVFRetriever(xb, nlist=nlist, nprobe=1)
        build_ms = (time.perf_counter() - t0) * 1000
        print(f"  Build: {build_ms:.0f} ms")

        # 簇大小分布
        sizes = np.array([len(lst) for lst in ivf.inverted_lists])
        size_mean = float(np.mean(sizes))
        size_std = float(np.std(sizes))
        size_min = int(np.min(sizes))
        size_max = int(np.max(sizes))
        # 变异系数 = std/mean, 越小越均匀
        cv = size_std / size_mean if size_mean > 0 else float("inf")
        print(f"  Cluster sizes: mean={size_mean:.0f}, std={size_std:.0f}, "
              f"min={size_min}, max={size_max}, CV={cv:.3f}")

        # 不同 nprobe 的召回率和延迟
        for nprobe in nprobe_values:
            if nprobe > nlist:
                continue

            ivf.set_nprobe(nprobe)

            t0 = time.perf_counter()
            ivf_res = ivf.search(queries, topN)
            search_ms = (time.perf_counter() - t0) * 1000
            ms_per_q = search_ms / nq

            # 计算平均 recall
            recalls = []
            for i in range(nq):
                ivf_ids = set(ivf_res.indices[i].tolist())
                ivf_ids.discard(-1)  # 丢弃无效 id
                overlap = len(gt_ids[i] & ivf_ids)
                recalls.append(overlap / topN)
            avg_recall = float(np.mean(recalls))

            probe_pct = 100.0 * nprobe / nlist
            speedup = brute_ms_per_q / ms_per_q if ms_per_q > 0 else 0

            results.append({
                "nlist": nlist,
                "nprobe": nprobe,
                "probe_pct": round(probe_pct, 1),
                "build_ms": round(build_ms, 0),
                "recall": round(avg_recall, 4),
                "ms_per_query": round(ms_per_q, 3),
                "speedup": round(speedup, 2),
                "cluster_cv": round(cv, 3),
                "cluster_min": size_min,
                "cluster_max": size_max,
            })

            print(f"  nprobe={nprobe:>3} ({probe_pct:>5.1f}%)  "
                  f"recall={avg_recall:.3f}  "
                  f"{ms_per_q:.3f} ms/q  "
                  f"speedup={speedup:.1f}x")
        print()

    return results


def print_summary_table(results: list[dict]) -> None:
    """打印汇总表：每个 nlist 取 recall>=95% 的最小 nprobe。"""
    print("\n" + "=" * 70)
    print("Summary: 达到 recall>=95% 所需的最小 nprobe")
    print("=" * 70)
    print(f"{'nlist':>6} {'nprobe':>6} {'probe%':>7} {'recall':>7} "
          f"{'ms/q':>8} {'speedup':>8} {'build_ms':>9} {'CV':>6}")
    print("-" * 70)

    nlist_set = sorted(set(r["nlist"] for r in results))
    for nlist in nlist_set:
        rows = [r for r in results if r["nlist"] == nlist]
        # 找 recall >= 0.95 的最小 nprobe
        good = [r for r in rows if r["recall"] >= 0.95]
        if good:
            best = min(good, key=lambda r: r["nprobe"])
        else:
            # 达不到 95%，取最高 recall
            best = max(rows, key=lambda r: r["recall"])
        print(f"{best['nlist']:>6} {best['nprobe']:>6} {best['probe_pct']:>6.1f}% "
              f"{best['recall']:>7.3f} {best['ms_per_query']:>7.3f} "
              f"{best['speedup']:>7.1f}x {best['build_ms']:>8.0f} "
              f"{best['cluster_cv']:>6.3f}")


def save_results(results: list[dict], out_dir: Path) -> None:
    """保存结果到 CSV 和可视化图。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    import csv
    csv_path = out_dir / "nlist_experiment.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    nlist_set = sorted(set(r["nlist"] for r in results))
    colors = plt.cm.tab10(np.linspace(0, 1, len(nlist_set)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: nprobe vs recall (每条线一个 nlist)
    ax = axes[0]
    for nlist, color in zip(nlist_set, colors):
        rows = sorted([r for r in results if r["nlist"] == nlist],
                       key=lambda r: r["nprobe"])
        nprobes = [r["nprobe"] for r in rows]
        recalls = [r["recall"] for r in rows]
        ax.plot(nprobes, recalls, "o-", color=color, label=f"nlist={nlist}")
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% recall")
    ax.set_xlabel("nprobe")
    ax.set_ylabel("Recall@topN")
    ax.set_title("Recall vs nprobe")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 右图: recall vs speedup (效率前沿)
    ax = axes[1]
    for nlist, color in zip(nlist_set, colors):
        rows = [r for r in results if r["nlist"] == nlist]
        recalls = [r["recall"] for r in rows]
        speedups = [r["speedup"] for r in rows]
        ax.scatter(recalls, speedups, color=color, label=f"nlist={nlist}", s=40)
    ax.axvline(x=0.95, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Recall@topN")
    ax.set_ylabel("Speedup vs brute-force")
    ax.set_title("Recall-Speedup Trade-off")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = out_dir / "nlist_experiment.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="nlist 参数敏感性实验")
    parser.add_argument("--data-dir", default="data/improved",
                        help="语料库目录 (含 passage_embeddings.npy)")
    parser.add_argument("--nlist", default="32,64,128,256,512",
                        help="要测试的 nlist 值 (逗号分隔)")
    parser.add_argument("--nprobe", default="1,2,4,8,16,32,64",
                        help="要测试的 nprobe 值 (逗号分隔)")
    parser.add_argument("--topN", type=int, default=200)
    parser.add_argument("--n-queries", type=int, default=100,
                        help="随机采样的查询数")
    parser.add_argument("--out-dir", default="outputs/nlist_experiment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 加载嵌入
    emb_path = Path(args.data_dir) / "passage_embeddings.npy"
    if not emb_path.exists():
        print(f"Error: {emb_path} not found")
        sys.exit(1)

    print(f"Loading embeddings from {emb_path}...")
    xb = np.load(emb_path).astype(np.float32)
    norms = np.linalg.norm(xb, axis=1, keepdims=True)
    xb = xb / (norms + 1e-8)
    print(f"  {xb.shape[0]} vectors, dim={xb.shape[1]}")

    # 随机采样查询（从语料库中取，模拟真实查询）
    rng = np.random.default_rng(args.seed)
    n_queries = min(args.n_queries, xb.shape[0])
    query_idx = rng.choice(xb.shape[0], size=n_queries, replace=False)
    queries = xb[query_idx]
    print(f"  {n_queries} random queries sampled\n")

    nlist_values = [int(x) for x in args.nlist.split(",")]
    nprobe_values = [int(x) for x in args.nprobe.split(",")]

    results = run_experiment(xb, queries, nlist_values, nprobe_values, args.topN)

    print_summary_table(results)
    save_results(results, Path(args.out_dir))


if __name__ == "__main__":
    main()
