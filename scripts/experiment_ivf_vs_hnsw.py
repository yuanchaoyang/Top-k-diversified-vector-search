#!/usr/bin/env python3
"""
IVF vs HNSW 性能对比实验

对比两种纯 NumPy ANN 索引在相同数据集上的表现：
1. 构建时间
2. 不同参数下的 recall@topN（以暴力搜索为 ground truth）
3. 搜索延迟（ms/query）
4. 加速比（vs 暴力搜索）

使用方法:
    PYTHONPATH=src python scripts/experiment_ivf_vs_hnsw.py
    PYTHONPATH=src python scripts/experiment_ivf_vs_hnsw.py --out-dir outputs/ivf_vs_hnsw
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diverse_search.index import NumpyBruteForceRetriever
from diverse_search.ivf import NumpyIVFRetriever
from diverse_search.hnsw import NumpyHNSWRetriever


def run_experiment(
    xb: np.ndarray,
    queries: np.ndarray,
    topN: int = 200,
    nlist: int = 128,
    hnsw_M: int = 16,
    ef_construction: int = 200,
) -> dict:
    """运行完整对比实验。"""
    n, d = xb.shape
    nq = queries.shape[0]

    # ── Brute-force ground truth ──
    print(f"Brute-force ({nq} queries, topN={topN})...")
    brute = NumpyBruteForceRetriever(xb)
    t0 = time.perf_counter()
    brute_res = brute.search(queries, topN)
    brute_ms_per_q = (time.perf_counter() - t0) * 1000 / nq
    gt_ids = [set(brute_res.indices[i].tolist()) for i in range(nq)]
    print(f"  {brute_ms_per_q:.3f} ms/query\n")

    # ── Build IVF ──
    print(f"Building IVF (nlist={nlist})...")
    t0 = time.perf_counter()
    ivf = NumpyIVFRetriever(xb, nlist=nlist, nprobe=1)
    ivf_build_ms = (time.perf_counter() - t0) * 1000
    print(f"  Build: {ivf_build_ms:.0f} ms")

    # 簇大小分布
    sizes = np.array([len(lst) for lst in ivf.inverted_lists])
    ivf_cluster_cv = float(np.std(sizes) / (np.mean(sizes) + 1e-8))
    print(f"  Cluster CV: {ivf_cluster_cv:.3f}, "
          f"min={np.min(sizes)}, max={np.max(sizes)}\n")

    # ── Build HNSW ──
    print(f"Building HNSW (M={hnsw_M}, ef_construction={ef_construction})...")
    t0 = time.perf_counter()
    hnsw = NumpyHNSWRetriever(
        xb, M=hnsw_M, ef_construction=ef_construction, ef_search=64,
    )
    hnsw_build_ms = (time.perf_counter() - t0) * 1000
    n_edges = sum(
        sum(len(neighbors) for neighbors in layer.values())
        for layer in hnsw.graph
    )
    print(f"  Build: {hnsw_build_ms:.0f} ms")
    print(f"  Layers: {hnsw.max_layer + 1}, total edges: {n_edges}\n")

    # ── 参数扫描 ──
    # IVF: 扫描 nprobe
    nprobe_values = [1, 2, 4, 8, 16, 32, 48, 64]
    # HNSW: 扫描 ef_search
    ef_values = [16, 32, 64, 100, 150, 200, 300, 400]

    results = {
        "meta": {
            "n_vectors": n,
            "dim": d,
            "n_queries": nq,
            "topN": topN,
            "brute_ms_per_q": round(brute_ms_per_q, 3),
            "ivf_build_ms": round(ivf_build_ms, 0),
            "hnsw_build_ms": round(hnsw_build_ms, 0),
            "ivf_nlist": nlist,
            "ivf_cluster_cv": round(ivf_cluster_cv, 3),
            "hnsw_M": hnsw_M,
            "hnsw_ef_construction": ef_construction,
            "hnsw_layers": hnsw.max_layer + 1,
            "hnsw_edges": n_edges,
        },
        "ivf": [],
        "hnsw": [],
    }

    def compute_recall(res, gt, nq, topN):
        recalls = []
        for i in range(nq):
            found = set(res.indices[i].tolist())
            found.discard(-1)
            recalls.append(len(gt[i] & found) / topN)
        return float(np.mean(recalls))

    # IVF sweep
    print("=== IVF nprobe sweep ===")
    print(f"{'nprobe':>8} {'probe%':>7} {'recall':>8} {'ms/q':>8} {'speedup':>8}")
    print("-" * 45)
    for nprobe in nprobe_values:
        if nprobe > nlist:
            continue
        ivf.set_nprobe(nprobe)
        t0 = time.perf_counter()
        ivf_res = ivf.search(queries, topN)
        search_ms = (time.perf_counter() - t0) * 1000
        ms_per_q = search_ms / nq

        recall = compute_recall(ivf_res, gt_ids, nq, topN)
        speedup = brute_ms_per_q / ms_per_q if ms_per_q > 0 else 0
        probe_pct = 100.0 * nprobe / nlist

        results["ivf"].append({
            "param": nprobe,
            "param_pct": round(probe_pct, 1),
            "recall": round(recall, 4),
            "ms_per_query": round(ms_per_q, 3),
            "speedup": round(speedup, 2),
        })
        print(f"{nprobe:>8} {probe_pct:>6.1f}% {recall:>8.4f} {ms_per_q:>7.3f} {speedup:>7.2f}x")

    # HNSW sweep
    print(f"\n=== HNSW ef_search sweep ===")
    print(f"{'ef':>8} {'recall':>8} {'ms/q':>8} {'speedup':>8}")
    print("-" * 38)
    for ef in ef_values:
        hnsw.set_ef_search(ef)
        t0 = time.perf_counter()
        hnsw_res = hnsw.search(queries, topN)
        search_ms = (time.perf_counter() - t0) * 1000
        ms_per_q = search_ms / nq

        recall = compute_recall(hnsw_res, gt_ids, nq, topN)
        speedup = brute_ms_per_q / ms_per_q if ms_per_q > 0 else 0

        results["hnsw"].append({
            "param": ef,
            "recall": round(recall, 4),
            "ms_per_query": round(ms_per_q, 3),
            "speedup": round(speedup, 2),
        })
        print(f"{ef:>8} {recall:>8.4f} {ms_per_q:>7.3f} {speedup:>7.2f}x")

    return results


def print_summary(results: dict) -> None:
    """打印汇总对比表。"""
    meta = results["meta"]
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"Dataset: {meta['n_vectors']} vectors, dim={meta['dim']}")
    print(f"Queries: {meta['n_queries']}, topN={meta['topN']}")
    print(f"Brute-force: {meta['brute_ms_per_q']:.3f} ms/query")
    print()

    print(f"{'':>12} {'IVF':>20} {'HNSW':>20}")
    print(f"{'Build time':>12} {meta['ivf_build_ms']:>17.0f} ms {meta['hnsw_build_ms']:>17.0f} ms")
    print(f"{'Structure':>12} {'nlist=' + str(meta['ivf_nlist']):>20} {'M=' + str(meta['hnsw_M']) + ', L=' + str(meta['hnsw_layers']):>20}")
    print()

    # 找到 recall >= 95% 的最低参数配置
    def best_at_95(rows):
        good = [r for r in rows if r["recall"] >= 0.95]
        if good:
            return min(good, key=lambda r: r["ms_per_query"])
        return max(rows, key=lambda r: r["recall"])

    ivf_best = best_at_95(results["ivf"])
    hnsw_best = best_at_95(results["hnsw"])

    print("达到 recall≥95% 的最优配置:")
    print(f"  IVF:  nprobe={ivf_best['param']:>3}  "
          f"recall={ivf_best['recall']:.4f}  "
          f"{ivf_best['ms_per_query']:.3f} ms/q  "
          f"{ivf_best['speedup']:.2f}x")
    print(f"  HNSW: ef={hnsw_best['param']:>5}  "
          f"recall={hnsw_best['recall']:.4f}  "
          f"{hnsw_best['ms_per_query']:.3f} ms/q  "
          f"{hnsw_best['speedup']:.2f}x")


def save_results(results: dict, out_dir: Path) -> None:
    """保存 CSV + 可视化图。"""
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV — IVF
    ivf_csv = out_dir / "ivf_sweep.csv"
    with open(ivf_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["param", "param_pct", "recall", "ms_per_query", "speedup"])
        w.writeheader()
        w.writerows(results["ivf"])
    print(f"\nSaved: {ivf_csv}")

    # CSV — HNSW
    hnsw_csv = out_dir / "hnsw_sweep.csv"
    with open(hnsw_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["param", "recall", "ms_per_query", "speedup"])
        w.writeheader()
        w.writerows(results["hnsw"])
    print(f"Saved: {hnsw_csv}")

    # Plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: Recall vs Parameter
    ax = axes[0]
    ivf_params = [r["param"] for r in results["ivf"]]
    ivf_recalls = [r["recall"] for r in results["ivf"]]
    hnsw_params = [r["param"] for r in results["hnsw"]]
    hnsw_recalls = [r["recall"] for r in results["hnsw"]]

    ax.plot(ivf_params, ivf_recalls, "o-", color="tab:blue", label="IVF (nprobe)")
    ax.plot(hnsw_params, hnsw_recalls, "s-", color="tab:orange", label="HNSW (ef_search)")
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% recall")
    ax.set_xlabel("Parameter value (nprobe / ef_search)")
    ax.set_ylabel("Recall@topN")
    ax.set_title("Recall vs Search Parameter")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图: Recall vs Speed (效率前沿)
    ax = axes[1]
    ivf_ms = [r["ms_per_query"] for r in results["ivf"]]
    hnsw_ms = [r["ms_per_query"] for r in results["hnsw"]]

    ax.plot(ivf_recalls, ivf_ms, "o-", color="tab:blue", label="IVF")
    ax.plot(hnsw_recalls, hnsw_ms, "s-", color="tab:orange", label="HNSW")
    ax.axvline(x=0.95, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Recall@topN")
    ax.set_ylabel("Latency (ms/query)")
    ax.set_title("Recall-Latency Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = out_dir / "ivf_vs_hnsw.png"
    plt.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="IVF vs HNSW 性能对比实验")
    parser.add_argument("--data-dir", default="data/improved",
                        help="语料库目录 (含 passage_embeddings.npy)")
    parser.add_argument("--nlist", type=int, default=128, help="IVF nlist")
    parser.add_argument("--hnsw-m", type=int, default=16, help="HNSW M")
    parser.add_argument("--ef-construction", type=int, default=200,
                        help="HNSW ef_construction")
    parser.add_argument("--topN", type=int, default=200)
    parser.add_argument("--n-queries", type=int, default=50,
                        help="随机采样的查询数")
    parser.add_argument("--out-dir", default="outputs/ivf_vs_hnsw")
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

    # 随机采样查询
    rng = np.random.default_rng(args.seed)
    n_queries = min(args.n_queries, xb.shape[0])
    query_idx = rng.choice(xb.shape[0], size=n_queries, replace=False)
    queries = xb[query_idx]
    print(f"  {n_queries} random queries sampled\n")

    results = run_experiment(
        xb, queries,
        topN=args.topN,
        nlist=args.nlist,
        hnsw_M=args.hnsw_m,
        ef_construction=args.ef_construction,
    )

    print_summary(results)
    save_results(results, Path(args.out_dir))


if __name__ == "__main__":
    main()
