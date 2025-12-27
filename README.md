# Top-k Diversified Vector Search (Cosine / Geometric Diversity)

This is a small, graduation-project-friendly codebase for **Top-k diversified vector search**.

Goal:
- Retrieve **k** vectors that are **relevant to the query** (high cosine similarity)
- While also being **diverse among themselves** (low cosine similarity between returned items)

We implement a practical and commonly-used pipeline:

1) **Candidate retrieval**: get top-N nearest neighbors (exact via NumPy or ANN via FAISS)
2) **Diversified rerank**: select final top-k with **MMR-style greedy reranking**

The project supports running experiments on ANN-Benchmarks datasets (HDF5), which include
`train` vectors, `test` query vectors, and `neighbors` ground-truth.

---

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# Optional (recommended): FAISS for faster retrieval
# pip install faiss-cpu
```

### 2) Download a dataset (ANN-Benchmarks)

Example (cosine / angular):

```bash
python scripts/download_dataset.py --dataset glove-100-angular --out data/glove-100-angular.hdf5
```

Other useful datasets:
- `nytimes-256-angular` (smaller)
- `sift-128-euclidean` (Euclidean; useful if you want to also study L2)

### 3) Run an experiment

Run a small experiment on a subset (so it finishes quickly on a laptop):

```bash
python scripts/run_experiment.py \
  --dataset data/glove-100-angular.hdf5 \
  --backend numpy \
  --k 10 --topN 100 \
  --lambdas 0.2 0.4 0.6 0.8 \
  --max_train 200000 --max_test 1000 \
  --out_dir outputs/demo
```

If you have FAISS installed:

```bash
python scripts/run_experiment.py \
  --dataset data/glove-100-angular.hdf5 \
  --backend faiss --index hnsw \
  --k 10 --topN 100 \
  --lambdas 0.2 0.4 0.6 0.8 \
  --max_train 500000 --max_test 2000 \
  --out_dir outputs/faiss_hnsw
```

Outputs:
- `results.csv`: per-lambda aggregated metrics
- `tradeoff_recall_vs_ild.png`: recall vs intra-list diversity

---

## Key metrics

- **Recall@k**: uses ground truth `neighbors` from ANN-Benchmarks (if present)
- **Relevance proxy**: mean cosine(query, selected)
- **Diversity**:
  - `AvgPairwiseCos`: average pairwise cosine similarity within top-k (lower is more diverse)
  - `ILD`: 1 - AvgPairwiseCos (higher is more diverse)
  - `MaxPairwiseCos`: the most similar pair inside top-k (lower is better)

---

## Project layout

```text
src/diverse_search/
  datasets.py      # download & load HDF5 datasets
  index.py         # candidate retrieval (NumPy brute force, optional FAISS)
  mmr.py           # diversified reranking (MMR)
  metrics.py       # recall & diversity metrics
  experiment.py    # evaluation pipeline + lambda sweep
scripts/
  download_dataset.py
  run_experiment.py
```

---

## Notes

- For cosine similarity retrieval, we **L2-normalize vectors** and then use inner product.
- MMR reranking runs on a small candidate set (topN), so O(N^2) pairwise similarity is fine.

