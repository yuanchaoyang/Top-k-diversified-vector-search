# Top-k Diversified Vector Search (Cosine / Geometric Diversity)

This is a small, graduation-project-friendly codebase for **Top-k diversified vector search**.

Goal:
- Retrieve **k** items that are **relevant to the query** (high cosine similarity)
- While also being **diverse among themselves** (low cosine similarity between returned items)

We implement a practical pipeline:

1) **Candidate retrieval**: get top-N nearest neighbors (exact via NumPy; optional ANN via FAISS)
2) **Diversified rerank**: select final top-k with **MMR-style greedy reranking** (plus other baselines)

The project supports running experiments on ANN-Benchmarks datasets (HDF5), which include
`train` vectors, `test` query vectors, and `neighbors` ground-truth.

---

## Project layout

```text
src/diverse_search/
  __init__.py
  datasets.py        # download & load HDF5 datasets
  index.py           # candidate retrieval (NumPy brute force, optional FAISS)
  mmr.py             # diversified reranking (MMR)
  diversify.py       # threshold / maxmin rerankers
  metrics.py         # recall & diversity metrics
  experiment.py      # evaluation pipeline + sweeps
  run_experiments.py # CLI for experiments
  cli_text.py        # interactive text demo
  cli_ann.py         # interactive ANN dataset demo
data/
  corpus_words.txt
outputs/
requirements.txt

## Query-adaptive λ for MMR

- Add `mmr_adaptive` to `--methods` to let the reranker pick λ(q) per query.
- Strategy (set with `--adaptive-lambda-strategy`):
  - `gap_piecewise` (default): gap = s1 − s_k (k=10 by default). If gap is large → λ≈λ_max; small → λ≈λ_min; mid-gap → λ_mid (defaults: 0.6 / 0.8 / 0.95).
  - `entropy`: original softmax-entropy heuristic (peaked → higher λ, flat → lower λ).
- Useful flags:
  - `--adaptive-lambda-range min,max` (default `0.6,0.95`; tune as needed)
  - `--adaptive-gap-k`, `--adaptive-gap-low`, `--adaptive-gap-high`, `--adaptive-gap-lambda-mid`
  - `--adaptive-lambda-temperature` / `--adaptive-lambda-entropy-power` / `--adaptive-lambda-topM` (entropy strategy)
- Keep a strong fixed baseline (`--lambdas 0.8`) for comparison.
- Example (gap heuristic):
  ```
  python -m diverse_search.run_experiments \
    --dataset glove-100-angular --k 10 --topN 200 \
    --methods baseline mmr mmr_adaptive \
    --lambdas 0.8 \
    --adaptive-lambda-strategy gap_piecewise \
    --adaptive-lambda-range 0.6,0.95 \
    --adaptive-gap-low 0.02 --adaptive-gap-high 0.08 --adaptive-gap-lambda-mid 0.8 \
    --out-dir outputs/dayX
  ```
