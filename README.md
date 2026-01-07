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
