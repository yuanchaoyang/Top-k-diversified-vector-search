# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Top-k diversified vector search: retrieve k items that are relevant to a query (high cosine similarity) while being diverse among themselves (low inter-item similarity). Uses a two-stage pipeline: candidate retrieval (top-N nearest neighbors) → diversified reranking (select final k).

## Common Commands

All commands require `PYTHONPATH=src` or run from `scripts/`. Uses Python venv at `.venv/`.

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install faiss-cpu  # optional ANN backend

# Download datasets (available: glove-25-angular, glove-50-angular, glove-100-angular,
#   nytimes-256-angular, lastfm-64-dot, sift-128-euclidean)
python scripts/download_dataset.py --dataset glove-100-angular --out data/glove-100-angular.hdf5

# Run experiments (generates outputs/results.csv and outputs/tradeoff.png)
PYTHONPATH=src python -m diverse_search.run_experiments \
  --dataset glove-100-angular --k 10 --topN 200 \
  --methods baseline mmr mmr_adaptive threshold maxmin \
  --lambdas 0.8 --out-dir outputs/dayX

# Interactive demos
PYTHONPATH=src python -m diverse_search.cli_text
PYTHONPATH=src python -m diverse_search.cli_ann --dataset data/glove-100-angular.hdf5

# Web demo (requires fastapi, uvicorn)
python demo/run_demo.py  # http://localhost:8000 — word embedding search
python demo/search_api.py  # http://localhost:8000 — MS MARCO passage search

# Adaptive lambda visualization
PYTHONPATH=src python scripts/visualize_adaptive.py --dataset glove-100-angular
PYTHONPATH=src python scripts/multi_dataset_experiment.py

# Train intent classifier (requires scikit-learn, sentence-transformers)
python scripts/train_diverse_intent.py  # merges all label sources → models/intent_v2/

# Evaluate intent model vs baselines (generates tables, plots, statistical tests)
PYTHONPATH=src python scripts/evaluate_intent_model.py \
  --words-file data/corpus_words.txt --embeddings-file data/corpus_words_embeddings.npy \
  --out-dir outputs/evaluation

# Build improved corpus (Wikipedia + Wikidata + WordNet + disambiguation pages)
python scripts/build_improved_corpus.py --output data/improved

# Build mixed corpus: Wikipedia + FineWeb + MS MARCO (requires datasets, sentence-transformers)
python scripts/prepare_mixed_corpus.py --n-wiki 25000 --n-web 15000 --n-msmarco 20000

# MS MARCO real-query evaluation (requires datasets, sentence-transformers)
python scripts/prepare_msmarco.py --n-passages 50000 --n-queries 1000
PYTHONPATH=src python scripts/evaluate_msmarco.py --data-dir data/msmarco --k 10 --topN 100

# Benchmark IVF vs HNSW retrieval (recall vs latency sweep)
PYTHONPATH=src python scripts/experiment_ivf_vs_hnsw.py --out-dir outputs/ivf_vs_hnsw

# Ablation: compare all-IVF vs all-HNSW vs dynamic switching, sweep intent thresholds
PYTHONPATH=src python scripts/experiment_ablation_index.py --data-dir data/msmarco --out-dir outputs/ablation
# With mixed queries (ambiguous words + MS MARCO):
PYTHONPATH=src python scripts/experiment_ablation_index.py --mixed-queries --n-ambiguous 100 --n-clear 100

# Quantitative ablation — 36 baseline grid + L1-L5 layered ablation (paper main results)
PYTHONPATH=src python scripts/experiment_quantitative.py \
  --msmarco-dir data/msmarco --improved-dir data/improved --k 10 --out-dir outputs/quant_v6

# Case study — side-by-side passage comparison for representative queries
PYTHONPATH=src python scripts/experiment_case_study.py \
  --msmarco-dir data/msmarco --improved-dir data/improved --k 5 --out-dir outputs/case_study

# Score comparison — ambiguous vs specific performance bar charts
PYTHONPATH=src python scripts/experiment_score_comparison.py \
  --improved-dir data/improved --k 10 --out-dir outputs/score_comparison

# Chapter 7 unified experiment — all paper tables + qualitative cases
PYTHONPATH=src python scripts/experiment_chapter7.py \
  --msmarco-dir data/msmarco --improved-dir data/improved --k 10 --out-dir outputs/chapter7

# Adaptive pipeline showcase — adaptation dashboard, robustness, oracle regret, Pareto
PYTHONPATH=src python scripts/experiment_adaptive.py \
  --improved-dir data/improved --k 10 --out-dir outputs/adaptive
```

## Architecture

```
Query → Intent Analysis → Retriever Selection → Candidate Retrieval (top-N) → Adaptive Reranker → Final k results
         (ML intent +       (IVF for ambiguous,     (with adaptive        (MMR + temporal
          temporal +          HNSW for specific)       nprobe/ef)            freshness)
          factual rules)
```

### Retriever backends (`src/diverse_search/`)

`index.py` is the factory/entry module. It defines `BaseRetriever`, `SearchResult`, and `build_retriever()`, then re-exports all concrete retrievers. External code always imports from `index.py`.

| Module | Class | Algorithm | Complexity | Use case |
|--------|-------|-----------|------------|----------|
| `brute_force.py` | `NumpyBruteForceRetriever` | Exact dot-product scan | O(N·d) | Baseline, 100% recall |
| `ivf.py` | `NumpyIVFRetriever` | K-means clustering + inverted lists | O(nprobe/nlist·N·d) | Ambiguous queries (cluster structure promotes diversity). `search(..., cluster_balanced=True)` uses per-cluster quota sampling; `.assignments` array exposes per-vector cluster IDs for cluster-aware MMR reranking |
| `hnsw.py` | `NumpyHNSWRetriever` | Multi-layer proximity graph | O(log N · ef · d) | Specific queries (high recall for relevance) |
| (optional) | `FaissRetriever` | FAISS library wrapper | varies | When faiss-cpu is installed |

Backend strings for `build_retriever()`: `"numpy"`, `"numpy_ivf"`, `"numpy_hnsw"`, `"faiss"`

### Adaptive retrieval parameters

Each retriever backend has adaptive parameter functions that map `(intent_score, temporal_score)` to search parameters using a shared 3-segment piecewise mapping:
- `ivf.py`: `compute_adaptive_nprobe()` → [8, 48], `compute_adaptive_topN()` → [100, 400]
- `hnsw.py`: `compute_adaptive_ef()` → [32, 200]

### Reranking modules

- `mmr.py` — MMR variants:
  - `mmr_rerank()` — full similarity matrix
  - `mmr_rerank_incremental()` — O(k·N·d), supports `min_score_ratio` filtering and `cluster_ids` + `cluster_penalty` for inter-cluster diversity
  - `mmr_rerank_temporal()` — adds β·freshness bonus + cluster-aware penalties. Score: λ·relevance − (1−λ)·redundancy + β·freshness − γ·cluster_redundancy
- `diversify.py` — `threshold_greedy_rerank` (accept if sim ≤ τ), `maxmin_rerank` (farthest-first)

### Signal fusion pipeline

- `adaptive_lambda.py` — Score-based adaptive λ strategies: gap_piecewise, gap_linear, entropy, hybrid, candidate_diversity. Entry point: `adaptive_lambda_from_scores()`. Also has `auto_tune_thresholds()` for data-driven threshold tuning
- `intent_model.py` — ML classifier: `IntentClassifier.load("models/intent_v3")`, predicts intent clarity (0=ambiguous, 1=specific), maps to λ via `predict_lambda()`
- `temporal.py` — `classify_temporal()` (rule-based time-sensitivity 0–1), `classify_factual()` (fact-seeking detection), `query_aware_lambda_and_beta()` fuses ML intent + factual rules + temporal signals → final (λ, β)

### Other core modules

- `search_api.py` — Clean API boundary: `DiversifiedSearchRequest`/`DiversifiedSearchResponse` dataclasses, wraps all methods into single `diversified_search()` function
- `metrics.py` — recall@k, mean relevance, avg/max pairwise cosine. Key derived metric: ILD (intra-list diversity) = 1 − avg pairwise cosine
- `cluster.py` — K-means pseudo-topic clustering for coverage metrics
- `experiment.py` — `run_sweep()` orchestrates parameter sweeps across methods, produces DataFrame with metrics + timing
- `datasets.py` — HDF5 dataset loading (ANN-Benchmarks format), auto-download from ann-benchmarks.com

### Experiment scripts (`scripts/`)

- `experiment_utils.py` — Shared utilities for experiment scripts: data loading, retriever construction, metric computation. Used by `experiment_quantitative.py` and `experiment_case_study.py`
- `experiment_quantitative.py` — Full ablation: 36 baseline grid (retriever × topN × reranking) + L1–L5 layered ablation with Wilcoxon signed-rank significance tests. Outputs to `outputs/quant_v6/`
- `experiment_case_study.py` — Qualitative case studies: baseline vs fixed MMR vs full pipeline, side-by-side passage display
- `experiment_score_comparison.py` — Ambiguous vs specific query group comparison with bar charts
- `experiment_ablation_index.py` — IVF vs HNSW vs dynamic switching comparison
- `experiment_ivf_vs_hnsw.py` — Recall vs latency sweep benchmarks
- `experiment_chapter7.py` — Unified script for all Chapter 7 paper tables and case studies (overall results, query-type split, layered ablation, significance tests, ANN sweep, intent model eval, qualitative cases). 中文 docstring
- `experiment_adaptive.py` — Adaptive pipeline showcase: adaptation dashboard, robustness boxplots, oracle regret CDF, speed-quality Pareto, parameter adaptation curves. 中文 docstring

### Module dependency graph

```
run_experiments.py → datasets.py, index.py, adaptive_lambda.py, experiment.py, cluster.py
experiment.py     → metrics.py, mmr.py, diversify.py, index.py, adaptive_lambda.py
search_api.py     → adaptive_lambda.py, index.py, mmr.py, diversify.py
temporal.py       → intent_model.py (via caller passing classifier)
demo/search_api.py → index.py, hnsw.py, mmr.py, metrics.py, intent_model.py, temporal.py
cli_text.py, cli_ann.py, demo/run_demo.py → search_api.py
experiment_quantitative.py, experiment_case_study.py → experiment_utils.py → index.py, mmr.py, metrics.py, intent_model.py, temporal.py
```

### Demo backends (`demo/`)

- `run_demo.py` — FastAPI web demo (word embedding search)
- `search_api.py` — MS MARCO passage search: builds 3 retrievers (brute_force, IVF, HNSW), dynamically selects index per query based on intent_score (< 0.4 → IVF, ≥ 0.4 → HNSW)
- `index.html` — Web frontend UI (served by both demo backends)

### Key parameters

**λ (lambda):** λ → 1 = maximize relevance; λ → 0 = maximize diversity. Default strategy: `AdaptiveLambdaConfig` → `entropy`, CLI `run_experiments` → `hybrid`

**β (beta):** Freshness boost weight in temporal reranking. Configured via `TemporalConfig` (beta_min=0.0, beta_max=0.15, per-source freshness scores)

**Demo search pipeline (`demo/search_api.py`) tuned parameters:**
- `lambda_min=0.30, lambda_max=0.90` — λ range for intent-adaptive mapping
- `min_score_ratio = 0.60 + 0.25 * intent_score` — adaptive candidate filter
- Quality boost: when top candidate score < 0.5, λ increases toward 0.9 to preserve relevance
- Three-layer signal fusion: ML intent classifier → factual rules → temporal detection → final (λ, β)
- IVF defaults: nlist=128, nprobe=16; HNSW defaults: M=16, ef_construction=200, ef_search=64

**Reranking methods:** `baseline` (no rerank), `mmr` (fixed λ), `mmr_adaptive` (per-query λ), `threshold`, `maxmin`

## Key Assumptions

- Vectors are L2-normalized (cosine similarity = dot product)
- Datasets use ANN-Benchmarks HDF5 format with `train`, `test`, `neighbors` keys
- Keep HDF5 datasets in `data/`, outputs in `outputs/`, models in `models/`
- Intent model training data in `data/chatgpt_labels.txt`, `data/chatgpt_sentence_labels.txt`, `data/chatgpt_diverse_labels.txt`
- Trained intent model in `models/intent_v3/` (current production); legacy models in `models/intent_chatgpt/` and `models/intent_v2/`
- Improved corpus in `data/improved/` (~45k passages); metadata in `data/improved/metadata.json`
- Some scripts have Chinese (中文) docstrings and comments — this is intentional, maintain the same language when editing those files

## Dependencies

**Required** (in `requirements.txt`): numpy, h5py, tqdm, matplotlib, pandas

**Optional:** faiss-cpu (ANN backend), scikit-learn (clustering/coverage, intent model training), sentence-transformers (text demos, intent model), fastapi + uvicorn (web demo), openai (API intent), requests (Ollama intent), scipy (statistical tests in evaluation), datasets (MS MARCO/Wikipedia/Wikidata download), nltk (WordNet definitions for corpus building)

## Testing

No automated tests. Smoke-test changes by running experiments and verifying `outputs/results.csv` and `outputs/tradeoff.png`.

## Code Style

Python 3, 4-space indentation, snake_case functions/variables, PascalCase classes, UPPER_CASE constants. No formatter configured; match existing style. Uses `from __future__ import annotations` and full type hints throughout.

## Git Conventions

Short imperative commit messages (e.g., "add adaptive lambda sweep"). Do not commit HDF5 datasets or large output files.
