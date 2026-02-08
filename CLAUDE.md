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
python demo/run_demo.py  # http://localhost:8000

# Adaptive lambda visualization
PYTHONPATH=src python scripts/visualize_adaptive.py --dataset glove-100-angular
PYTHONPATH=src python scripts/multi_dataset_experiment.py

# Intent-based adaptive lambda (requires Ollama)
ollama serve  # start Ollama server
ollama pull qwen2.5:0.5b  # download model
PYTHONPATH=src python scripts/test_ollama_intent.py

# Train intent classifier from ChatGPT labels (requires scikit-learn, sentence-transformers)
python scripts/train_from_chatgpt.py  # reads data/chatgpt_labels.txt → models/intent_chatgpt/

# Evaluate intent model vs baselines (generates tables, plots, statistical tests)
PYTHONPATH=src python scripts/evaluate_intent_model.py \
  --words-file data/corpus_words.txt --embeddings-file data/corpus_words_embeddings.npy \
  --out-dir outputs/evaluation

# MS MARCO real-query evaluation (requires datasets, sentence-transformers)
python scripts/prepare_msmarco.py --n-passages 50000 --n-queries 1000
PYTHONPATH=src python scripts/evaluate_msmarco.py --data-dir data/msmarco --k 10 --topN 100
```

## Architecture

```
Query → Candidate Retriever (top-N) → Reranker (MMR/adaptive) → Final k results
```

**Core modules in `src/diverse_search/`:**
- `index.py` — Candidate retrieval: `NumpyBruteForceRetriever` (exact) or `FaissRetriever` (ANN/HNSW). Factory: `build_retriever()`
- `mmr.py` — MMR reranking: score = λ·relevance − (1−λ)·redundancy. Has both full-matrix (`mmr_rerank`, builds N×N sim matrix) and incremental (`mmr_rerank_incremental`, O(k·N·d) time, O(N) memory) variants
- `diversify.py` — Alternative rerankers: `threshold_greedy_rerank` (accept if similarity ≤ τ), `maxmin_rerank` (farthest-first)
- `adaptive_lambda.py` — Score-based adaptive λ strategies: gap_piecewise, gap_linear, entropy, hybrid, candidate_diversity. Entry point: `adaptive_lambda_from_scores()`. Also has `auto_tune_thresholds()` and `create_auto_tuned_config()` for data-driven threshold tuning
- `intent_adaptive.py` — LLM-based intent analysis for λ selection (Ollama/OpenAI/rule backends). Caches results in module-level `INTENT_DICTIONARY`
- `intent_model.py` — Intent classifier: `IntentClassifier.load("models/intent_chatgpt")`, predicts intent clarity score (0=ambiguous, 1=specific) from word embeddings, maps to λ via `predict_lambda(word, lambda_min, lambda_max)`
- `search_api.py` — Clean API boundary: `DiversifiedSearchRequest`/`DiversifiedSearchResponse` dataclasses, wraps all methods into single `diversified_search()` function
- `metrics.py` — Evaluation: recall@k, mean relevance, avg/max pairwise cosine. Key derived metric: ILD (intra-list diversity) = 1 − avg pairwise cosine
- `cluster.py` — K-means pseudo-topic clustering for coverage metrics
- `experiment.py` — Experiment framework: `run_sweep()` orchestrates parameter sweeps across methods, produces DataFrame with metrics + timing
- `datasets.py` — HDF5 dataset loading (ANN-Benchmarks format), auto-download from ann-benchmarks.com

**Module dependency graph:**
```
run_experiments.py → datasets.py, index.py, adaptive_lambda.py, experiment.py
experiment.py     → metrics.py, mmr.py, diversify.py, index.py, cluster.py
search_api.py     → adaptive_lambda.py, index.py, mmr.py, diversify.py
intent_adaptive.py → intent_model.py, experiment.py
cli_text.py, cli_ann.py, demo/ → search_api.py
```

**Evaluation scripts in `scripts/`:**
- `evaluate_intent_model.py` — Full comparison: intent model vs fixed-λ vs adaptive vs oracle, with statistical tests (Wilcoxon), stratified analysis by intent group, and tradeoff plots
- `evaluate_msmarco.py` — Same evaluation on MS MARCO real text queries (requires `prepare_msmarco.py` first)
- `train_from_chatgpt.py` — Train intent classifier from `data/chatgpt_labels.txt` using sentence-transformer embeddings + sklearn (MLP/RF/GBR), saves to `models/intent_chatgpt/`
- `experiment_word_search.py` — Word embedding similarity search experiment

**Reranking methods:** `baseline` (no rerank), `mmr` (fixed λ), `mmr_adaptive` (per-query λ), `threshold`, `maxmin`

**Key parameter — λ (lambda):**
- λ → 1: maximize relevance (less diversity)
- λ → 0: maximize diversity (less relevance)
- Adaptive strategies auto-tune λ per query based on score distribution or LLM intent analysis
- Default strategy differs by context: `AdaptiveLambdaConfig` defaults to `entropy`, CLI `run_experiments` defaults to `hybrid`

## Key Assumptions

- Vectors are L2-normalized (cosine similarity = dot product)
- Datasets use ANN-Benchmarks HDF5 format with `train`, `test`, `neighbors` keys
- Keep HDF5 datasets in `data/`, outputs in `outputs/`, models in `models/`
- Intent model training data in `data/chatgpt_labels.txt` (JSON with word→score mappings), trained models in `models/intent_chatgpt/`

## Dependencies

**Required** (in `requirements.txt`): numpy, h5py, tqdm, matplotlib, pandas

**Optional:** faiss-cpu (ANN backend), scikit-learn (clustering/coverage, intent model training), sentence-transformers (text demos, intent model), fastapi + uvicorn (web demo), openai (API intent), requests (Ollama intent), scipy (statistical tests in evaluation), datasets (MS MARCO download)

## Testing

No automated tests. Smoke-test changes by running experiments and verifying `outputs/results.csv` and `outputs/tradeoff.png`.

## Code Style

Python 3, 4-space indentation, snake_case functions/variables, PascalCase classes, UPPER_CASE constants. No formatter configured; match existing style. Uses `from __future__ import annotations` and full type hints throughout.

## Git Conventions

Short imperative commit messages (e.g., "add adaptive lambda sweep"). Do not commit HDF5 datasets or large output files.
