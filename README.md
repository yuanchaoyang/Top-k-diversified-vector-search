# Top-k Diversified Vector Search

A query-adaptive two-stage pipeline for **top-k diversified vector retrieval**: return *k* results that are both **relevant** to the query (high cosine similarity) and **diverse** among themselves (low inter-item similarity).

> 4th Year Project, Artificial Intelligence, School of Informatics, University of Edinburgh

## Key Idea

Standard dense retrieval maximises query similarity alone, producing highly redundant top-k lists. This is especially problematic for ambiguous or exploratory queries (*spring*, *mercury*), where users benefit from coverage across multiple semantic facets rather than repeated variants of one dominant sense.

This project makes both retrieval and reranking **per-query adaptive**. Before retrieval begins, the system estimates:

- **Intent score** *s_I* &in; [0, 1] &mdash; ambiguous (0) vs. specific (1), from a learned regressor + factual fallback rules
- **Temporal score** *s_T* &in; [0, 1] &mdash; recency sensitivity, from lexical/pattern cues

These signals control:

1. **Backend selection** &mdash; IVF (cluster-based, favours diversity) vs. HNSW (graph-based, favours recall)
2. **ANN search parameters** &mdash; adaptive `nprobe` [8, 48], `ef_search` [32, 200], candidate pool *N_c* [100, 400]
3. **Reranking parameters** &mdash; MMR trade-off &lambda; [0.30, 0.90] and freshness bonus &beta; [0, 0.15]

## Architecture

```
Query text
  -> Sentence embedding (all-MiniLM-L6-v2)
  -> Three-signal query analysis
  |    ML intent classifier -> intent_score
  |    Factual fallback rules -> override if factoid
  |    Temporal detector -> temporal_score
  |    Fuse -> (lambda, beta, backend, nprobe/ef, topN)
  |
  -> Adaptive retriever selection
  |    intent_score < 0.4 -> IVF  (cluster structure promotes diversity)
  |    intent_score >= 0.4 -> HNSW (high recall for relevance)
  |
  -> Candidate retrieval (top-N_c)
  |    IVF: optional cluster-balanced sampling
  |    HNSW: beam search with adaptive ef
  |
  -> Incremental MMR reranking (O(N_c) memory)
  |    + relevance floor (min_score_ratio)
  |    + cluster-aware penalty (gamma=0.15 for IVF)
  |    + temporal freshness bonus (beta)
  |    + quality-aware graceful degradation
  |
  -> Final top-k results
```

## Project Layout

```
src/diverse_search/          Core library
  index.py                   Retriever factory + BaseRetriever + build_retriever()
  brute_force.py             Exact dot-product scan, O(Nd)
  ivf.py                     NumPy IVF index (k-means + inverted lists)
  hnsw.py                    NumPy HNSW index (multi-layer proximity graph)
  mmr.py                     MMR variants: standard, incremental, temporal
  diversify.py               Threshold-greedy and maxmin rerankers
  adaptive_lambda.py         Score-based adaptive-lambda strategies
  intent_model.py            Learned intent classifier (GradientBoosting regressor)
  temporal.py                Temporal/factual detection + signal fusion
  search_api.py              Programmatic API (DiversifiedSearchRequest/Response)
  metrics.py                 Recall@k, MeanRel, ILD, coverage, F1_RD
  cluster.py                 K-means pseudo-topic clustering
  datasets.py                ANN-Benchmarks HDF5 dataset loading
  experiment.py              Parameter sweep orchestration
  run_experiments.py         CLI entry point for experiments
  cli_text.py                Interactive text search demo
scripts/                     Training, data preparation, experiments
  train_diverse_intent.py    Train intent classifier from labelled data
  build_improved_corpus.py   Build multi-source passage corpus (~45k)
  prepare_msmarco.py         Download & encode MS MARCO passages
  experiment_chapter7.py     Unified script for all paper experiments
  experiment_quantitative.py Full ablation: 36 baselines + L1-L5 layered ablation
  experiment_case_study.py   Qualitative side-by-side passage comparison
  experiment_ablation_index.py  IVF vs HNSW vs dynamic switching
  experiment_ivf_vs_hnsw.py  Recall vs latency sweep benchmarks
  experiment_adaptive.py     Adaptive pipeline showcase
  experiment_score_comparison.py  Ambiguous vs specific bar charts
  sweep_backend_switch_threshold.py  Backend switch threshold sensitivity
  evaluate_independent_testset.py   Independent intent model evaluation
  experiment_utils.py        Shared data loading & metric utilities

demo/                        Web demo (FastAPI)
  search_api.py              MS MARCO / improved corpus passage search backend
  run_demo.py                Word embedding search backend
  index.html                 Web frontend UI

models/intent_v3/            Deployed intent model checkpoint
  intent_model.pkl           GradientBoosting regressor + fitted StandardScaler
  intent_labels.json         Exact-match label cache

data/                        Datasets (not committed, see setup below)
outputs/                     Experiment results (CSV, PNG, LaTeX)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional backends
pip install faiss-cpu          # FAISS ANN backend
pip install scikit-learn       # intent model training, clustering
pip install sentence-transformers  # text encoding
pip install fastapi uvicorn    # web demo
pip install scipy              # statistical significance tests
pip install datasets           # HuggingFace dataset download
pip install nltk               # WordNet definitions
```

## Data Preparation

### Option A: Improved Corpus (recommended, ~45k passages)

Five sources: Wikipedia (FineWiki), Wikidata descriptions, WordNet definitions, Wikipedia disambiguation pages, LoTTE Q&A passages.

```bash
python scripts/build_improved_corpus.py --output data/improved
```

### Option B: MS MARCO Passage Corpus

```bash
python scripts/prepare_msmarco.py --n-passages 50000 --n-queries 1000
```

## Running

### Web Demo

```bash
# Passage search (requires data/improved/ or data/msmarco/)
python demo/search_api.py          # http://localhost:8000

# Word embedding search
python demo/run_demo.py            # http://localhost:8000
```

### Experiments

All experiment commands require `PYTHONPATH=src` or `source .venv/bin/activate`:

```bash
# Chapter 7 unified experiment (paper tables + qualitative cases)
PYTHONPATH=src python scripts/experiment_chapter7.py \
  --improved-dir data/improved --k 10 --out-dir outputs/chapter7

# Full quantitative ablation (36 baselines + L1-L5 layered ablation)
PYTHONPATH=src python scripts/experiment_quantitative.py \
  --msmarco-dir data/msmarco --improved-dir data/improved --k 10 --out-dir outputs/quant_v6

# IVF vs HNSW recall-latency sweep
PYTHONPATH=src python scripts/experiment_ivf_vs_hnsw.py --out-dir outputs/ivf_vs_hnsw

# Backend switch threshold sensitivity
PYTHONPATH=src python scripts/sweep_backend_switch_threshold.py \
  --improved-dir data/improved --k 10 --out-dir outputs/backend_switch_threshold

```

### Interactive CLI

```bash
PYTHONPATH=src python -m diverse_search.cli_text
```

### Intent Model Training

```bash
# Train from merged label files -> models/intent_v3/
python scripts/train_diverse_intent.py

# Independent test-set evaluation
PYTHONPATH=src python scripts/evaluate_independent_testset.py \
  --test-set data/intent_test_set.json --model-dir models/intent_v3
```

## Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **MeanRel** | Average cosine similarity of the final top-k to the query |
| **ILD** | Intra-list diversity: 1 - average pairwise cosine among top-k |
| **Avg. F1** | Per-query harmonic mean of MeanRel and ILD, then averaged |
| **Recall@k** | Fraction of exact nearest neighbours recovered by ANN |
| **Coverage@k** | Fraction of distinct semantic clusters represented in top-k |

## Key Results (from Chapter 7)

**End-to-end results** on ~45k passage corpus, 160 queries, k=10:

| Method | MeanRel | ILD | Avg. F1 |
|--------|---------|-----|---------|
| Relevance-only baseline | 0.5495 | 0.5350 | 0.5279 |
| Fixed-MMR (lambda=0.7) | 0.5245 | 0.6363 | 0.5607 |
| L5: full adaptive pipeline | 0.5079 | 0.6395 | 0.5504 |

The adaptive pipeline broadens coverage for ambiguous queries (ILD 0.6775 vs baseline 0.6204) while remaining conservative on specific queries (MeanRel 0.5272 vs baseline 0.5205). Fixed-MMR achieves the best overall F1, but the adaptive system provides query-sensitive behaviour within a practical ANN serving architecture.

**Ablation layers** (L2-L5 progressively extend the learned-intent configuration):
- L1: score-based adaptive-lambda (heuristic baseline)
- L2: + learned intent controller
- L3: + dynamic IVF/HNSW backend selection
- L4: + cluster-aware reranking penalty
- L5: + temporal signals and freshness-aware reranking

## Datasets & External Resources

| Resource | Source | Usage |
|----------|--------|-------|
| Wikipedia passages | [HuggingFaceFW/finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) | Improved corpus (Bucket A) |
| Wikidata descriptions | [masaki-sakata/wikidata_descriptions](https://huggingface.co/datasets/masaki-sakata/wikidata_descriptions) | Improved corpus (Bucket B) |
| WordNet | NLTK `wordnet` + `omw-1.4` | Improved corpus (Bucket C) |
| Wikipedia disambiguation | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (20231101.en) | Improved corpus (Bucket D) |
| LoTTE passages | [colbertv2/lotte_passages](https://huggingface.co/datasets/colbertv2/lotte_passages) | Improved corpus (Bucket E) |
| MS MARCO | [microsoft/ms_marco](https://huggingface.co/datasets/microsoft/ms_marco) (v2.1) | Alternative passage corpus |
| Sentence encoder | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | All text encoding |

## Dependencies

**Required**: numpy, h5py, tqdm, matplotlib, pandas

**Optional**: faiss-cpu, scikit-learn, sentence-transformers, fastapi, uvicorn, scipy, datasets, nltk, openai, requests

## Code Style

Python 3, 4-space indentation, snake_case functions/variables, PascalCase classes. Uses `from __future__ import annotations` and full type hints throughout.

## License

University of Edinburgh, School of Informatics, 4th Year Project.
