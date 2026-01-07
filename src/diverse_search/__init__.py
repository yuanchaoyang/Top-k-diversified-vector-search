"""Top-k diversified vector search (cosine / geometric diversity).

This package is designed for a graduation project:
- candidate retrieval (exact or ANN)
- diversified reranking (MMR)
- evaluation (recall + diversity metrics)

See README.md for usage.
"""

from .datasets import load_hdf5, download_dataset
from .index import build_retriever
from .mmr import mmr_rerank
from .experiment import run_sweep
from .search_api import diversified_search, DiversifiedSearchRequest

__all__ = [
    "load_hdf5",
    "download_dataset",
    "build_retriever",
    "mmr_rerank",
    "run_sweep",
    "diversified_search",
    "DiversifiedSearchRequest",
]
