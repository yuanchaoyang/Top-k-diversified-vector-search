from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm


DATASET_URLS = {
    "glove-25-angular": "http://ann-benchmarks.com/glove-25-angular.hdf5",
    "glove-50-angular": "http://ann-benchmarks.com/glove-50-angular.hdf5",
    "glove-100-angular": "http://ann-benchmarks.com/glove-100-angular.hdf5",
    "nytimes-256-angular": "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
    "lastfm-64-dot": "http://ann-benchmarks.com/lastfm-64-dot.hdf5",

    "sift-128-euclidean": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
}


@dataclass(frozen=True)
class H5Dataset:
    """In-memory representation of an ANN-Benchmarks HDF5 dataset."""

    train: np.ndarray  # (n, d)
    test: np.ndarray  # (nq, d)
    neighbors: Optional[np.ndarray] = None  # (nq, 100) ground truth
    distances: Optional[np.ndarray] = None  # (nq, 100)

    @property
    def dim(self) -> int:
        return int(self.train.shape[1])


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


class _TqdmUpTo(tqdm):
    """Progress bar for urlretrieve."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_dataset(dataset: str, out_path: str | os.PathLike) -> Path:
    """Download a dataset by name into `out_path`.

    Args:
        dataset: Name key in DATASET_URLS, or a full URL.
        out_path: Destination file path.

    Returns:
        Path to downloaded file.
    """
    url = DATASET_URLS.get(dataset, dataset)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    with _TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=f"Downloading") as t:
        urllib.request.urlretrieve(url, filename=str(out_path), reporthook=t.update_to)

    return out_path


def load_hdf5(
    path: str | os.PathLike,
    *,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
    normalize: bool = True,
) -> H5Dataset:
    """Load an ANN-Benchmarks HDF5 dataset.

    This expects the common keys:
      - 'train': database vectors
      - 'test': query vectors
      - 'neighbors': ground-truth neighbor ids (optional)
      - 'distances': ground-truth distances (optional)

    Args:
        path: HDF5 file.
        max_train: Optional cap on number of train vectors (for quick experiments).
        max_test: Optional cap on number of test queries.
        normalize: If True, L2-normalize train & test for cosine retrieval.

    Returns:
        H5Dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with h5py.File(path, "r") as f:
        train = f["train"]
        test = f["test"]

        n_train = int(train.shape[0])
        n_test = int(test.shape[0])

        if max_train is None or max_train <= 0 or max_train > n_train:
            max_train = n_train
        if max_test is None or max_test <= 0 or max_test > n_test:
            max_test = n_test

        X_train = np.array(train[:max_train], dtype=np.float32)
        X_test = np.array(test[:max_test], dtype=np.float32)

        neighbors = None
        distances = None
        if "neighbors" in f:
            neighbors = np.array(f["neighbors"][:max_test], dtype=np.int32)
            # If we sub-sample train, ground truth indices might exceed max_train.
            # We'll keep them, and downstream code can optionally mask/filter.
        if "distances" in f:
            distances = np.array(f["distances"][:max_test], dtype=np.float32)

    if normalize:
        X_train = l2_normalize(X_train)
        X_test = l2_normalize(X_test)

    return H5Dataset(train=X_train, test=X_test, neighbors=neighbors, distances=distances)
