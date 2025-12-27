#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJ_ROOT / "src"))

from diverse_search.datasets import DATASET_URLS, download_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ANN-Benchmarks HDF5 dataset")
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset name (e.g. glove-100-angular) or a direct URL. "
            f"Known names: {', '.join(sorted(DATASET_URLS.keys()))}"
        ),
    )
    parser.add_argument("--out", required=True, help="Destination .hdf5 path")
    args = parser.parse_args()

    out = Path(args.out)
    path = download_dataset(args.dataset, out)
    print(f"Saved to: {path.resolve()}")


if __name__ == "__main__":
    main()
