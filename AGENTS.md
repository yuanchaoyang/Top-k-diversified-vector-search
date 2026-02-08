# Repository Guidelines

This repository hosts a compact Python codebase for top-k diversified vector search with MMR-style reranking and ANN-Benchmarks datasets.

## Project Structure & Module Organization
- `src/diverse_search/` core library: datasets, retrieval backends, rerankers (`mmr.py`, `diversify.py`), metrics, experiment pipeline, and CLIs.
- `scripts/` helper entrypoints that add `src/` to `PYTHONPATH` for standalone runs.
- `data/` downloaded HDF5 datasets.
- `outputs/` experiment artifacts such as `results.csv` and `tradeoff.png`.
- `requirements.txt` runtime dependencies.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` create and activate a local environment.
- `pip install -r requirements.txt` install dependencies.
- `PYTHONPATH=src python -m diverse_search.run_experiments --dataset glove-100-angular --k 10 --topN 200 --out-dir outputs/dayX` run a sweep and emit plots/CSVs.
- `PYTHONPATH=src python -m diverse_search.cli_text` interactive text demo.
- `PYTHONPATH=src python -m diverse_search.cli_ann --dataset data/glove-100-angular.hdf5` interactive ANN dataset demo.
- `python scripts/download_dataset.py --dataset glove-100-angular --out data/glove-100-angular.hdf5` fetch a dataset.
- Optional ANN backend: `pip install faiss-cpu` and pass `--backend faiss`.

## Coding Style & Naming Conventions
- Python 3 with 4-space indentation.
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.
- No formatter or linter is configured; keep edits consistent with the surrounding file and existing type hints.

## Testing Guidelines
- No automated tests are checked in.
- Smoke-test changes by running `run_experiments` and verifying `outputs/results.csv` and `outputs/tradeoff.png`.
- If you add tests, place them under `tests/` and mirror module names.

## Commit & Pull Request Guidelines
- Git history uses short, plain summaries with no strict convention. Use a concise imperative subject (e.g., “add adaptive lambda sweep”).
- PRs should describe the change, note commands run, and include any relevant output files or plots (or a brief metric summary). Link related issues when applicable.

## Data & Output Notes
- HDF5 datasets can be large; keep them in `data/` and avoid committing them unless explicitly required.
- Keep generated artifacts under `outputs/` and clean up or document any large result folders.
