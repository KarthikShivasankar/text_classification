# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TDSuite** is a text classification framework specialized for Technical Debt (TD) detection using transformer models. It supports binary classification, cross-validation, ensemble inference, and carbon emissions tracking. Models are shared on Hugging Face under the `karths/` namespace.

**Default inference backend is ONNX (CPU) — PyTorch is optional** (needed only for training or explicit GPU inference). Requires **Python ≥ 3.9**.

## Commands

### Installation (from PyPI)
```bash
pip install tdsuite            # CPU inference (ONNX) — no torch required
pip install "tdsuite[gpu]"     # + GPU inference (onnxruntime-gpu + torch CUDA 12.4)
pip install "tdsuite[train]"   # + full training stack
pip install "tdsuite[onnx]"    # + onnx/onnxscript for exporting custom models
```

### Installation (UV — preferred for development)
```bash
uv venv && uv sync           # CPU inference (ONNX) — no torch required
uv pip install -e ".[gpu]"   # + GPU inference (onnxruntime-gpu + torch CUDA 12.4)
uv pip install -e ".[train]" # + full training stack (torch, codecarbon, evaluate…)
uv pip install -e ".[onnx]"    # + onnx/onnxscript for exporting custom models
uv pip install -e ".[dev]"     # + ruff, black, isort, flake8
uv pip install -r test-requirements.txt  # + pytest, pytest-cov
# optimum (NOTE: optimum 2.x requires transformers<5 — conflicts with current env)
# The torch.onnx.export fallback works without optimum; only install if needed.
uv pip install "optimum[onnxruntime]" "transformers<5"
```

### Installation (pip fallback)
```bash
pip install -e .                     # CPU inference — no GPU needed
pip install -e ".[gpu,dev,test]"     # GPU + dev + test extras
# For training with CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[train,dev,test]"
```

### Publishing to PyPI
```bash
uv pip install build twine
python -m build
twine upload dist/*
```

### CLI Entry Points (after install)
```bash
# Data preparation
tdsuite-split-data --data_file <path_or_hf_dataset> --output_dir <dir> [--is_huggingface_dataset] [--is_numeric_labels] [--repo_column <col>]

# Training (requires CUDA GPU + pip install 'tdsuite[train]')
tdsuite-train --data_file <path_or_hf_dataset> --model_name <hf_model> --output_dir <dir> [--positive_category <label>] [--numeric_labels] [--is_huggingface_dataset] [--cross_validation --n_splits 5]

# Inference — ONNX CPU (default, model.onnx auto-downloaded from HF Hub)
tdsuite-inference --model_name <hf_model> --input_file <csv> [--batch_size 32]
tdsuite-inference --model_name <hf_model> --text "some text"

# Inference — ONNX GPU (requires pip install 'tdsuite[gpu]')
tdsuite-inference --model_name <hf_model> --device cuda --input_file <csv>

# Inference — local ONNX file
tdsuite-inference --onnx_path model.onnx --input_file <csv>

# Inference — PyTorch backend (use --use_torch, requires torch)
tdsuite-inference --model_path <dir> --use_torch --input_file <csv>

# Inference — ensemble (uses PyTorch backend automatically)
tdsuite-inference --model_paths model1 model2 --weights 0.5 0.5 --input_file <csv>
```

### GitHub Issues → Inference Pipeline
```bash
python scripts/fetch_github_issues.py --repo owner/repo --output issues.csv [--token $GITHUB_TOKEN]
python scripts/extract_issue_bodies.py --input issues.csv --output issue_texts.csv --min-length 50 --drop-duplicates
tdsuite-inference --model_name karths/binary_classification_train_TD --input_file issue_texts.csv
```

### Export ONNX (for custom/fine-tuned models only)
```bash
# Pre-trained models already have model.onnx on HF Hub — no export needed.
# Only use these for your own fine-tuned models:
python scripts/export_onnx.py --model_path outputs/my_model --output model.onnx
python scripts/export_and_upload_onnx.py  # batch-export all 17 pre-trained models
```

### Web UI
```bash
python app.py   # Gradio UI at http://localhost:7077 (requires gradio — not in requirements.txt)
```

### Code Quality
```bash
ruff check tdsuite/ scripts/   # primary linter + import sort (config in pyproject.toml)
ruff check --fix tdsuite/ scripts/
black tdsuite/ scripts/
isort tdsuite/ scripts/
flake8 tdsuite/ scripts/       # legacy linter, kept green (config in .flake8)
pytest --cov=tdsuite           # config in pyproject.toml [tool.pytest.ini_options]
```

CI (`.github/workflows/ci.yml`) runs ruff/black/isort + pytest on Python 3.9–3.12 and a build+twine-check on every push/PR to `main`. Releases publish to PyPI via Trusted Publishing (`.github/workflows/release.yml`) when a GitHub Release is published.

## Architecture

### Package Structure (`tdsuite/`)

**`tdsuite/cli.py`** — Single location for all `argparse.ArgumentParser` definitions. Each parser is a `get_*_parser()` function. All CLI entry-point scripts import from here. Includes `--use_torch` and `--device` flags for the inference parser.

**Config** (`tdsuite/config/config.py`): Dataclasses `ModelConfig`, `TrainingConfig`, `DataConfig`, `InferenceConfig` combined into a `Config` class with `save()`/`load()` JSON serialization.

**Data pipeline**:
- `tdsuite/data/dataset.py`: `TDProcessor` loads data from files or HF Hub; `BinaryTDProcessor` converts categorical labels to 0/1 and supports repository-based extraction. `TDDataset` is the PyTorch Dataset used by all `create_dataset()` calls.
- `tdsuite/data/data_splitter.py`: `DataSplitter` handles balanced train/test splits and top-repo extraction.
- `tdsuite/utils/data_utils.py`: Unified `load_dataset()` that tries local file (CSV/JSON/JSONL) then HF Hub. Note: the HF `load_dataset` is imported as `hf_load_dataset` to avoid shadowing the local function.

**Models** (`tdsuite/models/`):
- `BaseModel` wraps HF `AutoModelForSequenceClassification` with weighted loss support.
- `TransformerModel` extends `BaseModel` with `predict()`, `load_from_checkpoint()`, `save_pretrained()`.

**Trainers** (`tdsuite/trainers/`):
- `WeightedLossTrainer` extends HF `Trainer` to apply class weights for imbalanced data.
- `BaseTrainer` wraps training with CodeCarbon emissions tracking, metric computation, and visualization.
- `TDTrainer` extends `BaseTrainer` with `train_with_cross_validation()`, `train_with_early_stopping()`, `predict_with_confidence()`, and `predict_with_ensemble()`.

**Inference** (`tdsuite/utils/`):
- `onnx_inference.py`: `OnnxInferenceEngine` — **default inference path**, no torch required. Supports CPU and GPU (CUDAExecutionProvider). `from_pretrained(model_id)` auto-downloads `model.onnx` from HF Hub; if `model.onnx` is absent, falls back to `_export_to_onnx()` which uses `torch.onnx.export` (requires `torch` + `onnx`). tqdm is a top-level import.
- `inference.py`: `InferenceEngine` and `EnsembleInferenceEngine` — PyTorch-based, lazy-imported (only loaded when explicitly requested via `--use_torch` or ensemble mode).

**`tdsuite/inference.py`** (CLI entry point): Defaults to ONNX. Uses PyTorch only when `--use_torch` is passed or when ensemble mode (`--model_paths`/`--model_names`) is used. No top-level `import torch`.

**`tdsuite/utils/__init__.py`**: `OnnxInferenceEngine` imported eagerly; `InferenceEngine`/`EnsembleInferenceEngine` lazy-imported via `__getattr__` to avoid requiring torch on `import tdsuite`.

**CLI scripts** (`tdsuite/train.py`, `tdsuite/inference.py`, `tdsuite/split_data.py`): Thin wrappers that call `get_*_parser().parse_args()` from `tdsuite/cli.py`.

**GitHub scripts** (`scripts/`):
- `fetch_github_issues.py` — fetches all issues from a public repo via GitHub REST API into CSV.
- `extract_issue_bodies.py` — cleans the CSV to a `text` column ready for `tdsuite-inference`.
- `export_onnx.py` — exports a single model to ONNX (needs `tdsuite[onnx]`).
- `export_and_upload_onnx.py` — batch-exports and uploads all 17 pre-trained TDSuite models to their HF Hub repos. Reads `HF_TOKEN` from `.env`.

### Key Design Decisions
- **Inference is ONNX-first**: `pip install -e .` gives full CPU inference; PyTorch is not installed.
- **ONNX export fallback**: `OnnxInferenceEngine.from_pretrained` first tries to download `model.onnx` from HF Hub. If the file is absent, it calls `_export_to_onnx()` which uses `torch.onnx.export` (requires `torch` + `onnx`). The exported file is cached in the HF Hub snapshot directory.
- **optimum compatibility**: `optimum[onnxruntime]` 2.x requires `transformers<5`; the current environment has `transformers>=5`. Do not rely on `optimum.onnxruntime.ORTModelForSequenceClassification` — use the built-in `torch.onnx.export` fallback instead.
- **GPU inference uses ONNX**: `--device cuda` selects `CUDAExecutionProvider` (requires `pip install 'tdsuite[gpu]'`). PyTorch GPU path only available via `--use_torch`.
- Training **requires a CUDA GPU**; the train script raises an error if none is detected.
- Class imbalance is handled at two levels: `DataSplitter.balance_classes()` for data balancing and `WeightedLossTrainer` for loss weighting.
- Cross-validation saves each fold to a separate subdirectory `output_dir/fold_N/`.
- Inference outputs land in a timestamped subdirectory (`inference_YYYYMMDD_HHMMSS/`) under the model's output directory.
- Carbon emissions are tracked per-run via CodeCarbon and saved as `emissions.csv`/`emissions.json`.
- All `argparse` parsers live in `tdsuite/cli.py` — add new parameters there, not in the script files.
- `pyproject.toml` is the single source of truth for packaging, tool config (black, isort, pytest), and optional dependency groups (`gpu`, `train`, `onnx`, `optimum`, `dev`, `test`). There is no `setup.py` — packaging is fully PEP 621.
- **Version is single-sourced**: `pyproject.toml` declares `dynamic = ["version"]` and reads it from `tdsuite.__version__` (`tdsuite/__init__.py`). Bump the version in `__init__.py` only.
- **Core install stays torch/datasets-free**: modules in the default import path (`tdsuite.inference`, `tdsuite.utils.onnx_inference`, `tdsuite.split_data`) must not import `torch`, `datasets`, `codecarbon`, etc. at top level. The `datasets` import in `data_utils.py`/`data_splitter.py` and the PyTorch engines in `utils/inference.py` are lazy-loaded (via in-function imports and `__getattr__` in `tdsuite/utils/__init__.py` and `tdsuite/data/__init__.py`), so `pip install tdsuite` (ONNX-only) imports cleanly.
- **Lint/format**: ruff is the primary linter/import-sorter (`ruff check tdsuite/ scripts/`); `black` and `flake8` are also kept green. Intentional pre-import statements are marked `# noqa: E402`. Config lives in `pyproject.toml` (`[tool.ruff]`, `[tool.black]`, `[tool.isort]`) and `.flake8`.
