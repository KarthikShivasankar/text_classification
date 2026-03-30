# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TDSuite** is a text classification framework specialized for Technical Debt (TD) detection using transformer models. It supports binary classification, cross-validation, ensemble inference, and carbon emissions tracking. Models are shared on Hugging Face under the `karths/` namespace.

## Commands

### Installation (UV — preferred)
```bash
uv venv && uv sync           # installs torch with CUDA 12.4 via [tool.uv.sources]
uv pip install -e ".[dev]"   # adds black, isort, flake8
uv pip install -r test-requirements.txt  # adds pytest, pytest-cov
```

### Installation (pip fallback)
```bash
# Install PyTorch with CUDA first (required — plain PyPI torch is CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev,test]"
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

# Training (requires CUDA GPU)
tdsuite-train --data_file <path_or_hf_dataset> --model_name <hf_model> --output_dir <dir> [--positive_category <label>] [--numeric_labels] [--is_huggingface_dataset] [--cross_validation --n_splits 5]

# Inference — single model
tdsuite-inference --model_path <dir> --input_file <csv> [--batch_size 32]
tdsuite-inference --model_name <hf_model> --text "some text"
# Inference — ensemble
tdsuite-inference --model_paths model1 model2 --weights 0.5 0.5 --input_file <csv>
```

### GitHub Issues → Inference Pipeline
```bash
python scripts/fetch_github_issues.py --repo owner/repo --output issues.csv [--token $GITHUB_TOKEN]
python scripts/extract_issue_bodies.py --input issues.csv --output issue_texts.csv --min-length 50 --drop-duplicates
tdsuite-inference --model_name karths/binary_classification_train_TD --input_file issue_texts.csv
```

### Web UI
```bash
python app.py   # Gradio UI at http://localhost:7077 (requires gradio — not in requirements.txt)
```

### Code Quality
```bash
black tdsuite/ scripts/
isort tdsuite/ scripts/
flake8 tdsuite/ scripts/   # config in .flake8
pytest --cov=tdsuite       # config in pyproject.toml [tool.pytest.ini_options]
```

## Architecture

### Package Structure (`tdsuite/`)

**`tdsuite/cli.py`** — Single location for all `argparse.ArgumentParser` definitions. Each parser is a `get_*_parser()` function. All CLI entry-point scripts import from here.

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

**Inference** (`tdsuite/utils/inference.py`):
- `InferenceEngine`: single model, supports local or HF Hub, file-based batch predictions. tqdm is a top-level import.
- `EnsembleInferenceEngine`: weighted averaging across multiple models.

**CLI scripts** (`tdsuite/train.py`, `tdsuite/inference.py`, `tdsuite/split_data.py`): Thin wrappers that call `get_*_parser().parse_args()` from `tdsuite/cli.py`.

**GitHub scripts** (`scripts/`):
- `fetch_github_issues.py` — fetches all issues from a public repo via GitHub REST API into CSV. Uses `tdsuite.cli.get_fetch_issues_parser` if tdsuite is installed, otherwise falls back to inline argparse.
- `extract_issue_bodies.py` — cleans the CSV to a `text` column ready for `tdsuite-inference`.

### Key Design Decisions
- Training **requires a CUDA GPU**; the train script raises an error if none is detected.
- Class imbalance is handled at two levels: `DataSplitter.balance_classes()` for data balancing and `WeightedLossTrainer` for loss weighting.
- Cross-validation saves each fold to a separate subdirectory `output_dir/fold_N/`.
- Inference outputs land in a timestamped subdirectory (`inference_YYYYMMDD_HHMMSS/`) under the model's output directory.
- Carbon emissions are tracked per-run via CodeCarbon and saved as `emissions.csv`/`emissions.json`.
- All `argparse` parsers live in `tdsuite/cli.py` — add new parameters there, not in the script files.
- `pyproject.toml` is the single source of truth for packaging, tool config (black, isort, pytest), and optional dependency groups (`dev`, `test`). `setup.py` is kept for legacy compatibility but `pyproject.toml` takes precedence with UV.
