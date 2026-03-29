# TD-Classifier Suite

A suite for classifying technical debt in software issues and code comments using transformer models. Supports GPU training, CPU inference via ONNX, and end-to-end pipelines that pull issues directly from any public GitHub repository.

## Features

- **Binary Classification** — classify text as technical debt or not across 18 TD categories
- **Pre-trained Models** — 17 ready-to-use models on Hugging Face Hub, no training required
- **CPU Inference via ONNX** — export any model to ONNX and run inference without a GPU
- **GitHub Issues Pipeline** — fetch issues from any public repo, clean them, and classify in three commands
- **Flexible Training** — cross-validation, early stopping, class weighting for imbalanced data
- **Ensemble Inference** — combine multiple models with optional per-model weights
- **Carbon Tracking** — CodeCarbon emissions tracking during training and inference
- **Multiple Data Sources** — local CSV/JSON/JSONL files or Hugging Face datasets
- **Gradio Web UI** — browser-based fine-tuning and evaluation interface (`app.py`)

---

## Installation

### With UV (recommended)

[UV](https://docs.astral.sh/uv/) is a fast Python package manager.

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
git clone https://github.com/KarthikShivasankar/text_classification
cd text_classification

uv venv
uv pip install -e .                      # runtime
uv pip install -e ".[dev]"               # adds black, isort, flake8
uv pip install -r test-requirements.txt  # adds pytest, pytest-cov
uv pip install -e ".[onnx]"             # adds onnx, onnxruntime for CPU inference
```

After `uv venv`, point your IDE's Python interpreter to `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux/Mac).

### With pip

```bash
git clone https://github.com/KarthikShivasankar/text_classification
cd text_classification
pip install -e .
pip install -e ".[dev,test,onnx]"   # all optional groups
```

### CPU-only (no GPU)

Skip the heavy training stack — install only what's needed for ONNX inference:

```bash
uv venv
uv pip install transformers tokenizers tqdm pandas requests numpy onnxruntime onnx
```

See [CPU Inference with ONNX](#cpu-inference-with-onnx) for the next steps.

### Publishing to PyPI

```bash
uv pip install build twine
python -m build
twine upload dist/*
```

---

## Quick Start

### Analyze a GitHub repository for technical debt

```bash
# 1. Fetch the latest 100 issues
python scripts/fetch_github_issues.py --repo owner/repo --output issues.csv

# 2. Extract issue body text
python scripts/extract_issue_bodies.py --input issues.csv --output issue_texts.csv

# 3a. Classify (GPU)
tdsuite-inference --model_name karths/binary_classification_train_TD --input_file issue_texts.csv

# 3b. Classify (CPU, no GPU needed)
python scripts/export_onnx.py --model_name karths/binary_classification_train_TD --output model.onnx
tdsuite-inference --onnx_path model.onnx --input_file issue_texts.csv
```

---

## Hugging Face Integration

### Available Datasets

| Category | Dataset |
|----------|---------|
| General TD | [karths/binary-10IQR-TD](https://huggingface.co/datasets/karths/binary-10IQR-TD) |
| Architecture | [karths/binary-10IQR-architecture](https://huggingface.co/datasets/karths/binary-10IQR-architecture) |
| Code Quality | [karths/binary-10IQR-code](https://huggingface.co/datasets/karths/binary-10IQR-code) |
| Defects | [karths/binary-10IQR-defect](https://huggingface.co/datasets/karths/binary-10IQR-defect) |
| Infrastructure | [karths/binary-10IQR-infrastructure](https://huggingface.co/datasets/karths/binary-10IQR-infrastructure) |
| Performance | [karths/binary-10IQR-perf](https://huggingface.co/datasets/karths/binary-10IQR-perf) |
| Requirements | [karths/binary-10IQR-requirement](https://huggingface.co/datasets/karths/binary-10IQR-requirement) |
| Design | [karths/binary-10IQR-design](https://huggingface.co/datasets/karths/binary-10IQR-design) |
| Security | [karths/binary-10IQR-secu](https://huggingface.co/datasets/karths/binary-10IQR-secu) |
| Usability | [karths/binary-10IQR-usab](https://huggingface.co/datasets/karths/binary-10IQR-usab) |
| Compatibility | [karths/binary-10IQR-comp](https://huggingface.co/datasets/karths/binary-10IQR-comp) |
| Reliability | [karths/binary-10IQR-reli](https://huggingface.co/datasets/karths/binary-10IQR-reli) |
| Process | [karths/binary-10IQR-process](https://huggingface.co/datasets/karths/binary-10IQR-process) |
| Build | [karths/binary-10IQR-build](https://huggingface.co/datasets/karths/binary-10IQR-build) |
| Maintenance | [karths/binary-10IQR-main](https://huggingface.co/datasets/karths/binary-10IQR-main) |
| Automation | [karths/binary-10IQR-automation](https://huggingface.co/datasets/karths/binary-10IQR-automation) |
| People | [karths/binary-10IQR-people](https://huggingface.co/datasets/karths/binary-10IQR-people) |
| Portability | [karths/binary-10IQR-port](https://huggingface.co/datasets/karths/binary-10IQR-port) |

### Pre-trained Models

| Category | Model |
|----------|-------|
| General TD | [karths/binary_classification_train_TD](https://huggingface.co/karths/binary_classification_train_TD) |
| Architecture | [karths/binary_classification_train_architecture](https://huggingface.co/karths/binary_classification_train_architecture) |
| Code Quality | [karths/binary_classification_train_code](https://huggingface.co/karths/binary_classification_train_code) |
| Defects | [karths/binary_classification_train_defect](https://huggingface.co/karths/binary_classification_train_defect) |
| Infrastructure | [karths/binary_classification_train_infrastructure](https://huggingface.co/karths/binary_classification_train_infrastructure) |
| Performance | [karths/binary_classification_train_perf](https://huggingface.co/karths/binary_classification_train_perf) |
| Requirements | [karths/binary_classification_train_requirement](https://huggingface.co/karths/binary_classification_train_requirement) |
| Design | [karths/binary_classification_train_design](https://huggingface.co/karths/binary_classification_train_design) |
| Security | [karths/binary_classification_train_secu](https://huggingface.co/karths/binary_classification_train_secu) |
| Usability | [karths/binary_classification_train_usab](https://huggingface.co/karths/binary_classification_train_usab) |
| Reliability | [karths/binary_classification_train_reli](https://huggingface.co/karths/binary_classification_train_reli) |
| Process | [karths/binary_classification_train_process](https://huggingface.co/karths/binary_classification_train_process) |
| Build | [karths/binary_classification_train_build](https://huggingface.co/karths/binary_classification_train_build) |
| Maintenance | [karths/binary_classification_train_main](https://huggingface.co/karths/binary_classification_train_main) |
| Automation | [karths/binary_classification_train_automation](https://huggingface.co/karths/binary_classification_train_automation) |
| People | [karths/binary_classification_train_people](https://huggingface.co/karths/binary_classification_train_people) |
| Portability | [karths/binary_classification_train_port](https://huggingface.co/karths/binary_classification_train_port) |

---

## Usage

### 1. Data Preparation

Input data must be CSV, JSON, or JSONL with at least a `text` column and a `label` column (0/1):

```csv
text,label
"This code has complex logic that should be simplified",1
"The documentation is outdated and needs to be updated",0
"The architecture violates the single responsibility principle",1
```

Split a dataset into balanced train/test sets and extract top-contributing repositories:

```bash
tdsuite-split-data \
    --data_file "karths/binary-10IQR-TD" \
    --output_dir "data/split" \
    --is_numeric_labels \
    --repo_column "repo" \
    --is_huggingface_dataset
```

Outputs: `data/split/train.csv`, `data/split/test.csv`, `data/split/top_repos.csv`.

### 2. Training

Training requires a CUDA GPU. On CPU the script will warn and proceed, but runtime will be very long — use [ONNX inference](#cpu-inference-with-onnx) with a pre-trained model instead.

```bash
# Train on a Hugging Face dataset
tdsuite-train \
    --data_file "karths/binary-10IQR-TD" \
    --model_name "distilbert-base-uncased" \
    --numeric_labels \
    --output_dir "outputs/binary" \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --warmup_steps 1000

# Train on a local file with categorical labels
tdsuite-train \
    --data_file "data/split/train.csv" \
    --model_name "distilbert-base-uncased" \
    --positive_category "TD" \
    --output_dir "outputs/binary"

# 5-fold cross-validation
tdsuite-train \
    --data_file "karths/binary-10IQR-TD" \
    --model_name "distilbert-base-uncased" \
    --numeric_labels \
    --output_dir "outputs/cv" \
    --cross_validation \
    --n_splits 5
```

**Training arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_file` | *(required)* | Local file path or HF dataset name |
| `--model_name` | *(required)* | HF model name or local path |
| `--output_dir` | *(required)* | Directory to save model and outputs |
| `--text_column` | `text` | Text column name |
| `--label_column` | `label` | Label column name |
| `--positive_category` | — | Label value for the positive class (categorical labels) |
| `--numeric_labels` | `false` | Labels are already 0/1 integers |
| `--is_huggingface_dataset` | `false` | Load from Hugging Face Hub |
| `--num_epochs` | `3` | Training epochs |
| `--batch_size` | `16` | Per-device batch size |
| `--learning_rate` | `2e-5` | Learning rate |
| `--weight_decay` | `0.01` | Weight decay |
| `--warmup_steps` | `500` | LR warmup steps |
| `--gradient_accumulation_steps` | `1` | Gradient accumulation |
| `--cross_validation` | `false` | Enable k-fold cross-validation |
| `--n_splits` | `5` | Number of CV folds |
| `--max_length` | `512` | Max token sequence length |
| `--seed` | `42` | Random seed |

Cross-validation output structure:
```
outputs/cv/
├── fold_0/ … fold_N/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── cross_validation_results.json
└── cross_validation_visualization.png
```

### 3. Inference

#### Single model (GPU)

```bash
# From a local checkpoint
tdsuite-inference --model_path "outputs/binary" --input_file "data/split/test.csv"

# From Hugging Face
tdsuite-inference --model_name "karths/binary_classification_train_TD" --input_file "data/split/test.csv"

# Single text
tdsuite-inference --model_name "karths/binary_classification_train_TD" --text "This service has no error handling"
```

#### Ensemble (GPU)

```bash
tdsuite-inference \
    --model_names "karths/binary_classification_train_TD" \
                  "karths/binary_classification_train_code" \
    --input_file "data/split/test.csv" \
    --weights 0.6 0.4
```

#### CPU inference via ONNX (no GPU)

```bash
tdsuite-inference --onnx_path model.onnx --input_file "data/split/test.csv"
```

**Inference arguments:**

| Argument | Description |
|----------|-------------|
| `--model_path` | Local model directory |
| `--model_name` | HF model name |
| `--model_paths` | Multiple local models (ensemble) |
| `--model_names` | Multiple HF models (ensemble) |
| `--onnx_path` | ONNX model file for CPU inference |
| `--text` | Single text string to classify |
| `--input_file` | CSV/JSON file to classify |
| `--text_column` | Text column name (default: `text`) |
| `--output_file` | Path to save predictions |
| `--results_dir` | Custom results directory |
| `--batch_size` | Batch size (default: `32`) |
| `--max_length` | Max token length (default: `512`) |
| `--device` | `cuda` or `cpu` (default: `cuda`) |
| `--weights` | Per-model weights for ensemble |
| `--disable_progress_bar` | Hide tqdm progress bars |
| `--track_emissions` | Track carbon emissions (default: `true`) |

Each inference run creates a timestamped output directory:
```
outputs/binary/inference_YYYYMMDD_HHMMSS/
├── predictions_test.csv
└── metrics/
    ├── metrics.json
    ├── confusion_matrix.png
    └── roc_curve.png
```

---

## CPU Inference with ONNX

Export any trained or pre-trained model to ONNX and run inference on CPU without any GPU. ONNX inference is typically 2–4× faster than PyTorch on CPU.

### Step 1 — Install ONNX dependencies

```bash
uv pip install -e ".[onnx]"
# or: pip install onnx onnxruntime
```

### Step 2 — Export to ONNX

```bash
# From a local checkpoint
python scripts/export_onnx.py --model_path outputs/binary --output model.onnx

# Directly from Hugging Face (no training required)
python scripts/export_onnx.py \
    --model_name karths/binary_classification_train_TD \
    --output model.onnx
```

The tokenizer is saved alongside `model.onnx` automatically.

### Step 3 — Run CPU inference

```bash
tdsuite-inference --onnx_path model.onnx --input_file issue_texts.csv
```

`--onnx_path` bypasses PyTorch and GPU entirely. Do not combine with `--model_path` / `--model_name`.

**Export arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | — | Local model directory *(mutually exclusive with `--model_name`)* |
| `--model_name` | — | HF model name *(mutually exclusive with `--model_path`)* |
| `--output` | *(required)* | Destination `.onnx` file |
| `--max_length` | `512` | Sequence length for the export dummy input |
| `--opset` | `14` | ONNX opset version |

---

## GitHub Issues Analysis

Analyze technical debt in any public GitHub repository in three commands.

### Step 1 — Fetch issues

By default fetches the **latest 100 issues**, sorted newest-first.

```bash
# Latest 100 (default)
python scripts/fetch_github_issues.py --repo owner/repo --output issues.csv

# Latest N issues
python scripts/fetch_github_issues.py --repo owner/repo --limit 500 --output issues.csv

# Every issue across all pages
python scripts/fetch_github_issues.py --repo owner/repo --all --output issues.csv
```

**Rate limit exceeded?** Unauthenticated API calls are capped at 60/hour. Supply a token to raise this to 5,000/hour:

```bash
python scripts/fetch_github_issues.py \
    --repo owner/repo \
    --token "$GITHUB_TOKEN" \
    --output issues.csv
```

Create a token at <https://github.com/settings/tokens> — no scopes are needed for public repos.

When the limit is hit the script prints the exact wait time and the token creation link automatically.

Output columns: `id`, `number`, `title`, `body`, `state`, `created_at`, `updated_at`, `closed_at`, `user_login`, `labels`, `comments`, `url`.

### Step 2 — Clean issue bodies

```bash
python scripts/extract_issue_bodies.py \
    --input issues.csv \
    --output issue_texts.csv \
    --min-length 50 \
    --drop-duplicates \
    --keep-metadata
```

Produces a CSV with a `text` column ready for inference. `--keep-metadata` also keeps `number` and `title` for traceability back to the original issue.

### Step 3 — Run inference

```bash
# GPU
tdsuite-inference \
    --model_name "karths/binary_classification_train_TD" \
    --input_file issue_texts.csv

# CPU (ONNX)
tdsuite-inference \
    --onnx_path model.onnx \
    --input_file issue_texts.csv

# Ensemble — detect both general TD and code quality issues
tdsuite-inference \
    --model_names "karths/binary_classification_train_TD" \
                  "karths/binary_classification_train_code" \
    --input_file issue_texts.csv \
    --weights 0.6 0.4
```

---

## Project Structure

```
text_classification/
├── scripts/
│   ├── fetch_github_issues.py   # Fetch issues from any public GitHub repo → CSV
│   ├── extract_issue_bodies.py  # Clean issue CSV → text column ready for inference
│   └── export_onnx.py           # Export a transformer model to ONNX for CPU inference
├── tdsuite/
│   ├── cli.py                   # All argparse parsers (single source of truth)
│   ├── train.py                 # tdsuite-train entry point
│   ├── inference.py             # tdsuite-inference entry point
│   ├── split_data.py            # tdsuite-split-data entry point
│   ├── config/
│   │   └── config.py            # ModelConfig, TrainingConfig, DataConfig, InferenceConfig
│   ├── data/
│   │   ├── dataset.py           # TDDataset, TDProcessor, BinaryTDProcessor
│   │   └── data_splitter.py     # DataSplitter — balanced splits, top-repo extraction
│   ├── models/
│   │   ├── base.py              # BaseModel with weighted loss support
│   │   └── transformer.py       # TransformerModel (load, predict, save)
│   ├── trainers/
│   │   ├── base.py              # WeightedLossTrainer, BaseTrainer
│   │   └── td_trainer.py        # TDTrainer — cross-validation, early stopping, ensemble
│   └── utils/
│       ├── inference.py         # InferenceEngine, EnsembleInferenceEngine
│       ├── onnx_inference.py    # OnnxInferenceEngine — CPU inference via ONNX Runtime
│       ├── metrics.py           # compute_metrics, confusion matrix, ROC plots
│       └── data_utils.py        # load_dataset, preprocess_text
├── pyproject.toml               # Packaging, tool config (black/isort/pytest), optional deps
├── requirements.txt             # Runtime dependencies
└── test-requirements.txt        # pytest, pytest-cov
```

---

## Web UI

A Gradio interface for fine-tuning and evaluation without the CLI:

```bash
pip install gradio
python app.py
```

Opens at `http://localhost:7077`. Supports CSV upload, model selection, training, and downloading predictions.

---

## Output Files

**After training:**
```
outputs/binary/
├── pytorch_model.bin
├── config.json
├── tokenizer files
├── training_config.json
├── metrics.json
├── confusion_matrix.png
├── roc_curve.png
└── emissions/emissions.csv
```

**After inference:**
```
outputs/binary/inference_YYYYMMDD_HHMMSS/
├── predictions_<input>.csv
├── metrics/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── roc_curve.png
└── emissions/inference_emissions.csv
```

---

## Contributing

Contributions are welcome. Please open a pull request with a clear description of the change.

## License

MIT — see [LICENSE](LICENSE).
