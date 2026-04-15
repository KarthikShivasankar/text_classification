# TD-Classifier Suite

A suite for detecting and classifying **technical debt** in software repositories using transformer models. It covers 18 TD categories (architecture, security, performance, code quality, and more), ships 17 pre-trained models on Hugging Face, and works end-to-end from raw GitHub issues to structured predictions — with or without a GPU.

---

## Table of Contents

- [What is Technical Debt?](#what-is-technical-debt)
- [Use Cases](#use-cases)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Use Case Walkthroughs](#use-case-walkthroughs)
  - [Scan a GitHub repo for technical debt](#use-case-1--scan-a-github-repo-for-technical-debt)
  - [Classify your own dataset](#use-case-2--classify-your-own-dataset)
  - [Fine-tune a model on your codebase](#use-case-3--fine-tune-a-model-on-your-codebase)
  - [Run without a GPU](#use-case-4--run-without-a-gpu)
  - [Detect multiple TD types at once](#use-case-5--detect-multiple-td-types-ensemble)
  - [Use the web UI](#use-case-6--use-the-web-ui)
- [Hugging Face Models & Datasets](#hugging-face-models--datasets)
- [Full CLI Reference](#full-cli-reference)
  - [tdsuite-split-data](#tdsuite-split-data)
  - [tdsuite-train](#tdsuite-train)
  - [tdsuite-inference](#tdsuite-inference)
  - [export_onnx.py](#export_onnxpy)
  - [fetch_github_issues.py](#fetch_github_issuespy)
  - [extract_issue_bodies.py](#extract_issue_bodiespy)
- [CI/CD Integration](#cicd-integration)
  - [GitHub Actions — PR TD check](#github-actions--pr-td-check)
  - [GitHub Actions — nightly repo scan](#github-actions--nightly-repo-scan)
  - [GitLab CI](#gitlab-ci)
  - [Docker / self-hosted runners](#docker--self-hosted-runners)
- [Testing](#testing)
  - [Running the test suite](#running-the-test-suite)
  - [Test coverage by module](#test-coverage-by-module)
  - [What each test file covers](#what-each-test-file-covers)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---


## Use Cases

| Scenario | How this tool helps |
|----------|---------------------|
| **Repository health audit** | Fetch all issues from a GitHub repo and flag which ones describe technical debt |
| **Issue triage** | Automatically tag incoming issues so engineers know which ones carry TD risk |
| **Research / metrics** | Measure TD density across projects, releases, or teams over time |
| **Custom classifier** | Fine-tune on your own labelled dataset to match your team's definition of TD |
| **CI / pre-merge checks** | Pipe PR description or commit message through the model to surface TD before merge |
| **Multi-category analysis** | Run an ensemble to detect *which type* of TD is present (security, performance, etc.) |

---

## Features

- **18 TD categories** — general TD, architecture, code quality, security, performance, defects, infrastructure, requirements, design, usability, compatibility, reliability, process, build, maintenance, automation, people, portability
- **17 pre-trained models** on Hugging Face Hub — zero training required for inference
- **ONNX-first inference** — CPU by default, no PyTorch required; all 17 models ship `model.onnx` on Hugging Face Hub — auto-downloaded on first use
- **GitHub issues pipeline** — fetch → clean → classify in three commands
- **Custom training** — fine-tune on your own data with cross-validation, class weighting, and early stopping
- **Ensemble inference** — combine multiple category models with custom weights
- **Carbon tracking** — CodeCarbon emissions tracking on every training and inference run
- **Gradio web UI** — browser-based interface for non-CLI users

---

## Installation

> **Default backend is ONNX (CPU).** PyTorch and CUDA are optional extras — only needed for training or explicit GPU inference.

### With UV (recommended)

[UV](https://docs.astral.sh/uv/) is a fast Python package manager that replaces pip + venv.

```bash
# Install UV — macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install UV — Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
git clone https://github.com/KarthikShivasankar/text_classification
cd text_classification

uv venv                        # create .venv/
uv pip install -e .            # CPU inference (ONNX) — no GPU / PyTorch required
uv pip install -e ".[gpu]"     # + GPU inference via onnxruntime-gpu + torch (CUDA 12.4)
uv pip install -e ".[train]"   # + full training stack (torch, codecarbon, evaluate…)
uv pip install -e ".[onnx]"    # + onnx package for exporting your own models
uv pip install -e ".[dev]"     # + black, isort, flake8
```

> After `uv venv`, set your IDE's Python interpreter to `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux/Mac) so imports resolve correctly.

### With pip

```bash
git clone https://github.com/KarthikShivasankar/text_classification
cd text_classification
pip install -e .                       # CPU inference — no GPU needed
pip install -e ".[gpu,dev,test]"       # GPU + dev + test extras
```

### CPU-only (no GPU, inference only)

The default `pip install -e .` already gives you full CPU inference via ONNX. No additional steps required — `model.onnx` is auto-downloaded from Hugging Face Hub the first time you run inference:

```bash
pip install -e .
# Run inference directly — model.onnx downloads automatically on first use
tdsuite-inference --model_name karths/binary_classification_train_TD \
    --text "The auth module has no rate limiting"
```

### Publishing to PyPI

```bash
uv pip install build twine
python -m build
twine upload dist/*
```

---

## Quick Start

**Classify issues from a public GitHub repo in under 5 minutes — no GPU needed:**

```bash
# Install (CPU-only, no PyTorch required)
git clone https://github.com/KarthikShivasankar/text_classification && cd text_classification
uv venv && uv pip install -e .

# Fetch the 100 most recent issues
python scripts/fetch_github_issues.py --repo microsoft/vscode --output issues.csv

# Extract the body text
python scripts/extract_issue_bodies.py --input issues.csv --output issue_texts.csv --min-length 50

# Classify — model.onnx is auto-downloaded from Hugging Face Hub on first run
tdsuite-inference \
    --model_name karths/binary_classification_train_TD \
    --input_file issue_texts.csv
```

Results land in a timestamped folder: `outputs/.../inference_YYYYMMDD_HHMMSS/predictions_issue_texts.csv`

**GPU inference** (requires `pip install -e ".[gpu]"`):

```bash
tdsuite-inference \
    --model_name karths/binary_classification_train_TD \
    --device cuda \
    --input_file issue_texts.csv
```

---

## Use Case Walkthroughs

### Use Case 1 — Scan a GitHub repo for technical debt

Audit any public repository without writing any code.

```bash
# Step 1: fetch issues (default: latest 100, newest-first)
python scripts/fetch_github_issues.py \
    --repo owner/repo \
    --output issues.csv

# Fetch more issues
python scripts/fetch_github_issues.py --repo owner/repo --limit 500 --output issues.csv

# Fetch everything (slow on large repos)
python scripts/fetch_github_issues.py --repo owner/repo --all --output issues.csv
```

> **Rate limit?** Unauthenticated calls are capped at 60/hour. Add `--token $GITHUB_TOKEN` to raise this to 5,000/hour. Create a token (no scopes needed for public repos) at https://github.com/settings/tokens. The script prints the exact wait time and that link automatically when the limit is hit.

```bash
# Step 2: clean — keep only body text, drop short/duplicate entries
python scripts/extract_issue_bodies.py \
    --input issues.csv \
    --output issue_texts.csv \
    --min-length 50 \
    --drop-duplicates \
    --keep-metadata        # also keep 'number' and 'title' columns for traceability

# Step 3: classify — ONNX CPU by default, model.onnx auto-downloads from HF Hub
tdsuite-inference \
    --model_name karths/binary_classification_train_TD \
    --input_file issue_texts.csv

# Step 3 (GPU): requires pip install 'tdsuite[gpu]'
tdsuite-inference \
    --model_name karths/binary_classification_train_TD \
    --device cuda \
    --input_file issue_texts.csv
```

The output CSV has one row per issue with `predicted_class` (0/1) and `predicted_probability`. If you used `--keep-metadata` the `number` and `title` columns let you trace results directly back to GitHub issues.

---

### Use Case 2 — Classify your own dataset

You have a CSV of issue bodies, commit messages, or code review comments and want predictions.

```bash
# Your CSV must have a column with text. Default column name is 'text'.
# Example: my_issues.csv
#   text,project
#   "This module has no unit tests",projectA
#   "Hard-coded credentials in config.py",projectA

tdsuite-inference \
    --model_name karths/binary_classification_train_TD \
    --input_file my_issues.csv \
    --text_column text \
    --batch_size 64 \
    --output_file results/predictions.csv
```

If your CSV also has a `label` column with ground truth (0/1), metrics are computed automatically and saved alongside the predictions:

```bash
tdsuite-inference \
    --model_name karths/binary_classification_train_TD \
    --input_file labelled_data.csv \
    --output_file results/predictions.csv
# → results are saved + metrics.json, confusion_matrix.png, roc_curve.png
```

**Classify a single string:**

```bash
tdsuite-inference \
    --model_name karths/binary_classification_train_TD \
    --text "The authentication module has no rate limiting and stores passwords in plain text"
```

Output:
```json
{
  "text": "The authentication module has no rate limiting...",
  "predicted_class": 1,
  "predicted_probability": 0.94,
  "class_probabilities": [0.06, 0.94]
}
```

---

### Use Case 3 — Fine-tune a model on your codebase

Use your own labelled data to build a classifier tuned to your team's codebase or TD definition.

**Prepare your data**

Your training CSV needs a `text` column and a `label` column. Labels can be:
- **Numeric** (0 = not TD, 1 = TD): use `--numeric_labels`
- **Categorical** (e.g. "TD", "non-TD"): use `--positive_category TD`

```csv
text,label
"No input validation on the API endpoint",TD
"Refactored the payment module",non-TD
"Missing error handling in the database layer",TD
```

**Option A: train on a local file**

```bash
tdsuite-train \
    --data_file data/my_labelled_issues.csv \
    --model_name distilbert-base-uncased \
    --positive_category TD \
    --output_dir outputs/my_model \
    --num_epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --warmup_steps 500
```

**Option B: train on a Hugging Face dataset**

```bash
tdsuite-train \
    --data_file karths/binary-10IQR-TD \
    --model_name distilbert-base-uncased \
    --numeric_labels \
    --output_dir outputs/td_model \
    --num_epochs 5 \
    --batch_size 16
```

**Option C: cross-validation (recommended for small datasets)**

```bash
tdsuite-train \
    --data_file data/my_labelled_issues.csv \
    --model_name distilbert-base-uncased \
    --positive_category TD \
    --output_dir outputs/cv_model \
    --cross_validation \
    --n_splits 5 \
    --num_epochs 3
```

Each fold saves its own metrics and visualisations. After training, run inference with your new model:

```bash
tdsuite-inference \
    --model_path outputs/my_model \
    --input_file data/unlabelled.csv
```

> **No GPU?** Training on CPU is possible but slow. The recommended approach is to fine-tune on a GPU machine (e.g. Google Colab), then export to ONNX for local CPU inference.

---

### Use Case 4 — Run without a GPU

CPU inference is the **default** — no GPU, no PyTorch, no extra steps. All 21 pre-trained models ship a `model.onnx` file on Hugging Face Hub that downloads automatically on the first inference call.

**Install (CPU — no GPU required)**

```bash
pip install -e .   # onnxruntime is included; torch is NOT required
```

**Classify on CPU — model downloads automatically**

```bash
# Batch file — model.onnx auto-downloaded from HF Hub on first run
tdsuite-inference \
    --model_name karths/binary_classification_train_TD \
    --input_file issue_texts.csv

# Single string
tdsuite-inference \
    --model_name karths/binary_classification_train_TD \
    --text "No tests exist for this module"
```

**Use a local ONNX file (offline / custom model)**

```bash
# Export your own fine-tuned model once (requires pip install 'tdsuite[onnx]')
python scripts/export_onnx.py \
    --model_path outputs/my_model \
    --output models/my_model.onnx

# Then run offline
tdsuite-inference --onnx_path models/my_model.onnx --input_file issue_texts.csv
```

ONNX Runtime is typically **2–4× faster** than PyTorch on CPU and has no dependency on CUDA or torch.

---

### Use Case 5 — Detect multiple TD types (ensemble)

Run several category-specific models in parallel and combine their predictions. Useful when you want to know not just *whether* an issue is TD, but *what kind*.

```bash
# Weighted ensemble: general TD + security + code quality
tdsuite-inference \
    --model_names \
        karths/binary_classification_train_TD \
        karths/binary_classification_train_secu \
        karths/binary_classification_train_code \
    --input_file issue_texts.csv \
    --weights 0.5 0.3 0.2 \
    --output_file results/ensemble_predictions.csv
```

If `--weights` is omitted, equal weights are applied automatically. The final prediction is a weighted average of each model's class probabilities.

**Local ensemble (after training multiple models):**

```bash
tdsuite-inference \
    --model_paths \
        outputs/fold_0 \
        outputs/fold_1 \
        outputs/fold_2 \
    --input_file test_data.csv
```

---

### Use Case 6 — Use the web UI

A Gradio interface is available for teams that prefer not to use the CLI.

```bash
pip install gradio
python app.py
# → opens at http://localhost:7077
```

**Fine-tune tab:**
1. Upload a labelled CSV (text + label columns)
2. Select a base model and set train/test split
3. Click Fine-tune — view accuracy, classification report, confusion matrix inline
4. Download the predictions CSV

**Evaluate tab:**
1. Upload an unlabelled CSV
2. Select one or more pre-trained models (General TD, Code Quality, Types)
3. Run — download predictions with per-class probabilities

---

## Hugging Face Models & Datasets

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

---

## Full CLI Reference

### `tdsuite-split-data`

Splits a dataset into balanced train/test sets and optionally extracts top-contributing repositories.

```bash
tdsuite-split-data \
    --data_file karths/binary-10IQR-TD \
    --output_dir data/split \
    --is_numeric_labels \
    --repo_column repo \
    --is_huggingface_dataset
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_file` | *(required)* | Local file path (CSV/JSON/JSONL) or HF dataset name |
| `--output_dir` | *(required)* | Directory for train.csv, test.csv, top_repos.csv |
| `--test_size` | `0.2` | Fraction of data for the test split |
| `--random_state` | `42` | Random seed |
| `--repo_column` | — | Column containing repository names (enables top-repo extraction) |
| `--is_huggingface_dataset` | `false` | Load from Hugging Face Hub |
| `--is_numeric_labels` | `false` | Labels are already 0/1 integers |

---

### `tdsuite-train`

Fine-tune a transformer model on binary TD classification data.

```bash
tdsuite-train \
    --data_file karths/binary-10IQR-TD \
    --model_name distilbert-base-uncased \
    --numeric_labels \
    --output_dir outputs/my_model \
    --num_epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --warmup_steps 1000
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_file` | *(required)* | Local file path or HF dataset name |
| `--model_name` | *(required)* | HF model ID or local path |
| `--output_dir` | *(required)* | Save directory for model and metrics |
| `--text_column` | `text` | Text column name |
| `--label_column` | `label` | Label column name |
| `--positive_category` | — | Label string for the positive class (categorical labels) |
| `--numeric_labels` | `false` | Labels are already 0/1 integers |
| `--is_huggingface_dataset` | `false` | Load dataset from Hugging Face Hub |
| `--num_epochs` | `3` | Training epochs |
| `--batch_size` | `16` | Per-device batch size |
| `--learning_rate` | `2e-5` | Peak learning rate |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--warmup_steps` | `500` | Linear LR warmup steps |
| `--gradient_accumulation_steps` | `1` | Steps before a weight update |
| `--cross_validation` | `false` | Enable k-fold CV |
| `--n_splits` | `5` | Number of CV folds |
| `--max_length` | `512` | Maximum token sequence length |
| `--seed` | `42` | Random seed |
| `--device` | auto | `cuda` or `cpu` |

---

### `tdsuite-inference`

Run predictions on a file or a single string. Defaults to **ONNX on CPU** — PyTorch is not required.

```bash
# CPU inference (default) — model.onnx auto-downloaded from HF Hub
tdsuite-inference --model_name karths/binary_classification_train_TD --input_file issues.csv

# Single string
tdsuite-inference --model_name karths/binary_classification_train_TD \
    --text "No input validation on this endpoint"

# GPU inference — ONNX with CUDAExecutionProvider (requires pip install 'tdsuite[gpu]')
tdsuite-inference --model_name karths/binary_classification_train_TD \
    --device cuda --input_file issues.csv

# Local ONNX file (offline)
tdsuite-inference --onnx_path models/td.onnx --input_file issues.csv

# Local model checkpoint (PyTorch, requires --use_torch)
tdsuite-inference --model_path outputs/my_model --use_torch --input_file issues.csv

# Ensemble (uses PyTorch backend automatically)
tdsuite-inference \
    --model_names karths/binary_classification_train_TD karths/binary_classification_train_secu \
    --input_file issues.csv \
    --weights 0.6 0.4
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | — | Local model directory (ONNX auto-detected; or use `--use_torch`) |
| `--model_name` | — | HF model name — `model.onnx` downloaded automatically |
| `--model_paths` | — | Multiple local directories (ensemble, PyTorch) |
| `--model_names` | — | Multiple HF model names (ensemble, PyTorch) |
| `--onnx_path` | — | Explicit path to a local `.onnx` file |
| `--use_torch` | `false` | Force PyTorch backend (requires `pip install 'tdsuite[gpu]'`) |
| `--text` | — | Single text string to classify |
| `--input_file` | — | CSV or JSON file to classify |
| `--text_column` | `text` | Column containing text |
| `--output_file` | — | Path to save the predictions CSV |
| `--results_dir` | — | Custom results directory (default: timestamped subfolder) |
| `--batch_size` | `32` | Inference batch size |
| `--max_length` | `512` | Max token length |
| `--device` | `cpu` | `cpu` (ONNX CPU) or `cuda` (ONNX GPU via CUDAExecutionProvider) |
| `--weights` | — | Per-model weights for ensemble averaging |
| `--disable_progress_bar` | `false` | Suppress tqdm bars |
| `--track_emissions` | `true` | Record carbon emissions via CodeCarbon |

> `--onnx_path`, `--model_path`, `--model_name`, `--model_paths`, and `--model_names` are mutually exclusive.

---

### `export_onnx.py`

Export a custom or fine-tuned model to ONNX format. The 17 pre-trained models already have `model.onnx` on Hugging Face Hub — you only need this for your own fine-tuned models.

> Requires: `pip install 'tdsuite[onnx]'` (adds `torch` + `onnx` + `onnxscript`)

```bash
# Export a local fine-tuned checkpoint
python scripts/export_onnx.py --model_path outputs/my_model --output models/my_model.onnx

# Export directly from Hugging Face (e.g. your own model)
python scripts/export_onnx.py \
    --model_name my-org/my-custom-td-model \
    --output models/custom.onnx
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | — | Local model directory *(mutually exclusive with `--model_name`)* |
| `--model_name` | — | HF model name *(mutually exclusive with `--model_path`)* |
| `--output` | *(required)* | Destination `.onnx` file path |
| `--max_length` | `512` | Sequence length for the export dummy input |
| `--opset` | `14` | ONNX opset version |

The tokenizer is saved alongside the `.onnx` file automatically.

---

### `fetch_github_issues.py`

Fetch issues from any public GitHub repository into a CSV file.

```bash
python scripts/fetch_github_issues.py --repo owner/repo --output issues.csv
python scripts/fetch_github_issues.py --repo owner/repo --limit 500 --output issues.csv
python scripts/fetch_github_issues.py --repo owner/repo --all --output issues.csv
python scripts/fetch_github_issues.py --repo owner/repo --token "$GITHUB_TOKEN" --output issues.csv
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--repo` | *(required)* | Repository in `owner/repo` format |
| `--output` | `issues.csv` | Output CSV file path |
| `--state` | `all` | Issue state: `open`, `closed`, or `all` |
| `--limit` | `100` | Max issues to fetch, newest-first *(mutually exclusive with `--all`)* |
| `--all` | `false` | Fetch every issue across all pages *(mutually exclusive with `--limit`)* |
| `--token` | — | GitHub personal access token (raises rate limit from 60 to 5,000 req/h) |

Output columns: `id`, `number`, `title`, `body`, `state`, `created_at`, `updated_at`, `closed_at`, `user_login`, `labels`, `comments`, `url`.

---

### `extract_issue_bodies.py`

Clean a GitHub issues CSV down to a `text` column ready for `tdsuite-inference`.

```bash
python scripts/extract_issue_bodies.py \
    --input issues.csv \
    --output issue_texts.csv \
    --min-length 50 \
    --drop-duplicates \
    --keep-metadata
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | *(required)* | Input CSV (from `fetch_github_issues.py`) |
| `--output` | `issue_texts.csv` | Output CSV path |
| `--body-column` | `body` | Source column for issue text |
| `--min-length` | `20` | Drop rows shorter than N characters |
| `--drop-duplicates` | `false` | Remove duplicate body texts |
| `--keep-metadata` | `false` | Also retain `number` and `title` columns |

---

## CI/CD Integration

TD-Classifier Suite works well as a gate in automated pipelines. The recommended approach for CI is **ONNX inference** — it requires no GPU, has no PyTorch dependency, and cold-starts fast enough for pull-request checks.

### GitHub Actions — PR TD check

Flags a pull request if its description or changed commit messages contain technical debt language. The ONNX model is cached between runs so subsequent jobs skip the download.

```yaml
# .github/workflows/td-check.yml
name: Technical Debt Check

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  td-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install tdsuite (CPU / ONNX — no GPU required)
        run: pip install -e .

      - name: Cache downloaded ONNX model
        uses: actions/cache@v4
        with:
          # HF Hub caches to ~/.cache/huggingface; cache it between runs
          path: ~/.cache/huggingface
          key: hf-onnx-td-v1

      - name: Write PR description to file
        env:
          PR_BODY: ${{ github.event.pull_request.body }}
        run: |
          echo "$PR_BODY" > /tmp/pr_text.txt

      - name: Classify PR description
        id: classify
        run: |
          # model.onnx auto-downloads from HF Hub on first run (cached above)
          result=$(tdsuite-inference \
            --model_name karths/binary_classification_train_TD \
            --text "$(cat /tmp/pr_text.txt)")
          echo "$result"
          # Fail if predicted_class == 1 (TD detected)
          echo "$result" | python -c "
          import sys, json
          data = json.load(sys.stdin)
          if data['predicted_class'] == 1:
              print(f\"::warning::TD detected (confidence {data['predicted_probability']:.0%}). Review before merging.\")
              sys.exit(1)
          "
```

> Change `sys.exit(1)` to `sys.exit(0)` if you want informational warnings without blocking merges.

---

### GitHub Actions — nightly repo scan

Runs a full issues scan on a schedule and uploads the results as a workflow artifact.

```yaml
# .github/workflows/td-nightly.yml
name: Nightly TD Scan

on:
  schedule:
    - cron: "0 2 * * *"   # 02:00 UTC every day
  workflow_dispatch:        # allow manual runs

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install tdsuite (CPU / ONNX — no GPU required)
        run: pip install -e .

      - name: Cache HF Hub downloads
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: hf-onnx-td-v1

      - name: Fetch issues
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          python scripts/fetch_github_issues.py \
            --repo ${{ github.repository }} \
            --token "$GITHUB_TOKEN" \
            --limit 200 \
            --output /tmp/issues.csv

      - name: Clean issues
        run: |
          python scripts/extract_issue_bodies.py \
            --input /tmp/issues.csv \
            --output /tmp/issue_texts.csv \
            --min-length 50 \
            --drop-duplicates \
            --keep-metadata

      - name: Classify (model.onnx auto-downloads from HF Hub)
        run: |
          tdsuite-inference \
            --model_name karths/binary_classification_train_TD \
            --input_file /tmp/issue_texts.csv \
            --output_file /tmp/td_predictions.csv

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: td-scan-${{ github.run_id }}
          path: /tmp/td_predictions.csv
          retention-days: 30
```

---

### GitLab CI

```yaml
# .gitlab-ci.yml (relevant excerpt)
variables:
  MODEL_CACHE: "$CI_PROJECT_DIR/.cache/onnx"

td-check:
  stage: test
  image: python:3.11-slim
  cache:
    key: hf-onnx-td
    paths:
      - ~/.cache/huggingface/
  before_script:
    - pip install -e . -q
  script:
    - |
      # model.onnx auto-downloads from HF Hub (cached between runs)
      tdsuite-inference \
        --model_name karths/binary_classification_train_TD \
        --text "$CI_MERGE_REQUEST_DESCRIPTION" \
        | python -c "
      import sys, json
      data = json.load(sys.stdin)
      prob = data['predicted_probability']
      if data['predicted_class'] == 1:
          print(f'TD detected — confidence {prob:.0%}')
          sys.exit(1)
      print(f'No TD detected (confidence {1-prob:.0%})')
      "
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
```

---

### Docker / self-hosted runners

For air-gapped or self-hosted environments, bake the ONNX model into your runner image so the download step is eliminated entirely.

```dockerfile
# Dockerfile.runner — bake model.onnx into the image for offline / air-gapped use
FROM python:3.11-slim

WORKDIR /app
COPY . .
# Install tdsuite (CPU, no GPU), then pre-download model.onnx into the image
RUN pip install -e . && \
    python -c "from tdsuite.utils.onnx_inference import OnnxInferenceEngine; \
               OnnxInferenceEngine.from_pretrained('karths/binary_classification_train_TD')"

ENTRYPOINT ["tdsuite-inference"]
```

```bash
# Build once (model baked in — no network at runtime)
docker build -f Dockerfile.runner -t tdsuite-runner:latest .

# Use in any CI job
docker run --rm tdsuite-runner:latest \
    --model_name karths/binary_classification_train_TD \
    --text "Hard-coded API keys in the config module"
```

**Tips for CI environments:**

| Concern | Recommendation |
|---------|----------------|
| Cold-start time | Cache `~/.cache/huggingface` between runs — model.onnx is ~250–500 MB |
| No GPU available | Default ONNX backend runs entirely on CPU — no extra flags needed |
| GPU available | Add `--device cuda` and install `pip install 'tdsuite[gpu]'` |
| Blocking vs. warning | Set `sys.exit(0)` for informational-only checks |
| Multiple TD categories | Add `--model_names` with category models to detect *which type* of TD |
| Rate limits on issue fetch | Store `GITHUB_TOKEN` as a CI secret and pass via `--token` |
| Air-gapped networks | Pre-bake the model in a Docker image as shown above |

---

## Output Files

**After training:**

```
outputs/my_model/
├── pytorch_model.bin          # model weights
├── config.json                # HF model config
├── tokenizer_config.json      # tokenizer files
├── training_config.json       # CLI args used for this run
├── metrics.json               # accuracy, F1, MCC, AUC-ROC
├── confusion_matrix.png
├── roc_curve.png
└── emissions/
    └── emissions.csv          # CodeCarbon energy/CO2 data
```

**After cross-validation training:**

```
outputs/cv_model/
├── fold_0/ … fold_N/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── cross_validation_results.json
└── cross_validation_visualization.png
```

**After inference:**

```
outputs/my_model/inference_YYYYMMDD_HHMMSS/
├── predictions_<input_filename>.csv   # original columns + predicted_class + probabilities
├── metrics/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── roc_curve.png                  # only generated when ground truth labels are present
└── emissions/
    └── inference_emissions.csv
```

---

## Project Structure

```
text_classification/
├── scripts/
│   ├── fetch_github_issues.py        # fetch issues from any public GitHub repo → CSV
│   ├── extract_issue_bodies.py       # clean issue CSV → text column for inference
│   ├── export_onnx.py                # export a single model to ONNX (one-time, needs torch)
│   └── export_and_upload_onnx.py    # batch-export all 17 TDSuite models and upload to HF Hub
├── tdsuite/
│   ├── cli.py                        # all argparse parsers (single source of truth)
│   ├── train.py                      # tdsuite-train entry point
│   ├── inference.py                  # tdsuite-inference entry point (ONNX default)
│   ├── split_data.py                 # tdsuite-split-data entry point
│   ├── upload_to_hf.py               # upload a trained model to Hugging Face Hub
│   ├── config/
│   │   └── config.py                 # ModelConfig, TrainingConfig, DataConfig, InferenceConfig
│   ├── data/
│   │   ├── dataset.py                # TDDataset, TDProcessor, BinaryTDProcessor
│   │   └── data_splitter.py          # DataSplitter — balanced splits, top-repo extraction
│   ├── models/
│   │   ├── base.py                   # BaseModel with weighted loss support
│   │   └── transformer.py            # TransformerModel (load, predict, save)
│   ├── trainers/
│   │   ├── base.py                   # WeightedLossTrainer, BaseTrainer (emissions tracking)
│   │   └── td_trainer.py             # TDTrainer — cross-validation, early stopping, ensemble
│   └── utils/
│       ├── onnx_inference.py         # OnnxInferenceEngine — default CPU/GPU inference (no torch)
│       ├── inference.py              # InferenceEngine, EnsembleInferenceEngine (PyTorch)
│       ├── metrics.py                # compute_metrics, confusion matrix, ROC plots
│       └── data_utils.py             # load_dataset, preprocess_text
├── app.py                            # Gradio web UI (port 7077)
├── pyproject.toml                    # packaging, tool config, optional deps (gpu/train/onnx/dev)
└── test-requirements.txt             # pytest, pytest-cov
```

---

---

## Testing

The test suite uses **pytest** with coverage reporting. All tests run entirely offline — no GPU, no HuggingFace model downloads. Transformer model calls are mocked with `unittest.mock`.

### Running the test suite

**Install test dependencies first:**

```bash
# UV (recommended)
uv pip install -r test-requirements.txt

# pip fallback
pip install pytest pytest-cov
```

**Run all tests with coverage:**

```bash
pytest
```

This uses the config in `pyproject.toml` (`testpaths = ["tests"]`, `--cov=tdsuite`).

**Run a specific test file:**

```bash
pytest tests/test_config.py -v
pytest tests/test_data_splitter.py -v
pytest tests/test_metrics.py -v
```

**Run a specific test class or function:**

```bash
pytest tests/test_config.py::TestModelConfig -v
pytest tests/test_cli.py::TestInferenceParser::test_ensemble_model_paths -v
```

**Run with detailed coverage report:**

```bash
pytest --cov=tdsuite --cov-report=term-missing --cov-report=html
# Open htmlcov/index.html in a browser for the line-by-line report
```

**Run only fast unit tests (skip any marked slow):**

```bash
pytest -m "not slow" -v
```

**Expected output (all passing):**

```
tests/test_cli.py                      ............. 35 passed
tests/test_config.py                   ............. 22 passed
tests/test_data_splitter.py            ............. 17 passed
tests/test_data_utils.py               ............. 10 passed
tests/test_dataset.py                  ............. 22 passed
tests/test_extract_issue_bodies.py     ............. 20 passed
tests/test_inference.py                ............. 22 passed
tests/test_metrics.py                  ............. 14 passed
```

### Test coverage by module

| Module | Test file | What is tested |
|---|---|---|
| `tdsuite/config/config.py` | `test_config.py` | `ModelConfig`, `TrainingConfig`, `DataConfig`, `InferenceConfig`, `Config` — defaults, `to_dict`, `from_dict`, `save`, `load` |
| `tdsuite/utils/data_utils.py` | `test_data_utils.py` | `load_dataset_from_file` (CSV/JSON/JSONL), `load_dataset` auto-detection, `preprocess_text` truncation & whitespace |
| `tdsuite/data/dataset.py` | `test_dataset.py` | `TDDataset` indexing & length; `TDProcessor` load/tokenize; `BinaryTDProcessor` binary conversion, `extract_top_repo`, `extract_top_repos_by_category` |
| `tdsuite/data/data_splitter.py` | `test_data_splitter.py` | `DataSplitter` load, preprocess, `balance_classes`, `split_and_save` with/without repo column; standalone `split_data()` |
| `tdsuite/utils/metrics.py` | `test_metrics.py` | `compute_metrics` return structure, value ranges, perfect-prediction case, JSON/PNG outputs |
| `tdsuite/cli.py` | `test_cli.py` | All six `get_*_parser()` functions — required args, defaults, flags, mutually-exclusive groups, error cases |
| `scripts/extract_issue_bodies.py` | `test_extract_issue_bodies.py` | `clean_text` — code blocks, HTML, Markdown, URLs, emoji, whitespace; full CSV pipeline |
| `tdsuite/utils/inference.py` | `test_inference.py` | `InferenceEngine` `predict_single`, `predict_batch`, `predict_from_file`; `EnsembleInferenceEngine` init, weight normalisation, `predict_single`, `predict_batch` — all mocked |

### What each test file covers

**`tests/conftest.py`** — shared fixtures: `binary_df`, `categorical_df`, `repo_df`, `csv_file`, `json_file`, `jsonl_file`, `predictions_df`.

**`tests/test_config.py`** — verifies that every config dataclass serialises to/from JSON correctly and that `save()`/`load()` round-trips work for both file and directory targets.

**`tests/test_data_utils.py`** — verifies that local CSV/JSON/JSONL files load correctly, that missing files raise `FileNotFoundError`, unsupported formats raise `ValueError`, and that `preprocess_text` truncates and collapses whitespace.

**`tests/test_dataset.py`** — verifies `TDDataset` item shapes and label values; `TDProcessor` calls the tokenizer with the correct kwargs; `BinaryTDProcessor` converts categorical labels, raises on missing `positive_category`, and correctly extracts top repos.

**`tests/test_data_splitter.py`** — verifies train/test split files are created, sizes are approximately correct (±5%), train and test sets do not overlap, top-repo extraction writes `top_repos.csv`, and label mappings are saved for categorical data.

**`tests/test_metrics.py`** — verifies all metric keys are present, values are in `[0, 1]`, perfect predictions yield 1.0 for accuracy/F1/MCC, `metrics.json` and PNG plots are written when `output_dir` is given, and nothing is written when `output_dir` is `None`.

**`tests/test_cli.py`** — parses argument vectors for all six CLI commands and asserts correct defaults, custom values, and that `SystemExit` is raised for invalid/missing arguments and mutually-exclusive conflicts.

**`tests/test_extract_issue_bodies.py`** — calls `clean_text()` with various inputs (fenced code blocks, inline code, HTML tags, markdown links, images, headings, bold, lists, blockquotes, emoji, extra whitespace) and verifies both removal of noise and preservation of prose; also exercises the CSV pipeline and deduplication logic.

**`tests/test_inference.py`** — patches `TransformerModel` and `AutoTokenizer` (for `InferenceEngine`) and `AutoModelForSequenceClassification`/`AutoTokenizer` (for `EnsembleInferenceEngine`) to avoid any network access or GPU requirement; verifies output structure, probability ranges, file I/O, and error handling.

---

## Contributing

Contributions are welcome. Please open a pull request with a clear description of the change and ensure `flake8` passes:

```bash
flake8 tdsuite/ scripts/
```

## License

MIT — see [LICENSE](LICENSE).
