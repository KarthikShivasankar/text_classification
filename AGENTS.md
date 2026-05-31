# AGENTS.md

Guidance for AI coding agents and new contributors working in this repository.

## Project

**TDSuite** (`tdsuite`) is a text-classification framework for Technical Debt (TD)
detection using transformer models. **Inference defaults to ONNX on CPU — PyTorch is
optional** and only needed for training or explicit GPU inference. Published to PyPI as
[`tdsuite`](https://pypi.org/project/tdsuite/); models live under the `karths/` Hugging
Face namespace. Requires **Python ≥ 3.9**.

## Setup

```bash
# Preferred: uv (fast). Creates .venv/ and installs the dev + test toolchain.
uv venv
uv pip install -e ".[dev,test]"        # ruff, black, isort, flake8, pytest
# To also run/inspect the torch + datasets code paths and ONNX export:
uv pip install -e ".[train,onnx]"      # adds torch (CPU/GPU), datasets, onnx, onnxscript

# pip fallback
pip install -e ".[dev,test]"
```

> The default install (`pip install -e .`) is intentionally torch-free (ONNX only).
> Do not add top-level `import torch` / `from datasets import ...` to modules on the
> default import path (`tdsuite/inference.py`, `tdsuite/utils/onnx_inference.py`,
> `tdsuite/split_data.py`, and what they import). Import those lazily inside functions
> or behind `__getattr__` (see `tdsuite/utils/__init__.py` and `tdsuite/data/__init__.py`).

## Validate (run before every commit / PR)

```bash
ruff check tdsuite/ scripts/            # lint (primary)
black --check tdsuite/ scripts/         # formatting
isort --check-only tdsuite/ scripts/    # import order
flake8 tdsuite/ scripts/                # legacy linter (still kept green)
pytest                                   # 235 tests, fully offline (no GPU/network)
```

Auto-fix formatting/lint:

```bash
ruff check --fix tdsuite/ scripts/
black tdsuite/ scripts/
isort tdsuite/ scripts/
```

All of the above also run in CI (`.github/workflows/ci.yml`) on push/PR to `main`
across Python 3.9–3.12. Keep them green.

## Build & release

```bash
python -m build           # sdist + wheel into dist/
twine check dist/*        # validate package metadata
```

- **Version is single-sourced** from `tdsuite/__init__.py` (`__version__`); `pyproject.toml`
  reads it via `dynamic = ["version"]`. Bump it there only.
- Releasing is automated via **PyPI Trusted Publishing (OIDC)**: tag `vX.Y.Z`, push the
  tag, then publish a GitHub Release — `.github/workflows/release.yml` builds and uploads.
  No tokens are stored. PyPI versions are immutable; never reuse a version number.

## Conventions

- **All `argparse` parsers live in `tdsuite/cli.py`** (`get_*_parser()` functions). Add
  new CLI flags there, not in the entry-point scripts.
- `pyproject.toml` is the single source of truth for packaging, tool config, and optional
  dependency extras (`gpu`, `train`, `onnx`, `optimum`, `dev`, `test`). There is no `setup.py`.
- Line length **88**. Intentional pre-import statements (env vars / warnings filters set
  before imports) are marked `# noqa: E402`.
- Tests are offline: transformer/ONNX/torch calls are mocked with `unittest.mock`. New
  features need matching tests under `tests/`.
- Commit messages follow Conventional Commits (`feat:`, `fix:`, `docs:`, `build:`, `style:`…).

## Layout

```
tdsuite/        # package: cli, config, data, models, trainers, utils, entry points
scripts/        # GitHub-issues pipeline + ONNX export/verify helpers
tests/          # pytest suite (offline)
.github/workflows/  # ci.yml (lint+test+build), release.yml (trusted publishing)
pyproject.toml  # packaging + tool config (ruff/black/isort/pytest) + extras
```

See `CLAUDE.md` for a deeper architecture walkthrough and `README.md` for usage.
