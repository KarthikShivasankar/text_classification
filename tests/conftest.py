"""Shared pytest fixtures for TDSuite tests."""

import json
import os
import tempfile

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Temporary directory helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    """Return a temporary directory path (pathlib.Path)."""
    return tmp_path


# ---------------------------------------------------------------------------
# Sample DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_df():
    """Balanced binary-label DataFrame (40 rows, 20 per class)."""
    texts = [f"This is a technical debt issue number {i}" for i in range(20)]
    texts += [f"This is a normal code comment number {i}" for i in range(20)]
    labels = [1] * 20 + [0] * 20
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture
def categorical_df():
    """Binary DataFrame with categorical labels ('td' / 'non_td')."""
    texts = [f"Debt example {i}" for i in range(30)]
    labels = ["td"] * 15 + ["non_td"] * 15
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture
def repo_df():
    """DataFrame with repo column for top-repo extraction tests."""
    rows = []
    for i in range(10):
        rows.append({"text": f"issue {i}", "label": 1, "repo": "owner/alpha"})
    for i in range(5):
        rows.append({"text": f"issue {10 + i}", "label": 1, "repo": "owner/beta"})
    for i in range(20):
        rows.append({"text": f"normal {i}", "label": 0, "repo": "owner/alpha"})
    for i in range(10):
        rows.append({"text": f"normal {20 + i}", "label": 0, "repo": "owner/beta"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Temporary CSV / JSON / JSONL files
# ---------------------------------------------------------------------------

@pytest.fixture
def csv_file(tmp_dir, binary_df):
    """Write binary_df to a temp CSV and return its path."""
    path = str(tmp_dir / "data.csv")
    binary_df.to_csv(path, index=False)
    return path


@pytest.fixture
def json_file(tmp_dir, binary_df):
    """Write binary_df to a temp JSON file and return its path."""
    path = str(tmp_dir / "data.json")
    binary_df.to_json(path)
    return path


@pytest.fixture
def jsonl_file(tmp_dir, binary_df):
    """Write binary_df to a temp JSONL file and return its path."""
    path = str(tmp_dir / "data.jsonl")
    binary_df.to_json(path, orient="records", lines=True)
    return path


@pytest.fixture
def predictions_df():
    """DataFrame mimicking inference output for metrics tests."""
    return pd.DataFrame({
        "text": [f"text {i}" for i in range(10)],
        "label": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
        "predicted_class": [1, 0, 1, 0, 1, 0, 0, 0, 1, 1],
        "predicted_probability": [0.9, 0.8, 0.85, 0.75, 0.95, 0.6, 0.7, 0.65, 0.88, 0.55],
        "class_probabilities": [
            [0.1, 0.9], [0.8, 0.2], [0.15, 0.85], [0.75, 0.25],
            [0.05, 0.95], [0.4, 0.6], [0.7, 0.3], [0.65, 0.35],
            [0.12, 0.88], [0.45, 0.55],
        ],
    })
