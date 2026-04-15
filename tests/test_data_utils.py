"""Tests for tdsuite/utils/data_utils.py."""

import json
import os

import pandas as pd
import pytest

from tdsuite.utils.data_utils import (
    load_dataset,
    load_dataset_from_file,
    preprocess_text,
)


# ---------------------------------------------------------------------------
# load_dataset_from_file
# ---------------------------------------------------------------------------

class TestLoadDatasetFromFile:
    def test_csv(self, csv_file):
        df = load_dataset_from_file(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert "text" in df.columns
        assert "label" in df.columns
        assert len(df) == 40

    def test_json(self, json_file):
        df = load_dataset_from_file(json_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 40

    def test_jsonl(self, jsonl_file):
        df = load_dataset_from_file(jsonl_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 40

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset_from_file("/nonexistent/path/data.csv")

    def test_unsupported_format_raises(self, tmp_path):
        bad_file = str(tmp_path / "data.parquet")
        open(bad_file, "w").close()
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataset_from_file(bad_file)


# ---------------------------------------------------------------------------
# load_dataset (auto-detect local vs HF)
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_local_csv(self, csv_file):
        df = load_dataset(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 40

    def test_local_jsonl(self, jsonl_file):
        df = load_dataset(jsonl_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 40

    def test_nonexistent_path_tries_hf(self, monkeypatch):
        """Non-existent path should attempt HF load; we mock it to avoid network."""
        import tdsuite.utils.data_utils as du

        def mock_hf(name, split="train"):
            return pd.DataFrame({"text": ["mock"], "label": [0]})

        monkeypatch.setattr(du, "load_dataset_from_huggingface", mock_hf)
        df = load_dataset("fake/hf-dataset")
        assert len(df) == 1
        assert df["text"].iloc[0] == "mock"


# ---------------------------------------------------------------------------
# preprocess_text
# ---------------------------------------------------------------------------

class TestPreprocessText:
    def test_short_text_unchanged_length(self):
        text = "Hello world"
        result = preprocess_text(text, max_length=512)
        # content preserved (whitespace may be normalised)
        assert "Hello" in result
        assert "world" in result

    def test_truncation(self):
        long_text = "a" * 1000
        result = preprocess_text(long_text, max_length=100)
        # After truncation to 100 chars the whitespace collapse leaves ≤100 chars
        assert len(result) <= 100

    def test_whitespace_normalised(self):
        text = "too   many    spaces"
        result = preprocess_text(text)
        assert "  " not in result  # no double spaces

    def test_empty_string(self):
        assert preprocess_text("") == ""

    def test_custom_max_length(self):
        text = "word " * 200  # 1000 chars
        result = preprocess_text(text, max_length=50)
        assert len(result) <= 50
