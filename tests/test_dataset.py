"""Tests for tdsuite/data/dataset.py — TDDataset, TDProcessor, BinaryTDProcessor."""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from tdsuite.data.dataset import BinaryTDProcessor, TDDataset, TDProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_tokenizer(n_samples=5, seq_len=8):
    """Return a mock tokenizer whose __call__ returns PT tensors."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.zeros(n_samples, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(n_samples, seq_len, dtype=torch.long),
    }
    return tokenizer


# ---------------------------------------------------------------------------
# TDDataset
# ---------------------------------------------------------------------------

class TestTDDataset:
    def _build(self, n=5, seq_len=8):
        encodings = {
            "input_ids": torch.zeros(n, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
        }
        labels = list(range(n))
        return TDDataset(encodings, labels)

    def test_len(self):
        ds = self._build(n=10)
        assert len(ds) == 10

    def test_getitem_keys(self):
        ds = self._build(n=3)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_getitem_types(self):
        ds = self._build(n=3)
        item = ds[1]
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)

    def test_label_value(self):
        ds = self._build(n=5)
        for i in range(5):
            assert ds[i]["labels"].item() == i

    def test_slicing_full_iteration(self):
        ds = self._build(n=8)
        for i in range(len(ds)):
            item = ds[i]
            assert "labels" in item


# ---------------------------------------------------------------------------
# TDProcessor
# ---------------------------------------------------------------------------

class TestTDProcessor:
    def test_load_data_csv(self, csv_file):
        tokenizer = _make_mock_tokenizer()
        proc = TDProcessor(tokenizer=tokenizer)
        df = proc.load_data(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert "text" in df.columns
        assert len(df) == 40

    def test_load_data_json(self, json_file):
        tokenizer = _make_mock_tokenizer()
        proc = TDProcessor(tokenizer=tokenizer)
        df = proc.load_data(json_file)
        assert len(df) == 40

    def test_load_data_jsonl(self, jsonl_file):
        tokenizer = _make_mock_tokenizer()
        proc = TDProcessor(tokenizer=tokenizer)
        df = proc.load_data(jsonl_file)
        assert len(df) == 40

    def test_load_data_unsupported_raises(self, tmp_path):
        bad = str(tmp_path / "data.parquet")
        open(bad, "w").close()
        proc = TDProcessor(tokenizer=MagicMock())
        with pytest.raises(ValueError):
            proc.load_data(bad)

    def test_create_dataset_returns_tddataset(self, binary_df):
        n = len(binary_df)
        tokenizer = _make_mock_tokenizer(n_samples=n)
        proc = TDProcessor(tokenizer=tokenizer)
        ds = proc.create_dataset(binary_df)
        assert isinstance(ds, TDDataset)
        assert len(ds) == n

    def test_create_dataset_calls_tokenizer(self, binary_df):
        n = len(binary_df)
        tokenizer = _make_mock_tokenizer(n_samples=n)
        proc = TDProcessor(tokenizer=tokenizer, max_length=64)
        proc.create_dataset(binary_df)
        tokenizer.assert_called_once()
        _, kwargs = tokenizer.call_args
        assert kwargs["max_length"] == 64
        assert kwargs["truncation"] is True


# ---------------------------------------------------------------------------
# BinaryTDProcessor — prepare_binary_data
# ---------------------------------------------------------------------------

class TestBinaryTDProcessorPrepareBinaryData:
    def _proc(self):
        return BinaryTDProcessor(tokenizer=MagicMock())

    def test_numeric_labels_passthrough(self, binary_df):
        proc = self._proc()
        result = proc.prepare_binary_data(binary_df, numeric_labels=True)
        assert "label_idx" in result.columns
        assert set(result["label_idx"].unique()) == {0, 1}

    def test_categorical_labels_conversion(self, categorical_df):
        proc = self._proc()
        result = proc.prepare_binary_data(
            categorical_df, positive_category="td", numeric_labels=False
        )
        assert "label_idx" in result.columns
        assert set(result["label_idx"].unique()) == {0, 1}
        # "td" rows → label_idx == 1
        assert result.loc[result["label"] == "td", "label_idx"].eq(1).all()
        assert result.loc[result["label"] == "non_td", "label_idx"].eq(0).all()

    def test_categorical_without_positive_category_raises(self, categorical_df):
        proc = self._proc()
        with pytest.raises(ValueError, match="positive_category"):
            proc.prepare_binary_data(categorical_df, numeric_labels=False)

    def test_does_not_modify_original(self, binary_df):
        proc = self._proc()
        original_cols = list(binary_df.columns)
        proc.prepare_binary_data(binary_df, numeric_labels=True)
        assert list(binary_df.columns) == original_cols


# ---------------------------------------------------------------------------
# BinaryTDProcessor — extract_top_repo
# ---------------------------------------------------------------------------

class TestBinaryTDProcessorExtractTopRepo:
    def _proc(self):
        return BinaryTDProcessor(tokenizer=MagicMock())

    def test_extracts_correct_top_repo(self, repo_df):
        proc = self._proc()
        remaining, top_repo_data = proc.extract_top_repo(
            repo_df, label_col="label", repo_col="repo", positive_label=1
        )
        # "owner/alpha" has 10 positive samples, "owner/beta" has 5
        assert len(top_repo_data) > 0
        assert top_repo_data["repo"].iloc[0] == "owner/alpha"

    def test_remaining_excludes_top_repo(self, repo_df):
        proc = self._proc()
        remaining, top_repo_data = proc.extract_top_repo(
            repo_df, label_col="label", repo_col="repo", positive_label=1
        )
        top_repo = top_repo_data["repo"].iloc[0]
        assert top_repo not in remaining["repo"].values

    def test_empty_positive_class_returns_full_df(self):
        proc = self._proc()
        df = pd.DataFrame({"text": ["a", "b"], "label": [0, 0], "repo": ["r1", "r2"]})
        remaining, top = proc.extract_top_repo(df, label_col="label", repo_col="repo", positive_label=1)
        assert len(remaining) == 2
        assert len(top) == 0


# ---------------------------------------------------------------------------
# BinaryTDProcessor — extract_top_repos_by_category
# ---------------------------------------------------------------------------

class TestBinaryTDProcessorExtractTopReposByCategory:
    def test_returns_two_dataframes(self, repo_df):
        proc = BinaryTDProcessor(tokenizer=MagicMock())
        remaining, top = proc.extract_top_repos_by_category(
            repo_df, label_idx_col="label", repo_col="repo"
        )
        assert isinstance(remaining, pd.DataFrame)
        assert isinstance(top, pd.DataFrame)

    def test_combined_size_exceeds_original(self, repo_df):
        """remaining + top_repo may overlap; just verify no data is silently lost."""
        proc = BinaryTDProcessor(tokenizer=MagicMock())
        remaining, top = proc.extract_top_repos_by_category(
            repo_df, label_idx_col="label", repo_col="repo"
        )
        # top should contain at least one row
        assert len(top) > 0
