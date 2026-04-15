"""Tests for tdsuite/data/data_splitter.py — DataSplitter class and split_data()."""

import json
import os

import pandas as pd
import pytest

from tdsuite.data.data_splitter import DataSplitter, split_data


# ---------------------------------------------------------------------------
# DataSplitter — basic load / preprocess / balance / split
# ---------------------------------------------------------------------------

class TestDataSplitterLoadData:
    def test_load_csv(self, csv_file, tmp_path):
        splitter = DataSplitter(
            data_file=csv_file,
            output_dir=str(tmp_path / "out"),
            is_numeric_labels=True,
        )
        df = splitter.load_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 40

    def test_load_missing_file_raises(self, tmp_path):
        splitter = DataSplitter(
            data_file=str(tmp_path / "missing.csv"),
            output_dir=str(tmp_path / "out"),
        )
        with pytest.raises(FileNotFoundError):
            splitter.load_data()

    def test_load_unsupported_format_raises(self, tmp_path):
        bad = str(tmp_path / "data.parquet")
        open(bad, "w").close()
        splitter = DataSplitter(data_file=bad, output_dir=str(tmp_path / "out"))
        with pytest.raises(ValueError, match="Unsupported file format"):
            splitter.load_data()


class TestDataSplitterPreprocessData:
    def test_numeric_labels_unchanged(self, binary_df, tmp_path):
        splitter = DataSplitter(
            data_file="dummy.csv",
            output_dir=str(tmp_path),
            is_numeric_labels=True,
        )
        result = splitter.preprocess_data(binary_df.copy())
        assert set(result["label"].unique()) == {0, 1}

    def test_categorical_with_positive_category(self, categorical_df, tmp_path):
        splitter = DataSplitter(
            data_file="dummy.csv",
            output_dir=str(tmp_path),
            is_numeric_labels=False,
            positive_category="td",
        )
        result = splitter.preprocess_data(categorical_df.copy())
        assert set(result["label"].unique()) == {0, 1}

    def test_multi_class_categorical_encoding(self, tmp_path):
        df = pd.DataFrame({
            "text": ["a", "b", "c", "d", "e", "f"],
            "label": ["cat_A", "cat_B", "cat_C", "cat_A", "cat_B", "cat_C"],
        })
        splitter = DataSplitter(
            data_file="dummy.csv",
            output_dir=str(tmp_path),
            is_numeric_labels=False,
        )
        result = splitter.preprocess_data(df.copy())
        assert result["label"].dtype in (int, "int64", "int32")
        assert result["label"].nunique() == 3


class TestDataSplitterBalanceClasses:
    def test_balanced_output_equal_counts(self, tmp_path):
        df = pd.DataFrame({
            "text": [f"t{i}" for i in range(30)],
            "label": [1] * 20 + [0] * 10,
        })
        splitter = DataSplitter(
            data_file="dummy.csv",
            output_dir=str(tmp_path),
            is_numeric_labels=True,
        )
        balanced = splitter.balance_classes(df)
        counts = balanced["label"].value_counts()
        assert counts[0] == counts[1]

    def test_already_balanced_unchanged(self, binary_df, tmp_path):
        splitter = DataSplitter(
            data_file="dummy.csv",
            output_dir=str(tmp_path),
            is_numeric_labels=True,
        )
        balanced = splitter.balance_classes(binary_df)
        counts = balanced["label"].value_counts()
        assert counts[0] == counts[1]
        assert len(balanced) == len(binary_df)


class TestDataSplitterSplitAndSave:
    def test_creates_train_and_test_csv(self, csv_file, tmp_path):
        out = str(tmp_path / "split_out")
        splitter = DataSplitter(
            data_file=csv_file,
            output_dir=out,
            is_numeric_labels=True,
            test_size=0.2,
        )
        train_df, test_df, top_repo_df = splitter.split_and_save()
        assert os.path.exists(os.path.join(out, "train.csv"))
        assert os.path.exists(os.path.join(out, "test.csv"))

    def test_split_sizes_approx_correct(self, csv_file, tmp_path):
        out = str(tmp_path / "split_out")
        splitter = DataSplitter(
            data_file=csv_file,
            output_dir=out,
            is_numeric_labels=True,
            test_size=0.2,
        )
        train_df, test_df, _ = splitter.split_and_save()
        total = len(train_df) + len(test_df)
        assert abs(len(test_df) / total - 0.2) < 0.05

    def test_no_overlap_between_train_and_test(self, csv_file, tmp_path):
        out = str(tmp_path / "split_out")
        splitter = DataSplitter(
            data_file=csv_file,
            output_dir=out,
            is_numeric_labels=True,
        )
        train_df, test_df, _ = splitter.split_and_save()
        train_texts = set(train_df["text"])
        test_texts = set(test_df["text"])
        assert len(train_texts & test_texts) == 0

    def test_top_repo_extracted(self, tmp_path):
        """DataSplitter extracts the repo with most positive samples."""
        import pandas as pd

        rows = []
        for i in range(10):
            rows.append({"text": f"issue {i}", "label": 1, "repo": "owner/alpha"})
        for i in range(4):
            rows.append({"text": f"issue {10 + i}", "label": 1, "repo": "owner/beta"})
        for i in range(30):
            rows.append({"text": f"normal {i}", "label": 0, "repo": "owner/gamma"})

        df = pd.DataFrame(rows)
        csv_path = str(tmp_path / "repo_data.csv")
        df.to_csv(csv_path, index=False)

        out = str(tmp_path / "split_out")
        splitter = DataSplitter(
            data_file=csv_path,
            output_dir=out,
            is_numeric_labels=True,
            repo_column="repo",
        )
        _, _, top_repo_df = splitter.split_and_save()
        assert top_repo_df is not None
        assert os.path.exists(os.path.join(out, "top_repos.csv"))

    def test_label_mappings_saved_for_categorical(self, tmp_path):
        df = pd.DataFrame({
            "text": [f"t{i}" for i in range(20)],
            "label": ["td"] * 10 + ["non_td"] * 10,
        })
        csv_path = str(tmp_path / "cat.csv")
        df.to_csv(csv_path, index=False)
        out = str(tmp_path / "out")
        splitter = DataSplitter(
            data_file=csv_path,
            output_dir=out,
            is_numeric_labels=False,
            positive_category="td",
        )
        splitter.split_and_save()
        assert os.path.exists(os.path.join(out, "label_mappings.json"))


# ---------------------------------------------------------------------------
# standalone split_data()
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_creates_output_files(self, csv_file, tmp_path):
        out = str(tmp_path / "out")
        result = split_data(
            data_file=csv_file,
            output_dir=out,
            is_numeric_labels=True,
        )
        assert os.path.exists(os.path.join(out, "train.csv"))
        assert os.path.exists(os.path.join(out, "test.csv"))

    def test_returns_tuple(self, csv_file, tmp_path):
        out = str(tmp_path / "out")
        result = split_data(
            data_file=csv_file,
            output_dir=out,
            is_numeric_labels=True,
        )
        assert result is not None
        train_df, test_df, _ = result
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

    def test_categorical_auto_detection(self, tmp_path):
        """split_data auto-detects non_* prefix for binary categorical labels."""
        df = pd.DataFrame({
            "text": [f"t{i}" for i in range(20)],
            "label": ["defect"] * 10 + ["non_defect"] * 10,
        })
        csv_path = str(tmp_path / "cat.csv")
        df.to_csv(csv_path, index=False)
        out = str(tmp_path / "out")
        train_df, test_df, _ = split_data(
            data_file=csv_path,
            output_dir=out,
            is_numeric_labels=False,
        )
        # labels should be numeric after split_data
        assert set(train_df["label"].unique()).issubset({0, 1})

    def test_unsupported_format_raises(self, tmp_path):
        bad = str(tmp_path / "data.parquet")
        open(bad, "w").close()
        with pytest.raises(ValueError):
            split_data(data_file=bad, output_dir=str(tmp_path / "out"))
