"""Tests for tdsuite/cli.py — all six argument parsers."""

import pytest

from tdsuite.cli import (
    get_export_onnx_parser,
    get_extract_issues_parser,
    get_fetch_issues_parser,
    get_inference_parser,
    get_split_data_parser,
    get_train_parser,
)


# ---------------------------------------------------------------------------
# get_train_parser
# ---------------------------------------------------------------------------

class TestTrainParser:
    def _parse(self, args):
        return get_train_parser().parse_args(args)

    def test_required_args(self):
        ns = self._parse([
            "--data_file", "data.csv",
            "--model_name", "bert-base-uncased",
            "--output_dir", "./out",
        ])
        assert ns.data_file == "data.csv"
        assert ns.model_name == "bert-base-uncased"
        assert ns.output_dir == "./out"

    def test_defaults(self):
        ns = self._parse([
            "--data_file", "d.csv",
            "--model_name", "m",
            "--output_dir", "o",
        ])
        assert ns.num_epochs == 3
        assert ns.batch_size == 16
        assert ns.learning_rate == pytest.approx(2e-5)
        assert ns.seed == 42
        assert ns.cross_validation is False
        assert ns.n_splits == 5
        assert ns.numeric_labels is False

    def test_cross_validation_flag(self):
        ns = self._parse([
            "--data_file", "d.csv",
            "--model_name", "m",
            "--output_dir", "o",
            "--cross_validation",
            "--n_splits", "10",
        ])
        assert ns.cross_validation is True
        assert ns.n_splits == 10

    def test_numeric_labels_flag(self):
        ns = self._parse([
            "--data_file", "d.csv",
            "--model_name", "m",
            "--output_dir", "o",
            "--numeric_labels",
        ])
        assert ns.numeric_labels is True

    def test_missing_required_raises(self):
        with pytest.raises(SystemExit):
            get_train_parser().parse_args(["--data_file", "d.csv"])


# ---------------------------------------------------------------------------
# get_inference_parser
# ---------------------------------------------------------------------------

class TestInferenceParser:
    def _parse(self, args):
        return get_inference_parser().parse_args(args)

    def test_model_name_with_text(self):
        ns = self._parse(["--model_name", "karths/model", "--text", "hello world"])
        assert ns.model_name == "karths/model"
        assert ns.text == "hello world"

    def test_model_path_with_input_file(self):
        ns = self._parse(["--model_path", "./model", "--input_file", "data.csv"])
        assert ns.model_path == "./model"
        assert ns.input_file == "data.csv"

    def test_ensemble_model_paths(self):
        ns = self._parse([
            "--model_paths", "model1", "model2",
            "--input_file", "data.csv",
            "--weights", "0.6", "0.4",
        ])
        assert ns.model_paths == ["model1", "model2"]
        assert ns.weights == pytest.approx([0.6, 0.4])

    def test_defaults(self):
        ns = self._parse(["--model_name", "m", "--text", "t"])
        assert ns.batch_size == 32
        assert ns.max_length == 512
        assert ns.text_column == "text"
        assert ns.disable_progress_bar is False

    def test_onnx_path_accepted(self):
        ns = self._parse(["--model_name", "m", "--text", "t", "--onnx_path", "model.onnx"])
        assert ns.onnx_path == "model.onnx"

    def test_model_group_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            self._parse(["--model_name", "m", "--model_path", "p", "--text", "t"])

    def test_input_group_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            self._parse(["--model_name", "m", "--text", "t", "--input_file", "f.csv"])

    def test_missing_model_raises(self):
        with pytest.raises(SystemExit):
            self._parse(["--text", "hello"])

    def test_missing_input_raises(self):
        with pytest.raises(SystemExit):
            self._parse(["--model_name", "m"])


# ---------------------------------------------------------------------------
# get_export_onnx_parser
# ---------------------------------------------------------------------------

class TestExportOnnxParser:
    def _parse(self, args):
        return get_export_onnx_parser().parse_args(args)

    def test_model_name(self):
        ns = self._parse(["--model_name", "karths/model", "--output", "model.onnx"])
        assert ns.model_name == "karths/model"
        assert ns.output == "model.onnx"

    def test_model_path(self):
        ns = self._parse(["--model_path", "./local_model", "--output", "out.onnx"])
        assert ns.model_path == "./local_model"

    def test_defaults(self):
        ns = self._parse(["--model_name", "m", "--output", "m.onnx"])
        assert ns.max_length == 512
        assert ns.opset == 14

    def test_custom_opset_and_length(self):
        ns = self._parse([
            "--model_name", "m",
            "--output", "m.onnx",
            "--max_length", "256",
            "--opset", "17",
        ])
        assert ns.max_length == 256
        assert ns.opset == 17

    def test_source_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            self._parse(["--model_name", "m", "--model_path", "p", "--output", "o.onnx"])

    def test_missing_output_raises(self):
        with pytest.raises(SystemExit):
            self._parse(["--model_name", "m"])


# ---------------------------------------------------------------------------
# get_split_data_parser
# ---------------------------------------------------------------------------

class TestSplitDataParser:
    def _parse(self, args):
        return get_split_data_parser().parse_args(args)

    def test_required_args(self):
        ns = self._parse(["--data_file", "d.csv", "--output_dir", "out"])
        assert ns.data_file == "d.csv"
        assert ns.output_dir == "out"

    def test_defaults(self):
        ns = self._parse(["--data_file", "d.csv", "--output_dir", "out"])
        assert ns.test_size == pytest.approx(0.2)
        assert ns.random_state == 42
        assert ns.repo_column is None
        assert ns.is_huggingface_dataset is False
        assert ns.is_numeric_labels is False

    def test_all_flags(self):
        ns = self._parse([
            "--data_file", "d.csv",
            "--output_dir", "out",
            "--test_size", "0.3",
            "--random_state", "123",
            "--repo_column", "repo",
            "--is_numeric_labels",
        ])
        assert ns.test_size == pytest.approx(0.3)
        assert ns.random_state == 123
        assert ns.repo_column == "repo"
        assert ns.is_numeric_labels is True


# ---------------------------------------------------------------------------
# get_fetch_issues_parser
# ---------------------------------------------------------------------------

class TestFetchIssuesParser:
    def _parse(self, args):
        return get_fetch_issues_parser().parse_args(args)

    def test_required_repo(self):
        ns = self._parse(["--repo", "owner/repo"])
        assert ns.repo == "owner/repo"

    def test_defaults(self):
        ns = self._parse(["--repo", "owner/repo"])
        assert ns.output == "issues.csv"
        assert ns.state == "all"
        assert ns.token is None
        assert ns.limit == 100
        assert ns.fetch_all is False

    def test_fetch_all_flag(self):
        ns = self._parse(["--repo", "owner/repo", "--all"])
        assert ns.fetch_all is True

    def test_limit_and_all_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            self._parse(["--repo", "owner/repo", "--all", "--limit", "50"])

    def test_state_choices(self):
        for state in ("open", "closed", "all"):
            ns = self._parse(["--repo", "owner/repo", "--state", state])
            assert ns.state == state

    def test_invalid_state_raises(self):
        with pytest.raises(SystemExit):
            self._parse(["--repo", "owner/repo", "--state", "invalid"])


# ---------------------------------------------------------------------------
# get_extract_issues_parser
# ---------------------------------------------------------------------------

class TestExtractIssuesParser:
    def _parse(self, args):
        return get_extract_issues_parser().parse_args(args)

    def test_required_input(self):
        ns = self._parse(["--input", "issues.csv"])
        assert ns.input == "issues.csv"

    def test_defaults(self):
        ns = self._parse(["--input", "issues.csv"])
        assert ns.output == "issue_texts.csv"
        assert ns.body_column == "body"
        assert ns.min_length == 20
        assert ns.keep_metadata is False
        assert ns.drop_duplicates is False

    def test_all_options(self):
        ns = self._parse([
            "--input", "in.csv",
            "--output", "out.csv",
            "--body-column", "description",
            "--min-length", "50",
            "--keep-metadata",
            "--drop-duplicates",
        ])
        assert ns.output == "out.csv"
        assert ns.body_column == "description"
        assert ns.min_length == 50
        assert ns.keep_metadata is True
        assert ns.drop_duplicates is True
