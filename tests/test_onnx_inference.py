"""Tests for tdsuite/utils/onnx_inference.py — OnnxInferenceEngine.

All model I/O is mocked so no network access or GPU is required.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers to build a minimal in-memory ONNX session mock
# ---------------------------------------------------------------------------

def _make_ort_session(num_classes: int = 2, batch_size: int = 1):
    """Return a mock onnxruntime.InferenceSession."""
    session = MagicMock()
    # Inputs the session expects
    inp_ids = MagicMock()
    inp_ids.name = "input_ids"
    attn = MagicMock()
    attn.name = "attention_mask"
    session.get_inputs.return_value = [inp_ids, attn]

    # Providers
    session.get_providers.return_value = ["CPUExecutionProvider"]

    # run() returns [logits]
    def _run(output_names, inputs):
        bs = inputs["input_ids"].shape[0]
        logits = np.zeros((bs, num_classes), dtype=np.float32)
        logits[:, 1] = 2.0  # always predicts class 1
        return [logits]

    session.run.side_effect = _run
    return session


def _make_tokenizer(seq_len: int = 16):
    """Return a mock tokenizer."""
    tok = MagicMock()

    def _tok(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        return {
            "input_ids": np.zeros((n, seq_len), dtype=np.int64),
            "attention_mask": np.ones((n, seq_len), dtype=np.int64),
        }

    tok.side_effect = _tok
    return tok


# ---------------------------------------------------------------------------
# Fixtures — patch onnxruntime and transformers
# ---------------------------------------------------------------------------

@pytest.fixture
def onnx_engine(tmp_path):
    """OnnxInferenceEngine backed by mocked ort session and tokenizer."""
    fake_onnx = str(tmp_path / "model.onnx")
    Path(fake_onnx).write_bytes(b"fake")  # file must exist for __init__ check

    with patch("tdsuite.utils.onnx_inference._require_onnxruntime") as mock_ort_fn, \
         patch("tdsuite.utils.onnx_inference.OnnxInferenceEngine._load_tokenizer") as mock_tok_fn:

        mock_ort = MagicMock()
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_ort.InferenceSession.return_value = _make_ort_session()
        mock_ort_fn.return_value = mock_ort
        mock_tok_fn.return_value = _make_tokenizer()

        from tdsuite.utils.onnx_inference import OnnxInferenceEngine
        engine = OnnxInferenceEngine(onnx_path=fake_onnx, show_progress=False)
        yield engine


# ---------------------------------------------------------------------------
# predict_single
# ---------------------------------------------------------------------------

class TestOnnxPredictSingle:
    def test_returns_dict(self, onnx_engine):
        result = onnx_engine.predict_single("some text")
        assert isinstance(result, dict)

    def test_required_keys(self, onnx_engine):
        result = onnx_engine.predict_single("test text")
        for key in ("text", "predicted_class", "predicted_probability", "class_probabilities"):
            assert key in result

    def test_predicted_class_is_int(self, onnx_engine):
        result = onnx_engine.predict_single("test text")
        assert isinstance(result["predicted_class"], int)

    def test_probability_in_range(self, onnx_engine):
        result = onnx_engine.predict_single("test text")
        assert 0.0 <= result["predicted_probability"] <= 1.0

    def test_class_probabilities_sum_to_one(self, onnx_engine):
        result = onnx_engine.predict_single("test text")
        assert sum(result["class_probabilities"]) == pytest.approx(1.0, abs=1e-5)

    def test_text_preserved(self, onnx_engine):
        text = "specific input text"
        result = onnx_engine.predict_single(text)
        assert result["text"] == text

    def test_always_predicts_class_1(self, onnx_engine):
        """Mock logits give class 1 always."""
        result = onnx_engine.predict_single("anything")
        assert result["predicted_class"] == 1


# ---------------------------------------------------------------------------
# predict_batch
# ---------------------------------------------------------------------------

class TestOnnxPredictBatch:
    def test_returns_list_of_dicts(self, onnx_engine):
        results = onnx_engine.predict_batch(["a", "b", "c"], batch_size=2)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_each_item_has_required_keys(self, onnx_engine):
        for item in onnx_engine.predict_batch(["a", "b"]):
            for key in ("text", "predicted_class", "predicted_probability", "class_probabilities"):
                assert key in item

    def test_single_text_batch(self, onnx_engine):
        results = onnx_engine.predict_batch(["only one"])
        assert len(results) == 1

    def test_batch_larger_than_batch_size(self, onnx_engine):
        texts = [f"text {i}" for i in range(10)]
        results = onnx_engine.predict_batch(texts, batch_size=3)
        assert len(results) == 10

    def test_probabilities_sum_to_one(self, onnx_engine):
        for item in onnx_engine.predict_batch(["x", "y"]):
            assert sum(item["class_probabilities"]) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# predict_from_file
# ---------------------------------------------------------------------------

class TestOnnxPredictFromFile:
    def test_returns_dataframe(self, onnx_engine, csv_file):
        result = onnx_engine.predict_from_file(csv_file)
        assert isinstance(result, pd.DataFrame)

    def test_result_has_prediction_columns(self, onnx_engine, csv_file):
        result = onnx_engine.predict_from_file(csv_file)
        assert "predicted_class" in result.columns
        assert "predicted_probability" in result.columns
        assert "class_probabilities" in result.columns

    def test_result_length_matches_input(self, onnx_engine, csv_file):
        original = pd.read_csv(csv_file)
        result = onnx_engine.predict_from_file(csv_file)
        assert len(result) == len(original)

    def test_saves_to_file(self, onnx_engine, csv_file, tmp_path):
        out = str(tmp_path / "out" / "predictions.csv")
        onnx_engine.predict_from_file(csv_file, output_file=out)
        assert os.path.exists(out)

    def test_unsupported_format_raises(self, onnx_engine, tmp_path):
        bad = str(tmp_path / "data.txt")
        Path(bad).write_text("bad")
        with pytest.raises(ValueError):
            onnx_engine.predict_from_file(bad)

    def test_missing_text_column_raises(self, onnx_engine, tmp_path):
        csv = str(tmp_path / "no_text.csv")
        pd.DataFrame({"other": ["a", "b"]}).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="text"):
            onnx_engine.predict_from_file(csv, text_column="text")

    def test_json_file(self, onnx_engine, json_file):
        result = onnx_engine.predict_from_file(json_file)
        assert "predicted_class" in result.columns

    def test_jsonl_file(self, onnx_engine, jsonl_file):
        result = onnx_engine.predict_from_file(jsonl_file)
        assert "predicted_class" in result.columns


# ---------------------------------------------------------------------------
# from_pretrained — fallback to _export_to_onnx
# ---------------------------------------------------------------------------

class TestFromPretrainedFallback:
    def test_fallback_triggered_on_hub_404(self, tmp_path):
        """When HF Hub raises, _export_to_onnx is called instead."""
        fake_onnx = str(tmp_path / "model.onnx")
        Path(fake_onnx).write_bytes(b"fake")

        with patch("huggingface_hub.hf_hub_download", side_effect=Exception("EntryNotFound")), \
             patch("tdsuite.utils.onnx_inference.OnnxInferenceEngine._export_to_onnx",
                   return_value=fake_onnx) as mock_export, \
             patch("tdsuite.utils.onnx_inference.OnnxInferenceEngine.__init__",
                   return_value=None):
            from tdsuite.utils.onnx_inference import OnnxInferenceEngine
            OnnxInferenceEngine.from_pretrained("some/model")
            mock_export.assert_called_once_with(
                model_id="some/model", max_length=512, token=None
            )

    def test_no_fallback_when_hub_succeeds(self, tmp_path):
        """When HF Hub returns a path, _export_to_onnx is NOT called."""
        fake_onnx = str(tmp_path / "model.onnx")
        Path(fake_onnx).write_bytes(b"fake")

        with patch("huggingface_hub.hf_hub_download", return_value=fake_onnx), \
             patch("tdsuite.utils.onnx_inference.OnnxInferenceEngine._export_to_onnx") as mock_export, \
             patch("tdsuite.utils.onnx_inference.OnnxInferenceEngine.__init__",
                   return_value=None):
            from tdsuite.utils.onnx_inference import OnnxInferenceEngine
            OnnxInferenceEngine.from_pretrained("some/model")
            mock_export.assert_not_called()

    def test_token_forwarded_to_export(self, tmp_path):
        """The token arg is passed through to _export_to_onnx."""
        fake_onnx = str(tmp_path / "model.onnx")
        Path(fake_onnx).write_bytes(b"fake")

        with patch("huggingface_hub.hf_hub_download", side_effect=Exception("404")), \
             patch("tdsuite.utils.onnx_inference.OnnxInferenceEngine._export_to_onnx",
                   return_value=fake_onnx) as mock_export, \
             patch("tdsuite.utils.onnx_inference.OnnxInferenceEngine.__init__",
                   return_value=None):
            from tdsuite.utils.onnx_inference import OnnxInferenceEngine
            OnnxInferenceEngine.from_pretrained("some/model", token="hf_abc")
            _, kwargs = mock_export.call_args
            assert kwargs.get("token") == "hf_abc"


# ===========================================================================
# OnnxEnsembleInferenceEngine
# ===========================================================================

def _make_member_engine(class1_prob: float = 0.8):
    """Return a mock ONNX member engine exposing _tokenize / _run_session."""
    member = MagicMock()

    def _tok(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": np.zeros((len(texts), 16), dtype=np.int64)}

    def _run(inputs):
        bs = inputs["input_ids"].shape[0]
        probs = np.zeros((bs, 2), dtype=np.float32)
        probs[:, 1] = class1_prob
        probs[:, 0] = 1.0 - class1_prob
        return probs

    member._tokenize.side_effect = _tok
    member._run_session.side_effect = _run
    return member


@pytest.fixture
def onnx_ensemble_engine():
    """OnnxEnsembleInferenceEngine backed by two mocked member engines."""
    members = [_make_member_engine(0.8), _make_member_engine(0.6)]

    with patch("tdsuite.utils.onnx_inference._build_onnx_member", side_effect=members), \
         patch("tdsuite.utils.onnx_inference.auto_select_device", return_value="cpu"):
        from tdsuite.utils.onnx_inference import OnnxEnsembleInferenceEngine
        engine = OnnxEnsembleInferenceEngine(
            model_names=["m1", "m2"], show_progress=False
        )
        yield engine


class TestOnnxEnsembleInit:
    def test_builds_two_members(self, onnx_ensemble_engine):
        assert len(onnx_ensemble_engine.engines) == 2

    def test_equal_weights_default(self, onnx_ensemble_engine):
        assert onnx_ensemble_engine.weights == pytest.approx([0.5, 0.5])

    def test_custom_weights_normalised(self):
        members = [_make_member_engine(0.8), _make_member_engine(0.6)]
        with patch("tdsuite.utils.onnx_inference._build_onnx_member", side_effect=members), \
             patch("tdsuite.utils.onnx_inference.auto_select_device", return_value="cpu"):
            from tdsuite.utils.onnx_inference import OnnxEnsembleInferenceEngine
            engine = OnnxEnsembleInferenceEngine(
                model_names=["m1", "m2"], weights=[3.0, 1.0], show_progress=False
            )
        assert engine.weights == pytest.approx([0.75, 0.25])

    def test_wrong_weight_count_raises(self):
        members = [_make_member_engine(0.8), _make_member_engine(0.6)]
        with patch("tdsuite.utils.onnx_inference._build_onnx_member", side_effect=members), \
             patch("tdsuite.utils.onnx_inference.auto_select_device", return_value="cpu"):
            from tdsuite.utils.onnx_inference import OnnxEnsembleInferenceEngine
            with pytest.raises(ValueError, match="weights"):
                OnnxEnsembleInferenceEngine(
                    model_names=["m1", "m2"], weights=[0.5], show_progress=False
                )

    def test_no_models_raises(self):
        with patch("tdsuite.utils.onnx_inference.auto_select_device", return_value="cpu"):
            from tdsuite.utils.onnx_inference import OnnxEnsembleInferenceEngine
            with pytest.raises(ValueError, match="At least one"):
                OnnxEnsembleInferenceEngine(show_progress=False)


class TestOnnxEnsemblePredictSingle:
    def test_required_keys(self, onnx_ensemble_engine):
        result = onnx_ensemble_engine.predict_single("test text")
        for key in ("text", "predicted_class", "predicted_probability", "class_probabilities"):
            assert key in result

    def test_predicted_class_is_int(self, onnx_ensemble_engine):
        result = onnx_ensemble_engine.predict_single("test text")
        assert isinstance(result["predicted_class"], int)

    def test_class_probabilities_sum_to_one(self, onnx_ensemble_engine):
        result = onnx_ensemble_engine.predict_single("test text")
        assert sum(result["class_probabilities"]) == pytest.approx(1.0, abs=1e-5)

    def test_weighted_average_is_correct(self, onnx_ensemble_engine):
        """Equal weights of class-1 probs 0.8 and 0.6 => 0.7."""
        result = onnx_ensemble_engine.predict_single("anything")
        assert result["predicted_class"] == 1
        assert result["class_probabilities"][1] == pytest.approx(0.7, abs=1e-5)

    def test_text_preserved(self, onnx_ensemble_engine):
        text = "specific input text"
        assert onnx_ensemble_engine.predict_single(text)["text"] == text


class TestOnnxEnsemblePredictBatch:
    def test_returns_list_of_dicts(self, onnx_ensemble_engine):
        results = onnx_ensemble_engine.predict_batch(["a", "b", "c"], batch_size=2)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_batch_larger_than_batch_size(self, onnx_ensemble_engine):
        texts = [f"text {i}" for i in range(10)]
        results = onnx_ensemble_engine.predict_batch(texts, batch_size=3)
        assert len(results) == 10

    def test_probabilities_sum_to_one(self, onnx_ensemble_engine):
        for item in onnx_ensemble_engine.predict_batch(["x", "y"]):
            assert sum(item["class_probabilities"]) == pytest.approx(1.0, abs=1e-5)


class TestOnnxEnsemblePredictFromFile:
    def test_result_has_prediction_columns(self, onnx_ensemble_engine, csv_file):
        result = onnx_ensemble_engine.predict_from_file(csv_file)
        for col in ("predicted_class", "predicted_probability", "class_probabilities"):
            assert col in result.columns

    def test_result_length_matches_input(self, onnx_ensemble_engine, csv_file):
        original = pd.read_csv(csv_file)
        result = onnx_ensemble_engine.predict_from_file(csv_file)
        assert len(result) == len(original)

    def test_saves_to_file(self, onnx_ensemble_engine, csv_file, tmp_path):
        out = str(tmp_path / "out" / "ensemble_predictions.csv")
        onnx_ensemble_engine.predict_from_file(csv_file, output_file=out)
        assert os.path.exists(out)

    def test_unsupported_format_raises(self, onnx_ensemble_engine, tmp_path):
        bad = str(tmp_path / "data.txt")
        Path(bad).write_text("bad")
        with pytest.raises(ValueError):
            onnx_ensemble_engine.predict_from_file(bad)

    def test_json_file(self, onnx_ensemble_engine, json_file):
        result = onnx_ensemble_engine.predict_from_file(json_file)
        assert "predicted_class" in result.columns
