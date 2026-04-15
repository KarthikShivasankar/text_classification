"""Tests for tdsuite/utils/inference.py — InferenceEngine and EnsembleInferenceEngine.

All model loading is mocked so no network access or GPU is required.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

def _make_output(batch_size: int, num_classes: int = 2):
    """Minimal model output with a real tensor for .logits."""
    output = MagicMock()
    logits = torch.zeros(batch_size, num_classes)
    logits[:, 1] = 1.0  # always predicts class 1
    output.logits = logits
    return output


def _dynamic_model(num_classes: int = 2):
    """
    Model mock whose __call__ inspects the actual input_ids batch size
    and returns a properly-shaped output each time.
    """
    model = MagicMock()
    model.eval.return_value = None
    model.to.return_value = model

    def _forward(**kwargs):
        bs = kwargs.get("input_ids", torch.zeros(1, 16)).shape[0]
        return _make_output(bs, num_classes)

    # MagicMock.side_effect is invoked whenever the mock object is called.
    model.side_effect = _forward
    return model


def _make_tokenizer():
    """Mock tokenizer that returns real tensors sized to the input batch."""
    tokenizer = MagicMock()

    def _tok(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        return {
            "input_ids": torch.zeros(n, 16, dtype=torch.long),
            "attention_mask": torch.ones(n, 16, dtype=torch.long),
        }

    tokenizer.side_effect = _tok
    return tokenizer


# ---------------------------------------------------------------------------
# InferenceEngine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ie_engine():
    """InferenceEngine with fully mocked model + tokenizer."""
    with patch("tdsuite.utils.inference.TransformerModel") as MockTM, \
         patch("tdsuite.utils.inference.AutoTokenizer") as MockTok:

        MockTM.from_pretrained.return_value = _dynamic_model()
        MockTok.from_pretrained.return_value = _make_tokenizer()

        from tdsuite.utils.inference import InferenceEngine
        engine = InferenceEngine(model_name="dummy/model", device="cpu")
        engine.show_progress = False
        yield engine


# ---------------------------------------------------------------------------
# InferenceEngine — predict_single
# ---------------------------------------------------------------------------

class TestInferenceEnginePredictSingle:
    def test_returns_dict(self, ie_engine):
        result = ie_engine.predict_single("Some text about technical debt")
        assert isinstance(result, dict)

    def test_required_keys(self, ie_engine):
        result = ie_engine.predict_single("test text")
        for key in ("text", "predicted_class", "predicted_probability", "class_probabilities"):
            assert key in result

    def test_predicted_class_is_int(self, ie_engine):
        result = ie_engine.predict_single("test text")
        assert isinstance(result["predicted_class"], int)

    def test_probability_in_range(self, ie_engine):
        result = ie_engine.predict_single("test text")
        assert 0.0 <= result["predicted_probability"] <= 1.0

    def test_class_probabilities_sum_to_one(self, ie_engine):
        result = ie_engine.predict_single("test text")
        assert sum(result["class_probabilities"]) == pytest.approx(1.0, abs=1e-5)

    def test_text_preserved(self, ie_engine):
        text = "My specific input text"
        result = ie_engine.predict_single(text)
        assert result["text"] == text


# ---------------------------------------------------------------------------
# InferenceEngine — predict_batch
# ---------------------------------------------------------------------------

class TestInferenceEnginePredictBatch:
    def test_returns_list_of_dicts(self, ie_engine):
        results = ie_engine.predict_batch(["text one", "text two", "text three"], batch_size=2)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_each_item_has_required_keys(self, ie_engine):
        for item in ie_engine.predict_batch(["a", "b"]):
            for key in ("text", "predicted_class", "predicted_probability", "class_probabilities"):
                assert key in item

    def test_single_text_batch(self, ie_engine):
        results = ie_engine.predict_batch(["only one"])
        assert len(results) == 1

    def test_batch_larger_than_batch_size(self, ie_engine):
        texts = [f"text {i}" for i in range(10)]
        results = ie_engine.predict_batch(texts, batch_size=3)
        assert len(results) == 10


# ---------------------------------------------------------------------------
# InferenceEngine — predict_from_file
# ---------------------------------------------------------------------------

class TestInferenceEnginePredictFromFile:
    def test_returns_dataframe(self, ie_engine, csv_file):
        result = ie_engine.predict_from_file(csv_file)
        assert isinstance(result, pd.DataFrame)

    def test_result_has_prediction_columns(self, ie_engine, csv_file):
        result = ie_engine.predict_from_file(csv_file)
        assert "predicted_class" in result.columns
        assert "predicted_probability" in result.columns

    def test_result_length_matches_input(self, ie_engine, csv_file):
        original = pd.read_csv(csv_file)
        result = ie_engine.predict_from_file(csv_file)
        assert len(result) == len(original)

    def test_saves_to_file(self, ie_engine, csv_file, tmp_path):
        import os
        out = str(tmp_path / "preds" / "out.csv")
        ie_engine.predict_from_file(csv_file, output_file=out)
        assert os.path.exists(out)

    def test_unsupported_format_raises(self, ie_engine, tmp_path):
        bad = str(tmp_path / "data.txt")
        open(bad, "w").close()
        with pytest.raises(ValueError, match="CSV or JSON"):
            ie_engine.predict_from_file(bad)


# ---------------------------------------------------------------------------
# EnsembleInferenceEngine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ee_engine():
    """EnsembleInferenceEngine with two mocked models."""
    with patch("tdsuite.utils.inference.AutoModelForSequenceClassification") as MockModel, \
         patch("tdsuite.utils.inference.AutoTokenizer") as MockTok:

        MockModel.from_pretrained.return_value = _dynamic_model()
        MockTok.from_pretrained.return_value = _make_tokenizer()

        from tdsuite.utils.inference import EnsembleInferenceEngine
        engine = EnsembleInferenceEngine(model_names=["m1", "m2"], device="cpu")
        engine.show_progress = False
        yield engine


# ---------------------------------------------------------------------------
# EnsembleInferenceEngine — init validation
# ---------------------------------------------------------------------------

class TestEnsembleInferenceEngineInit:
    def test_loads_two_models(self, ee_engine):
        assert len(ee_engine.models) == 2

    def test_equal_weights_default(self, ee_engine):
        assert ee_engine.weights == pytest.approx([0.5, 0.5])

    def test_custom_weights_normalised(self):
        with patch("tdsuite.utils.inference.AutoModelForSequenceClassification") as MockModel, \
             patch("tdsuite.utils.inference.AutoTokenizer") as MockTok:

            MockModel.from_pretrained.return_value = _dynamic_model()
            MockTok.from_pretrained.return_value = _make_tokenizer()

            from tdsuite.utils.inference import EnsembleInferenceEngine
            engine = EnsembleInferenceEngine(
                model_names=["m1", "m2"], weights=[2.0, 2.0], device="cpu"
            )
        assert engine.weights == pytest.approx([0.5, 0.5])

    def test_wrong_weight_count_raises(self):
        with patch("tdsuite.utils.inference.AutoModelForSequenceClassification") as MockModel, \
             patch("tdsuite.utils.inference.AutoTokenizer") as MockTok:

            MockModel.from_pretrained.return_value = _dynamic_model()
            MockTok.from_pretrained.return_value = _make_tokenizer()

            from tdsuite.utils.inference import EnsembleInferenceEngine
            with pytest.raises(ValueError, match="weights"):
                EnsembleInferenceEngine(
                    model_names=["m1", "m2"], weights=[0.5], device="cpu"
                )

    def test_no_models_raises(self):
        from tdsuite.utils.inference import EnsembleInferenceEngine
        with pytest.raises(ValueError, match="At least one"):
            EnsembleInferenceEngine(device="cpu")


# ---------------------------------------------------------------------------
# EnsembleInferenceEngine — predict_single
# ---------------------------------------------------------------------------

class TestEnsembleInferenceEnginePredictSingle:
    def test_returns_dict_with_required_keys(self, ee_engine):
        result = ee_engine.predict_single("some text")
        for key in ("text", "predicted_class", "predicted_probability", "class_probabilities"):
            assert key in result

    def test_probability_in_range(self, ee_engine):
        result = ee_engine.predict_single("some text")
        assert 0.0 <= result["predicted_probability"] <= 1.0

    def test_class_probs_sum_to_one(self, ee_engine):
        result = ee_engine.predict_single("some text")
        assert sum(result["class_probabilities"]) == pytest.approx(1.0, abs=1e-5)

    def test_empty_text_raises(self, ee_engine):
        with pytest.raises(ValueError, match="empty"):
            ee_engine.predict_single("")


# ---------------------------------------------------------------------------
# EnsembleInferenceEngine — predict_batch
# ---------------------------------------------------------------------------

class TestEnsembleInferenceEnginePredictBatch:
    def test_returns_list(self, ee_engine):
        results = ee_engine.predict_batch(["a", "b", "c"])
        assert isinstance(results, list)
        assert len(results) == 3

    def test_each_item_has_keys(self, ee_engine):
        for item in ee_engine.predict_batch(["text one", "text two"]):
            for key in ("text", "predicted_class", "predicted_probability", "class_probabilities"):
                assert key in item

    def test_probabilities_sum_to_one(self, ee_engine):
        for item in ee_engine.predict_batch(["x", "y", "z"]):
            assert sum(item["class_probabilities"]) == pytest.approx(1.0, abs=1e-5)

    def test_empty_list_raises(self, ee_engine):
        with pytest.raises(ValueError, match="empty"):
            ee_engine.predict_batch([])
