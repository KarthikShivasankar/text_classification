"""Tests for tdsuite/utils/metrics.py — compute_metrics()."""

import json
import os

import pandas as pd
import pytest

from tdsuite.utils.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect_df(n=20):
    """DataFrame where predictions perfectly match labels."""
    labels = [1] * (n // 2) + [0] * (n // 2)
    return pd.DataFrame({
        "label": labels,
        "predicted_class": labels,
        "predicted_probability": [0.95] * (n // 2) + [0.05] * (n // 2),
        "class_probabilities": [[0.05, 0.95]] * (n // 2) + [[0.95, 0.05]] * (n // 2),
    })


def _random_df(n=20):
    """DataFrame with realistic but imperfect predictions."""
    import random
    random.seed(42)
    labels = [1, 0] * (n // 2)
    preds = [1 if random.random() > 0.3 else 0 for _ in range(n)]
    probs = [[1 - p * 0.8, p * 0.8] for p in [0.9, 0.2, 0.8, 0.1] * (n // 4)]
    return pd.DataFrame({
        "label": labels,
        "predicted_class": preds,
        "predicted_probability": [p[1] for p in probs],
        "class_probabilities": probs,
    })


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

class TestComputeMetricsReturnStructure:
    def test_returns_dict(self, predictions_df):
        result = compute_metrics(predictions_df, save_plots=False)
        assert isinstance(result, dict)

    def test_required_keys_present(self, predictions_df):
        result = compute_metrics(predictions_df, save_plots=False)
        for key in ("accuracy", "precision", "recall", "f1", "mcc", "confusion_matrix"):
            assert key in result, f"Missing key: {key}"

    def test_roc_auc_included_when_probs_available(self, predictions_df):
        result = compute_metrics(predictions_df, save_plots=False)
        assert "roc_auc" in result

    def test_values_in_range(self, predictions_df):
        result = compute_metrics(predictions_df, save_plots=False)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range"

    def test_confusion_matrix_shape(self, predictions_df):
        result = compute_metrics(predictions_df, save_plots=False)
        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2


# ---------------------------------------------------------------------------
# Perfect predictions
# ---------------------------------------------------------------------------

class TestComputeMetricsPerfect:
    def test_accuracy_is_1(self):
        result = compute_metrics(_perfect_df(), save_plots=False)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_f1_is_1(self):
        result = compute_metrics(_perfect_df(), save_plots=False)
        assert result["f1"] == pytest.approx(1.0)

    def test_mcc_is_1(self):
        result = compute_metrics(_perfect_df(), save_plots=False)
        assert result["mcc"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Saving to disk
# ---------------------------------------------------------------------------

class TestComputeMetricsSaveToDisk:
    def test_metrics_json_written(self, predictions_df, tmp_path):
        compute_metrics(predictions_df, output_dir=str(tmp_path), save_plots=False)
        metrics_file = tmp_path / "metrics.json"
        assert metrics_file.exists()

    def test_metrics_json_readable(self, predictions_df, tmp_path):
        compute_metrics(predictions_df, output_dir=str(tmp_path), save_plots=False)
        with open(tmp_path / "metrics.json") as f:
            data = json.load(f)
        assert "accuracy" in data

    def test_plots_written_when_save_plots_true(self, predictions_df, tmp_path):
        compute_metrics(predictions_df, output_dir=str(tmp_path), save_plots=True)
        assert (tmp_path / "metrics_summary.png").exists()
        assert (tmp_path / "confusion_matrix.png").exists()

    def test_no_plots_when_save_plots_false(self, predictions_df, tmp_path):
        compute_metrics(predictions_df, output_dir=str(tmp_path), save_plots=False)
        assert not (tmp_path / "metrics_summary.png").exists()


# ---------------------------------------------------------------------------
# Without output_dir (no disk writes)
# ---------------------------------------------------------------------------

class TestComputeMetricsNoDisk:
    def test_no_files_written_when_no_output_dir(self, predictions_df, tmp_path):
        # Run in tmp_path so we can detect unexpected files
        orig_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            compute_metrics(predictions_df, save_plots=False)
        finally:
            os.chdir(orig_dir)
        # No json should be written in the working directory
        assert not (tmp_path / "metrics.json").exists()
