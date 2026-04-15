"""Tests for tdsuite/config/config.py — all four config dataclasses + master Config."""

import json
import os

import pytest

from tdsuite.config.config import (
    Config,
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.model_name == "distilroberta-base"
        assert cfg.num_labels == 2
        assert cfg.max_length == 512
        assert cfg.model_path is None
        assert cfg.device is None

    def test_custom_values(self):
        cfg = ModelConfig(model_name="bert-base-uncased", num_labels=3, max_length=128, device="cpu")
        assert cfg.model_name == "bert-base-uncased"
        assert cfg.num_labels == 3
        assert cfg.max_length == 128
        assert cfg.device == "cpu"

    def test_to_dict_roundtrip(self):
        cfg = ModelConfig(model_name="roberta-base", num_labels=2, max_length=256)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["model_name"] == "roberta-base"
        cfg2 = ModelConfig.from_dict(d)
        assert cfg2.model_name == cfg.model_name
        assert cfg2.num_labels == cfg.num_labels
        assert cfg2.max_length == cfg.max_length

    def test_save_to_directory(self, tmp_path):
        cfg = ModelConfig(model_name="distilbert-base-uncased")
        cfg.save(str(tmp_path))
        saved = tmp_path / "model_config.json"
        assert saved.exists()
        with open(saved) as f:
            data = json.load(f)
        assert data["model_name"] == "distilbert-base-uncased"

    def test_save_to_file(self, tmp_path):
        out = str(tmp_path / "my_model_config.json")
        cfg = ModelConfig(model_name="distilbert-base-uncased")
        cfg.save(out)
        assert os.path.exists(out)

    def test_load_roundtrip(self, tmp_path):
        cfg = ModelConfig(model_name="bert-base-uncased", num_labels=4)
        cfg.save(str(tmp_path))
        cfg2 = ModelConfig.load(str(tmp_path / "model_config.json"))
        assert cfg2.model_name == "bert-base-uncased"
        assert cfg2.num_labels == 4


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.output_dir == "./output"
        assert cfg.num_train_epochs == 5
        assert cfg.learning_rate == pytest.approx(5e-5)
        assert cfg.fp16 is True

    def test_to_dict_roundtrip(self):
        cfg = TrainingConfig(num_train_epochs=3, learning_rate=2e-5)
        d = cfg.to_dict()
        cfg2 = TrainingConfig.from_dict(d)
        assert cfg2.num_train_epochs == 3
        assert cfg2.learning_rate == pytest.approx(2e-5)

    def test_save_and_load(self, tmp_path):
        cfg = TrainingConfig(num_train_epochs=10, per_device_train_batch_size=8)
        cfg.save(str(tmp_path))
        cfg2 = TrainingConfig.load(str(tmp_path / "training_config.json"))
        assert cfg2.num_train_epochs == 10
        assert cfg2.per_device_train_batch_size == 8

    def test_report_to_default_list(self):
        cfg = TrainingConfig()
        assert isinstance(cfg.report_to, list)
        assert "none" in cfg.report_to


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------

class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.text_column == "text"
        assert cfg.label_column == "label"
        assert cfg.test_size == pytest.approx(0.15)
        assert cfg.balance_classes is False
        assert cfg.n_splits == 5

    def test_to_dict_contains_all_fields(self):
        cfg = DataConfig(train_file="train.csv", test_size=0.2)
        d = cfg.to_dict()
        assert "train_file" in d
        assert "test_size" in d
        assert d["test_size"] == pytest.approx(0.2)

    def test_save_and_load(self, tmp_path):
        cfg = DataConfig(train_file="train.csv", balance_classes=True)
        cfg.save(str(tmp_path))
        cfg2 = DataConfig.load(str(tmp_path / "data_config.json"))
        assert cfg2.train_file == "train.csv"
        assert cfg2.balance_classes is True


# ---------------------------------------------------------------------------
# InferenceConfig
# ---------------------------------------------------------------------------

class TestInferenceConfig:
    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.batch_size == 32
        assert cfg.max_length == 512
        assert cfg.output_file is None

    def test_save_and_load(self, tmp_path):
        cfg = InferenceConfig(batch_size=64, max_length=256)
        cfg.save(str(tmp_path))
        cfg2 = InferenceConfig.load(str(tmp_path / "inference_config.json"))
        assert cfg2.batch_size == 64
        assert cfg2.max_length == 256


# ---------------------------------------------------------------------------
# Master Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.inference, InferenceConfig)
        assert cfg.category is None
        assert cfg.categories is None

    def test_to_dict_structure(self):
        cfg = Config()
        d = cfg.to_dict()
        assert "model" in d
        assert "training" in d
        assert "data" in d
        assert "inference" in d

    def test_from_dict_roundtrip(self):
        cfg = Config(
            model=ModelConfig(model_name="roberta-base"),
            category="architecture",
        )
        d = cfg.to_dict()
        cfg2 = Config.from_dict(d)
        assert cfg2.model.model_name == "roberta-base"
        assert cfg2.category == "architecture"

    def test_save_and_load(self, tmp_path):
        cfg = Config(
            model=ModelConfig(model_name="distilbert-base-uncased", num_labels=3),
            category="defect",
        )
        cfg.save(str(tmp_path))
        assert (tmp_path / "config.json").exists()
        cfg2 = Config.load(str(tmp_path / "config.json"))
        assert cfg2.model.model_name == "distilbert-base-uncased"
        assert cfg2.model.num_labels == 3
        assert cfg2.category == "defect"

    def test_save_to_explicit_file(self, tmp_path):
        out = str(tmp_path / "explicit_config.json")
        cfg = Config()
        cfg.save(out)
        assert os.path.exists(out)

    def test_categories_list(self):
        cfg = Config(categories=["arch", "defect", "doc"])
        d = cfg.to_dict()
        cfg2 = Config.from_dict(d)
        assert cfg2.categories == ["arch", "defect", "doc"]
