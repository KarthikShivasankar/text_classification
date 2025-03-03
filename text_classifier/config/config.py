"""Configuration classes for text classification."""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any


@dataclass
class ModelConfig:
    """Configuration for the model."""

    model_name: str = "distilroberta-base"
    num_labels: int = 2
    max_length: int = 512

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    def save(self, output_dir: str):
        """Save the configuration to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create a configuration from a dictionary."""
        return cls(**config_dict)

    @classmethod
    def load(cls, config_path: str):
        """Load a configuration from a JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    output_dir: str = "./output"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    save_strategy: str = "epoch"
    save_total_limit: int = 1
    evaluation_strategy: str = "epoch"
    logging_dir: Optional[str] = None
    logging_steps: int = 100
    dataloader_num_workers: int = 8
    report_to: List[str] = field(default_factory=lambda: ["none"])

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    def save(self, output_dir: str):
        """Save the configuration to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create a configuration from a dictionary."""
        return cls(**config_dict)

    @classmethod
    def load(cls, config_path: str):
        """Load a configuration from a JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class DataConfig:
    """Configuration for data processing."""

    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    test_size: float = 0.15
    random_state: int = 42
    n_splits: int = 5

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    def save(self, output_dir: str):
        """Save the configuration to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "data_config.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create a configuration from a dictionary."""
        return cls(**config_dict)

    @classmethod
    def load(cls, config_path: str):
        """Load a configuration from a JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class BinaryClassificationConfig:
    """Configuration for binary classification."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    category: str = ""
    balance_classes: bool = True
    extract_top_repo: bool = True

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict(),
            "category": self.category,
            "balance_classes": self.balance_classes,
            "extract_top_repo": self.extract_top_repo,
        }

    def save(self, output_dir: str):
        """Save the configuration to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        with open(
            os.path.join(output_dir, "binary_classification_config.json"), "w"
        ) as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create a configuration from a dictionary."""
        model_config = ModelConfig.from_dict(config_dict.pop("model"))
        training_config = TrainingConfig.from_dict(config_dict.pop("training"))
        data_config = DataConfig.from_dict(config_dict.pop("data"))

        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            **config_dict
        )

    @classmethod
    def load(cls, config_path: str):
        """Load a configuration from a JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class MultiClassificationConfig:
    """Configuration for multi-class classification."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    categories: List[str] = field(default_factory=list)
    extract_top_repos: bool = True

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict(),
            "categories": self.categories,
            "extract_top_repos": self.extract_top_repos,
        }

    def save(self, output_dir: str):
        """Save the configuration to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        with open(
            os.path.join(output_dir, "multi_classification_config.json"), "w"
        ) as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create a configuration from a dictionary."""
        model_config = ModelConfig.from_dict(config_dict.pop("model"))
        training_config = TrainingConfig.from_dict(config_dict.pop("training"))
        data_config = DataConfig.from_dict(config_dict.pop("data"))

        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            **config_dict
        )

    @classmethod
    def load(cls, config_path: str):
        """Load a configuration from a JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
