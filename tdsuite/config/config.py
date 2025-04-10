"""Configuration classes for technical debt classification."""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any


@dataclass
class ModelConfig:
    """Configuration for the model."""

    model_name: str = "distilroberta-base"
    model_path: Optional[str] = None
    num_labels: int = 2
    max_length: int = 512
    device: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary."""
        config_dict = asdict(self)
        # Convert device to string if it's a torch.device object
        if hasattr(config_dict['device'], '__class__') and config_dict['device'].__class__.__name__ == 'device':
            config_dict['device'] = str(config_dict['device'])
        return config_dict

    def save(self, output_dir: str):
        """Save the configuration to a JSON file."""
        # Check if output_dir is a file path or a directory path
        if os.path.splitext(output_dir)[1]:  # If it has an extension, it's a file path
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            # Save the configuration to the file
            with open(output_dir, "w") as f:
                json.dump(self.to_dict(), f, indent=4)
        else:
            # It's a directory path, create it and save the configuration to a file in it
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
    eval_strategy: str = "epoch"
    logging_dir: Optional[str] = None
    logging_steps: int = 100
    dataloader_num_workers: int = 8
    report_to: List[str] = field(default_factory=lambda: ["none"])
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    def save(self, output_dir: str):
        """Save the configuration to a JSON file."""
        # Check if output_dir is a file path or a directory path
        if os.path.splitext(output_dir)[1]:  # If it has an extension, it's a file path
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            # Save the configuration to the file
            with open(output_dir, "w") as f:
                json.dump(self.to_dict(), f, indent=4)
        else:
            # It's a directory path, create it and save the configuration to a file in it
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

    # Local file paths
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Hugging Face dataset
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    dataset_split: str = "train"
    
    # Column names
    text_column: str = "text"
    label_column: str = "label"
    
    # Data processing
    max_length: int = 512
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    test_size: float = 0.15
    random_state: int = 42
    n_splits: int = 5
    
    # Class balancing
    balance_classes: bool = False
    
    # Repository extraction
    extract_top_repo: bool = False
    repo_column: Optional[str] = None

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    def save(self, output_dir: str):
        """Save the configuration to a JSON file."""
        # Check if output_dir is a file path or a directory path
        if os.path.splitext(output_dir)[1]:  # If it has an extension, it's a file path
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            # Save the configuration to the file
            with open(output_dir, "w") as f:
                json.dump(self.to_dict(), f, indent=4)
        else:
            # It's a directory path, create it and save the configuration to a file in it
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
class InferenceConfig:
    """Configuration for inference."""

    batch_size: int = 32
    num_workers: int = 4
    output_file: Optional[str] = None
    device: Optional[str] = None
    max_length: int = 512

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)

    def save(self, output_dir: str):
        """Save the configuration to a JSON file."""
        # Check if output_dir is a file path or a directory path
        if os.path.splitext(output_dir)[1]:  # If it has an extension, it's a file path
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            # Save the configuration to the file
            with open(output_dir, "w") as f:
                json.dump(self.to_dict(), f, indent=4)
        else:
            # It's a directory path, create it and save the configuration to a file in it
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "inference_config.json"), "w") as f:
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
class Config:
    """Main configuration class that combines all configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    category: Optional[str] = None
    categories: Optional[List[str]] = None

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict(),
            "inference": self.inference.to_dict(),
            "category": self.category,
            "categories": self.categories
        }

    def save(self, output_dir: str):
        """Save the configuration to a JSON file."""
        # Check if output_dir is a file path or a directory path
        if os.path.splitext(output_dir)[1]:  # If it has an extension, it's a file path
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            # Save the configuration to the file
            with open(output_dir, "w") as f:
                json.dump(self.to_dict(), f, indent=4)
        else:
            # It's a directory path, create it and save the configuration to a file in it
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create a configuration from a dictionary."""
        return cls(
            model=ModelConfig.from_dict(config_dict.get("model", {})),
            training=TrainingConfig.from_dict(config_dict.get("training", {})),
            data=DataConfig.from_dict(config_dict.get("data", {})),
            inference=InferenceConfig.from_dict(config_dict.get("inference", {})),
            category=config_dict.get("category"),
            categories=config_dict.get("categories")
        )

    @classmethod
    def load(cls, config_path: str):
        """Load a configuration from a JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict) 