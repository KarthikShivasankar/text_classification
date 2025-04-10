"""Base model classes for technical debt classification."""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Optional, Union, Any


def load_model_and_tokenizer(model_name: str, num_labels: int, max_length: int = 512):
    """
    Load a pre-trained model and tokenizer.

    Args:
        model_name: Name of the pre-trained model
        num_labels: Number of labels for classification
        max_length: Maximum sequence length

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    return model, tokenizer


class BaseModel(nn.Module):
    """Base model class for technical debt classification."""

    def __init__(self, model, class_weights=None, device=None):
        """
        Initialize the model.

        Args:
            model: The base model
            class_weights: Weights for each class
            device: Device to use (cuda or cpu)
        """
        super().__init__()
        self.model = model
        self.class_weights = class_weights

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float).to(
                self.device
            )

        # Only move the model to the device if it's not None
        if self.model is not None:
            self.model.to(self.device)

    def compute_loss(self, logits, labels):
        """
        Compute the weighted loss.

        Args:
            logits: Model output logits
            labels: Ground truth labels

        Returns:
            Loss value
        """
        if self.class_weights is not None:
            loss_func = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_func = nn.CrossEntropyLoss()

        return loss_func(logits, labels)

    def save_model(self, output_dir: str):
        """
        Save the model to disk.

        Args:
            output_dir: Directory to save the model
        """
        self.model.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(
        cls, model_path: str, num_labels: int, max_length: int = 512, class_weights=None, device=None
    ):
        """
        Load a model from disk.

        Args:
            model_path: Path to the saved model
            num_labels: Number of labels for classification
            max_length: Maximum sequence length
            class_weights: Weights for each class
            device: Device to use (cuda or cpu)

        Returns:
            BaseModel
        """
        # Create a TransformerModel instance
        from .transformer import TransformerModel
        return TransformerModel(
            model_name=model_path,
            num_labels=num_labels,
            max_length=max_length,
            class_weights=class_weights,
            device=device
        ) 