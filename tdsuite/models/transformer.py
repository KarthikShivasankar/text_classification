"""Transformer model implementation for technical debt classification."""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Optional, Union, Any

from .base import BaseModel, load_model_and_tokenizer


class TransformerModel(BaseModel):
    """Transformer-based model for technical debt classification."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        max_length: int = 512,
        class_weights=None,
        device=None,
    ):
        """
        Initialize the transformer model.

        Args:
            model_name: Name of the pre-trained model or path to a local model
            num_labels: Number of labels for classification
            max_length: Maximum sequence length
            class_weights: Weights for each class
            device: Device to use (cuda or cpu)
        """
        # Store parameters
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.class_weights = class_weights
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the parent class first
        super().__init__(model=None, class_weights=None, device=None)
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        # Update the model and class_weights in the parent class
        self.model = self.model
        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float).to(self.device)
        
        # Move the model to the device
        self.model.to(self.device)

    def _load_model_and_tokenizer(self):
        """
        Load model and tokenizer from Hugging Face or local path.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        # Check if model_name is a local path
        is_local_path = os.path.exists(self.model_name)
        
        if is_local_path:
            # Load from local path
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels
            )
        else:
            # Load from Hugging Face
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=self.max_length)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels
            )
        
        # Move model to device
        model = model.to(self.device)
        
        return model, tokenizer

    def predict(self, texts: Union[str, List[str]], batch_size: int = 32):
        """
        Make predictions on the input texts.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for inference

        Returns:
            Predictions and probabilities
        """
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize inputs
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to device
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        # Set model to evaluation mode
        self.model.eval()

        # Make predictions
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

        # Move to CPU for numpy conversion
        predictions = predictions.cpu().numpy()
        probabilities = probabilities.cpu().numpy()

        return predictions, probabilities

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device=None):
        """
        Load a model from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint directory
            device: Device to use (cuda or cpu)

        Returns:
            Loaded model
        """
        # Load config
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                config = json.load(f)
            
            model_name = config.get("model_name", checkpoint_path)
            num_labels = config.get("num_labels", 2)
            max_length = config.get("max_length", 512)
        else:
            # Default values if config not found
            model_name = checkpoint_path
            num_labels = 2
            max_length = 512

        # Create model
        model = cls(
            model_name=model_name,
            num_labels=num_labels,
            max_length=max_length,
            device=device,
        )

        return model

    def to(self, device):
        """
        Move the model to the specified device.
        
        Args:
            device: Device to move the model to
            
        Returns:
            Self for method chaining
        """
        self.device = device
        self.model = self.model.to(device)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device)
        return self
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        Forward pass of the model.
        
        This method is required by the Hugging Face Trainer.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            position_ids: Position IDs
            head_mask: Head mask
            inputs_embeds: Input embeddings
            labels: Labels for loss computation
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs
        """
        # Get the signature of the model's forward method
        import inspect
        model_signature = inspect.signature(self.model.forward)
        model_params = model_signature.parameters.keys()
        
        # Filter out parameters that aren't accepted by the model
        filtered_inputs = {}
        for key, value in {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids,
            'head_mask': head_mask,
            'inputs_embeds': inputs_embeds,
            'labels': labels,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict
        }.items():
            if key in model_params and value is not None:
                filtered_inputs[key] = value
        
        # Call the model with the filtered inputs
        return self.model(**filtered_inputs)
    
    def save_pretrained(self, output_dir):
        """
        Save the model and tokenizer to the specified directory.
        
        Args:
            output_dir: Directory to save the model and tokenizer
        """
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model
        self.model.save_pretrained(output_dir)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save model configuration
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "class_weights": self.class_weights.tolist() if self.class_weights is not None else None,
            "device": str(self.device)  # Convert device to string
        }
        
        with open(os.path.join(output_dir, "model_config.json"), "w") as f:
            import json
            json.dump(config, f, indent=4) 