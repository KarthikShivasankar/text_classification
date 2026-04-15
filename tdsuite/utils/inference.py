"""Inference utilities for technical debt classification."""

import argparse
import json
import os
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import warnings

from tdsuite.config.config import Config
from tdsuite.data.dataset import TDProcessor
from ..models.transformer import TransformerModel

# Suppress FutureWarning from codecarbon
warnings.simplefilter(action='ignore', category=FutureWarning)


class InferenceEngine:
    """Engine for performing inference with technical debt classification models."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the inference engine.

        Args:
            model_path: Path to the model checkpoint or Hugging Face model name
            model_name: Optional model name for Hugging Face models
            max_length: Maximum sequence length
            device: Device to use for inference
        """
        self.model_path = model_path
        self.model_name = model_name or model_path
        self.max_length = max_length
        self.device = device
        self.show_progress = True  # Default to showing progress bars

        # Load model and tokenizer
        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _load_model(self) -> TransformerModel:
        """
        Load the model from path or Hugging Face.

        Returns:
            Loaded model
        """
        # Check if model_path is a local path
        if self.model_path and os.path.exists(self.model_path):
            # Load from local checkpoint
            model = TransformerModel.load_from_checkpoint(self.model_path)
        else:
            # Load from Hugging Face
            model = TransformerModel.from_pretrained(
                self.model_name,
                num_labels=2,  # Default to binary classification
                max_length=self.max_length,
                device=self.device,
            )

        model.to(self.device)
        model.eval()
        return model

    def predict_single(self, text: str) -> Dict[str, Union[str, float, List[float]]]:
        """
        Predict the class for a single text.

        Args:
            text: Input text

        Returns:
            Dictionary containing the text, predicted class, predicted probability,
            and class probabilities
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_prob = probabilities[0][predicted_class].item()

        # Get class probabilities
        class_probs = probabilities[0].tolist()

        return {
            "text": text,
            "predicted_class": predicted_class,
            "predicted_probability": predicted_prob,
            "class_probabilities": class_probs,
        }

    def predict_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Dict[str, Union[str, float, List[float]]]]:
        """
        Predict classes for a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for inference

        Returns:
            List of dictionaries containing predictions for each text
        """
        predictions = []

        # Create batches with tqdm progress bar if enabled
        num_batches = (len(texts) + batch_size - 1) // batch_size
        batches = range(0, len(texts), batch_size)
        
        # Use tqdm for progress tracking if enabled
        if self.show_progress:
            from tqdm import tqdm
            batches = tqdm(batches, total=num_batches, desc="Processing batches", unit="batch")

        # Process in batches
        for i in batches:
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                predicted_probs = probabilities[torch.arange(len(batch_texts)), predicted_classes]

            # Convert to list of dictionaries
            for j, (text, pred_class, pred_prob) in enumerate(
                zip(batch_texts, predicted_classes, predicted_probs)
            ):
                predictions.append(
                    {
                        "text": text,
                        "predicted_class": pred_class.item(),
                        "predicted_probability": pred_prob.item(),
                        "class_probabilities": probabilities[j].tolist(),
                    }
                )

        return predictions

    def predict_from_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        text_column: str = "text",
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        Read texts from a file, perform inference, and optionally save predictions.

        Args:
            input_file: Path to input file (CSV or JSON)
            output_file: Optional path to save predictions
            text_column: Name of the column containing texts
            batch_size: Batch size for inference

        Returns:
            DataFrame containing the predictions
        """
        # Read input file
        if input_file.endswith(".csv"):
            df = pd.read_csv(input_file)
        elif input_file.endswith(".json"):
            df = pd.read_json(input_file)
        else:
            raise ValueError("Input file must be CSV or JSON")

        # Get texts
        texts = df[text_column].tolist()

        # Get predictions
        predictions = self.predict_batch(texts, batch_size)

        # Convert to DataFrame
        results_df = pd.DataFrame(predictions)

        # Merge with original DataFrame to preserve labels if they exist
        if "label" in df.columns:
            results_df["label"] = df["label"]
            
            # Map 0 to "non_" and 1 to the positive category
            unique_labels = df["label"].unique()
            if len(unique_labels) != 2:
                raise ValueError("Binary classification requires exactly 2 unique labels")
            
            # Find the positive category (the one that doesn't start with "non_")
            positive_label = None
            non_label = None
            for label in unique_labels:
                if str(label).startswith("non_"):
                    non_label = label
                else:
                    positive_label = label
            
            # If we couldn't identify the labels by prefix, use the first one as non_ and second as positive
            if non_label is None or positive_label is None:
                sorted_labels = sorted(unique_labels)
                non_label = sorted_labels[0]
                positive_label = sorted_labels[1]
            
            # Create mapping: 0 -> non_, 1 -> positive
            label_map = {0: non_label, 1: positive_label}
            results_df["predicted_class"] = results_df["predicted_class"].map(label_map)

        # Save predictions if output file is specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results_df.to_csv(output_file, index=False)

        return results_df


class EnsembleInferenceEngine:
    """Engine for performing inference with an ensemble of models."""

    def __init__(
        self,
        model_paths: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize the ensemble inference engine.

        Args:
            model_paths: Paths to local model checkpoints
            model_names: Names of models on Hugging Face
            max_length: Maximum sequence length
            device: Device to use for inference
            weights: Optional weights for each model in the ensemble
        """
        print(f"Initializing EnsembleInferenceEngine with:")
        print(f"  model_paths: {model_paths}")
        print(f"  model_names: {model_names}")
        print(f"  device: {device}")
        
        self.model_paths = model_paths or []
        self.model_names = model_names or []
        self.max_length = max_length
        self.device = device
        self.show_progress = True

        # Validate inputs
        if not self.model_paths and not self.model_names:
            raise ValueError("At least one of model_paths or model_names must be provided")

        # Load models and tokenizers
        self.models = []
        self.tokenizers = []
        
        # Load models from local paths
        for model_path in self.model_paths:
            try:
                print(f"Loading model from path: {model_path}")
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model.to(self.device)
                model.eval()
                self.models.append(model)
                self.tokenizers.append(tokenizer)
                print(f"Successfully loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {str(e)}")
                continue
            
        # Load models from Hugging Face
        for model_name in self.model_names:
            try:
                print(f"Loading model from Hugging Face: {model_name}")
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model.to(self.device)
                model.eval()
                self.models.append(model)
                self.tokenizers.append(tokenizer)
                print(f"Successfully loaded model {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {str(e)}")
                continue
            
        # Validate that we loaded at least one model
        if not self.models:
            raise ValueError("No models were successfully loaded")
            
        # Set weights (default to equal weights if not provided)
        if weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            if len(weights) != len(self.models):
                raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(self.models)})")
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
        
        print(f"Successfully loaded {len(self.models)} models with weights: {self.weights}")

    def predict_single(self, text: str) -> Dict[str, Union[str, float, List[float]]]:
        """
        Predict the class for a single text using ensemble of models.

        Args:
            text: Input text

        Returns:
            Dictionary containing the text, predicted class, predicted probability,
            and class probabilities
        """
        if not text:
            raise ValueError("Input text is empty")
            
        # Get predictions from each model
        all_probabilities = []
        
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            try:
                # Tokenize input
                inputs = tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    all_probabilities.append(probabilities[0].cpu().numpy())
            except Exception as e:
                print(f"Error processing model {i+1}: {str(e)}")
                continue
        
        if not all_probabilities:
            # Handle the case where no models produced valid probabilities
            print("No valid predictions from any model. Returning default predictions.")
            num_classes = 2  # Default to binary classification
            return {
                "text": text,
                "predicted_class": 0,
                "predicted_probability": 1.0/num_classes,
                "class_probabilities": [1.0/num_classes] * num_classes,
            }
        
        # Weighted average of probabilities
        weighted_probs = np.zeros_like(all_probabilities[0])
        for i, probs in enumerate(all_probabilities):
            weighted_probs += probs * self.weights[i]
        
        # Get predicted class and probability
        predicted_class = np.argmax(weighted_probs)
        predicted_prob = weighted_probs[predicted_class]
        
        return {
            "text": text,
            "predicted_class": int(predicted_class),
            "predicted_probability": float(predicted_prob),
            "class_probabilities": weighted_probs.tolist(),
        }

    def predict_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Dict[str, Union[str, float, List[float]]]]:
        """
        Predict classes for a batch of texts using an ensemble of models.

        Args:
            texts: List of input texts
            batch_size: Batch size for inference

        Returns:
            List of dictionaries containing predictions for each text
        """
        if not texts:
            raise ValueError("Input texts list is empty")
            
        if not self.models or not self.tokenizers:
            raise ValueError("No models or tokenizers available in the ensemble")
            
        print(f"Processing {len(texts)} texts with {len(self.models)} models")
        
        # Process in batches
        predictions = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Use tqdm for progress tracking if enabled
        if self.show_progress:
            from tqdm import tqdm
            batches = tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Processing batches")
        else:
            batches = range(0, len(texts), batch_size)
        
        for i in batches:
            batch_texts = texts[i:i + batch_size]
            batch_predictions = self._process_batch(batch_texts)
            predictions.extend(batch_predictions)
            
        return predictions

    def _process_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float, List[float]]]]:
        """Process a single batch of texts."""
        all_probabilities = []
        
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            try:
                # Tokenize inputs
                inputs = tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    all_probabilities.append(probabilities.cpu().numpy())
            except Exception as e:
                print(f"Error processing model {i+1}: {str(e)}")
                continue
        
        if not all_probabilities:
            # Handle the case where no models produced valid probabilities
            print("No valid predictions from any model. Returning default predictions.")
            num_classes = 2  # Default to binary classification
            predictions = []
            for text in texts:
                predictions.append({
                    "text": text,
                    "predicted_class": 0,
                    "predicted_probability": 1.0/num_classes,
                    "class_probabilities": [1.0/num_classes] * num_classes,
                })
            return predictions
        
        # Stack and average probabilities across models
        num_examples = len(texts)
        num_classes = all_probabilities[0].shape[1]
        weighted_probabilities = np.zeros((num_examples, num_classes))
        
        for i, probs in enumerate(all_probabilities):
            weighted_probabilities += probs * self.weights[i]
        
        # Get predicted classes and probabilities
        predicted_classes = np.argmax(weighted_probabilities, axis=1)
        predicted_probs = np.array([
            weighted_probabilities[j, predicted_classes[j]] 
            for j in range(len(texts))
        ])
        
        # Convert to list of dictionaries
        predictions = []
        for j, (text, pred_class, pred_prob) in enumerate(
            zip(texts, predicted_classes, predicted_probs)
        ):
            predictions.append(
                {
                    "text": text,
                    "predicted_class": int(pred_class),
                    "predicted_probability": float(pred_prob),
                    "class_probabilities": weighted_probabilities[j].tolist(),
                }
            )
        
        return predictions

    def predict_from_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        text_column: str = "text",
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        Read texts from a file, perform ensemble inference, and optionally save predictions.

        Args:
            input_file: Path to input file (CSV or JSON)
            output_file: Optional path to save predictions
            text_column: Name of the column containing texts
            batch_size: Batch size for inference

        Returns:
            DataFrame containing the predictions
        """
        # Read input file
        if input_file.endswith(".csv"):
            df = pd.read_csv(input_file)
        elif input_file.endswith(".json"):
            df = pd.read_json(input_file)
        else:
            raise ValueError("Input file must be CSV or JSON")

        # Get texts
        texts = df[text_column].tolist()

        # Get predictions
        predictions = self.predict_batch(texts, batch_size)

        # Convert to DataFrame
        results_df = pd.DataFrame(predictions)

        # Merge with original DataFrame to preserve labels if they exist
        if "label" in df.columns:
            results_df["label"] = df["label"]
            
            # Map 0 to "non_" and 1 to the positive category
            unique_labels = df["label"].unique()
            if len(unique_labels) != 2:
                raise ValueError("Binary classification requires exactly 2 unique labels")
            
            # Find the positive category (the one that doesn't start with "non_")
            positive_label = None
            non_label = None
            for label in unique_labels:
                if str(label).startswith("non_"):
                    non_label = label
                else:
                    positive_label = label
            
            # If we couldn't identify the labels by prefix, use the first one as non_ and second as positive
            if non_label is None or positive_label is None:
                sorted_labels = sorted(unique_labels)
                non_label = sorted_labels[0]
                positive_label = sorted_labels[1]
            
            # Create mapping: 0 -> non_, 1 -> positive
            label_map = {0: non_label, 1: positive_label}
            results_df["predicted_class"] = results_df["predicted_class"].map(label_map)

        # Save predictions if output file is specified
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            results_df.to_csv(output_file, index=False)

        return results_df


def main():
    """Main function for inference."""
    parser = argparse.ArgumentParser(description="Inference for technical debt classification")
    
    # Model arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model_path", type=str, help="Path to a local model")
    model_group.add_argument("--model_name", type=str, help="Name of a model on Hugging Face")
    model_group.add_argument("--model_paths", type=str, nargs="+", help="Paths to multiple local models")
    model_group.add_argument("--model_names", type=str, nargs="+", help="Names of multiple models on Hugging Face")
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Text to classify")
    input_group.add_argument("--input_file", type=str, help="Path to a file with texts (one per line)")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, help="Path to save predictions")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--device", type=str, help="Device to use for inference (cuda, cpu)")
    parser.add_argument("--weights", type=float, nargs="+", help="Weights for ensemble models")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if using ensemble
    if args.model_paths or args.model_names:
        # Create ensemble inference engine
        engine = EnsembleInferenceEngine(
            model_paths=args.model_paths or args.model_names,
            model_names=args.model_names,
            max_length=args.max_length,
            device=args.device,
            weights=args.weights,
        )
    else:
        # Create single model inference engine
        engine = InferenceEngine(
            model_path=args.model_path,
            model_name=args.model_name,
            max_length=args.max_length,
            device=args.device,
        )
    
    # Perform inference
    if args.text is not None:
        result = engine.predict_single(args.text)
        print(json.dumps(result, indent=2))
    else:
        df = engine.predict_from_file(args.input_file, args.output_file)
        if args.output_file is None:
            print(df.to_string()) 