#!/usr/bin/env python
"""
Script for training technical debt classification models.
"""

import os
# Disable oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import json
import pandas as pd
import torch
from transformers import TrainingArguments, AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from typing import Optional, List, Dict, Any
from sklearn.model_selection import train_test_split

from tdsuite.models.transformer import TransformerModel
from tdsuite.data.dataset import TDDataset, TDProcessor, BinaryTDProcessor
from tdsuite.trainers import TDTrainer
from tdsuite.config.config import Config, ModelConfig, TrainingConfig, DataConfig
from transformers import AutoModelForSequenceClassification


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a technical debt classification model")
    
    # Data arguments
    parser.add_argument("--data_file", type=str, required=True,
                      help="Path to data file or Hugging Face dataset name")
    parser.add_argument("--text_column", type=str, default="text",
                      help="Name of the text column")
    parser.add_argument("--label_column", type=str, default="label",
                      help="Name of the label column")
    parser.add_argument("--is_huggingface_dataset", action="store_true",
                      help="Whether the data is a Hugging Face dataset")
    parser.add_argument("--numeric_labels", action="store_true",
                      help="Whether the labels are already numeric (0 or 1)")
    parser.add_argument("--positive_category", type=str,
                      help="Positive category for binary classification")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                      help="Name of the model to use")
    parser.add_argument("--max_length", type=int, default=512,
                      help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save the model and outputs")
    parser.add_argument("--num_epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500,
                      help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                      help="Number of gradient accumulation steps")
    
    # Cross-validation arguments
    parser.add_argument("--cross_validation", action="store_true",
                      help="Whether to use cross-validation")
    parser.add_argument("--n_splits", type=int, default=5,
                      help="Number of splits for cross-validation")
    
    # Additional arguments
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                      help="Device to use (cuda, cpu, or None for auto-detection)")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Check for GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. This model requires a GPU to run.")
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create processor
    processor = BinaryTDProcessor(tokenizer, max_length=args.max_length)
    
    # Load and prepare data
    data = processor.load_data(
        args.data_file,
        text_column=args.text_column,
        label_column=args.label_column
    )
    
    data = processor.prepare_binary_data(
        data,
        positive_category=args.positive_category,
        text_column=args.text_column,
        label_column=args.label_column,
        numeric_labels=args.numeric_labels
    )
    
    # Split data into train and eval sets
    train_data, eval_data = train_test_split(
        data,
        test_size=0.1,  # Use 10% for evaluation
        random_state=args.seed,
        stratify=data["label_idx"]
    )
    
    # Create datasets
    train_dataset = processor.create_dataset(
        train_data,
        text_column=args.text_column,
        label_column="label_idx"
    )
    
    eval_dataset = processor.create_dataset(
        eval_data,
        text_column=args.text_column,
        label_column="label_idx"
    )
    
    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )
    model.to(device)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch",  # Always use epoch-based evaluation
        save_strategy="epoch",
        load_best_model_at_end=not args.cross_validation,  # Only load best model when not using cross-validation
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        # Disable wandb and codecarbon by default
        report_to="none",  # Disable wandb
        disable_tqdm=False,  # Keep progress bars
        no_cuda=False,  # Use CUDA if available
    )
    
    # Create trainer
    trainer = TDTrainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        output_dir=args.output_dir,
        track_emissions=True,  # Enable emissions tracking by default
        n_splits=args.n_splits,
    )
    
    if args.cross_validation:
        # Train with cross-validation
        trainer.train_with_cross_validation(
            train_dataset,
            eval_dataset=eval_dataset,
            is_binary=True
        )
    else:
        # Train without cross-validation
        trainer.train(train_dataset, eval_dataset=eval_dataset)
    
    # Save model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training configuration
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main() 