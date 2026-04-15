#!/usr/bin/env python
"""
Script for training technical debt classification models.
"""

import os
import argparse
import json
import pandas as pd
import torch
from transformers import TrainingArguments, AutoTokenizer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from typing import Optional, List, Dict, Any

from tdsuite.models.transformer import TransformerModel
from tdsuite.data.dataset import TDDataset, TDProcessor, BinaryTDProcessor, MultiTDProcessor
from tdsuite.trainers import TDTrainer
from tdsuite.config.config import Config, ModelConfig, TrainingConfig, DataConfig


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
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                      help="Path to local model or Hugging Face model name")
    parser.add_argument("--max_length", type=int, default=512,
                      help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500,
                      help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                      help="Number of gradient accumulation steps")
    
    # Classification type arguments
    parser.add_argument("--classification_type", type=str, choices=["binary", "multi"],
                      required=True, help="Type of classification")
    parser.add_argument("--positive_category", type=str,
                      help="Positive category for binary classification")
    parser.add_argument("--categories", type=str, nargs="+",
                      help="Categories for multi-class classification")
    parser.add_argument("--numeric_labels", action="store_true",
                      help="Whether the labels are already numeric (0 or 1 for binary, 0,1,2... for multi)")
    parser.add_argument("--num_classes", type=int,
                      help="Number of classes (required if numeric_labels=True for multi-class)")
    
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
    
    # Create processor based on classification type
    if args.classification_type == "binary":
        processor = BinaryTDProcessor(tokenizer, max_length=args.max_length)
    else:
        if args.numeric_labels and not args.num_classes:
            raise ValueError("num_classes must be specified for multi-class classification with numeric labels")
        processor = MultiTDProcessor(tokenizer, max_length=args.max_length)
    
    # Load and prepare data
    data = processor.load_data(
        args.data_file,
        text_column=args.text_column,
        label_column=args.label_column
    )
    
    if args.classification_type == "binary":
        data = processor.prepare_binary_data(
            data,
            positive_category=args.positive_category,
            text_column=args.text_column,
            label_column=args.label_column,
            numeric_labels=args.numeric_labels
        )
    else:
        data, cat2idx, idx2cat = processor.prepare_multi_class_data(
            data,
            categories=args.categories,
            text_column=args.text_column,
            label_column=args.label_column,
            numeric_labels=args.numeric_labels,
            num_classes=args.num_classes
        )
    
    # Create dataset
    dataset = processor.create_dataset(
        data,
        text_column=args.text_column,
        label_column="label_idx"
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create model configurations
    model_config = ModelConfig(
        model_name=args.model_name,
        num_labels=2 if args.classification_type == "binary" else (args.num_classes if args.numeric_labels else len(args.categories)),
        max_length=args.max_length,
        device=device
    )
    
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    data_config = DataConfig(
        text_column=args.text_column,
        label_column="label_idx",
        max_length=args.max_length
    )
    
    # Initialize model
    model = TransformerModel.from_pretrained(
        args.model_name,
        num_labels=model_config.num_labels,
        max_length=model_config.max_length,
        device=device
    )
    
    # Create trainer
    trainer = TDTrainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_config,
        data_config=data_config,
        output_dir=args.output_dir
    )
    
    # Train model
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Save model and configurations
    model.save_pretrained(args.output_dir)
    model_config.save(os.path.join(args.output_dir, "model_config.json"))
    training_config.save(os.path.join(args.output_dir, "training_config.json"))
    data_config.save(os.path.join(args.output_dir, "data_config.json"))
    
    # Save category mappings for multi-class classification
    if args.classification_type == "multi":
        with open(os.path.join(args.output_dir, "category_mappings.json"), "w") as f:
            json.dump({
                "cat2idx": cat2idx,
                "idx2cat": idx2cat
            }, f, indent=2)


if __name__ == "__main__":
    main() 