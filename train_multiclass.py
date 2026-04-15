#!/usr/bin/env python
"""
Script for training multi-class classification models.
"""

import os
import argparse
import pandas as pd
import torch
from transformers import TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from text_classifier.models.model import load_model_and_tokenizer
from text_classifier.data.dataset import MultiClassificationProcessor, DataProcessor
from text_classifier.trainers.trainer import MultiClassificationTrainer
from text_classifier.config.config import (
    MultiClassificationConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a multi-class classification model"
    )

    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data_file", type=str, help="Path to data file")
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Output directory"
    )
    parser.add_argument(
        "--model_name", type=str, default="distilroberta-base", help="Model name"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--n_splits", type=int, default=5, help="Number of folds for cross-validation"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--fp16", action="store_true", help="Use mixed precision training"
    )
    parser.add_argument(
        "--no_balance", action="store_true", help="Do not balance classes"
    )
    parser.add_argument(
        "--no_extract_top_repo",
        action="store_true",
        help="Do not extract top repository",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="text-classification",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--label_column", type=str, default="label", help="Column name for labels"
    )
    parser.add_argument(
        "--text_column", type=str, default="text", help="Column name for text"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load configuration if provided
    if args.config:
        config = MultiClassificationConfig.load(args.config)
    else:
        # Create configuration from arguments
        data_config = DataConfig(
            train_file=args.data_file,
            n_splits=args.n_splits,
            random_state=args.seed,
            label_column=args.label_column,
            text_column=args.text_column,
        )

        # Load data to determine number of labels
        if not args.data_file:
            raise ValueError("No data file provided")

        df = pd.read_csv(args.data_file)
        num_labels = len(df[args.label_column].unique())

        model_config = ModelConfig(
            model_name=args.model_name,
            num_labels=num_labels,
            max_length=args.max_length,
        )

        training_config = TrainingConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            fp16=args.fp16,
            report_to=["wandb"] if args.wandb else ["none"],
        )

        config = MultiClassificationConfig(
            model=model_config,
            training=training_config,
            data=data_config,
            balance_classes=not args.no_balance,
            extract_top_repo=not args.no_extract_top_repo,
        )

    # Create output directory
    os.makedirs(config.training.output_dir, exist_ok=True)

    # Save configuration
    config.save(config.training.output_dir)

    # Load data
    if not config.data.train_file:
        raise ValueError("No data file provided")

    df = pd.read_csv(config.data.train_file)

    # Initialize model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        config.model.model_name, config.model.num_labels, config.model.max_length
    )

    # Initialize data processor
    data_processor = MultiClassificationProcessor(tokenizer, config.model.max_length)

    # Balance classes if enabled
    if config.balance_classes:
        df = data_processor.balance_classes(df, config.data.label_column)

    # Extract top repository if enabled
    if config.extract_top_repo:
        main_data, top_repo_data = data_processor.extract_top_repo(
            df, config.data.label_column, "repo", 1
        )

        # Save top repository data
        top_repo_file = os.path.join(config.training.output_dir, "top_repo_data.csv")
        top_repo_data.to_csv(top_repo_file, index=False)
    else:
        main_data = df

    # Create label to index mapping
    labels = sorted(main_data[config.data.label_column].unique())
    cat2idx = {label: i for i, label in enumerate(labels)}
    idx2cat = {i: label for i, label in enumerate(labels)}

    # Save label mappings
    with open(os.path.join(config.training.output_dir, "label_mapping.txt"), "w") as f:
        for label, idx in cat2idx.items():
            f.write(f"{label}\t{idx}\n")

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(main_data[config.data.label_column]),
        y=main_data[config.data.label_column].values,
    )

    # Initialize training arguments
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        lr_scheduler_type=config.training.lr_scheduler_type,
        fp16=config.training.fp16,
        save_strategy=config.training.save_strategy,
        save_total_limit=config.training.save_total_limit,
        evaluation_strategy=config.training.evaluation_strategy,
        logging_dir=config.training.logging_dir,
        logging_steps=config.training.logging_steps,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to=config.training.report_to,
    )

    # Initialize Weights & Biases if enabled
    if args.wandb:
        import wandb

        wandb.init(project=args.wandb_project, config=config.to_dict())

    # Create compute metrics function
    compute_metrics = MultiClassificationTrainer.create_compute_metrics(
        config.training.output_dir, cat2idx, idx2cat, is_binary=False
    )

    # Initialize trainer
    trainer = MultiClassificationTrainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        output_dir=config.training.output_dir,
    )

    # Train with k-fold cross-validation
    fold_results = trainer.train_with_k_fold(
        data_processor,
        main_data,
        config.data.text_column,
        config.data.label_column,
        config.data.n_splits,
        cat2idx,
    )

    # Evaluate on top repository data if extracted
    if config.extract_top_repo and len(top_repo_data) > 0:
        print("Evaluating on top repository data...")

        # Create test dataset
        test_dataset = data_processor.create_dataset(
            top_repo_data, config.data.text_column, config.data.label_column, cat2idx
        )

        # Evaluate
        predictions = trainer.evaluate(test_dataset)

        # Save predictions with probabilities
        result_df = trainer.predict_with_probabilities(
            test_dataset, top_repo_data, idx2cat
        )

        # Save results
        result_df.to_csv(
            os.path.join(config.training.output_dir, "top_repo_predictions.csv"),
            index=False,
        )

    print(f"Training completed. Results saved to {config.training.output_dir}")


if __name__ == "__main__":
    main()
