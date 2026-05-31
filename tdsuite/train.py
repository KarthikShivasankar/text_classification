#!/usr/bin/env python
"""
Script for training technical debt classification models.
"""

import os

# Disable oneDNN warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress warnings
import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import json  # noqa: E402

import torch  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

from tdsuite.cli import get_train_parser  # noqa: E402
from tdsuite.data.dataset import BinaryTDProcessor  # noqa: E402
from tdsuite.trainers import TDTrainer  # noqa: E402


def parse_args():
    """Parse command line arguments."""
    return get_train_parser().parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Check for GPU availability
    if not torch.cuda.is_available():
        print(
            "WARNING: No GPU detected. Training will be very slow on CPU "
            "and is not recommended.\n"
            "If you only need inference on CPU, export a pre-trained model "
            "to ONNX instead:\n"
            "  python scripts/export_onnx.py "
            "--model_name karths/binary_classification_train_TD --output model.onnx\n"
            "  tdsuite-inference --onnx_path model.onnx --input_file data.csv\n"
            "Proceeding with CPU training..."
        )
        device = torch.device("cpu")
    else:
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
        args.data_file, text_column=args.text_column, label_column=args.label_column
    )

    data = processor.prepare_binary_data(
        data,
        positive_category=args.positive_category,
        text_column=args.text_column,
        label_column=args.label_column,
        numeric_labels=args.numeric_labels,
    )

    # Split data into train and eval sets
    train_data, eval_data = train_test_split(
        data,
        test_size=0.1,  # Use 10% for evaluation
        random_state=args.seed,
        stratify=data["label_idx"],
    )

    # Create datasets
    train_dataset = processor.create_dataset(
        train_data, text_column=args.text_column, label_column="label_idx"
    )

    eval_dataset = processor.create_dataset(
        eval_data, text_column=args.text_column, label_column="label_idx"
    )

    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
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
        # Only load best model when not using cross-validation
        load_best_model_at_end=not args.cross_validation,
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
            train_dataset, eval_dataset=eval_dataset, is_binary=True
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
