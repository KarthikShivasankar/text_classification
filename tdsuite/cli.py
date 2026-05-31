"""Centralized CLI argument parsers for tdsuite."""

import argparse


def get_train_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the train command."""
    parser = argparse.ArgumentParser(
        description="Train a technical debt classification model"
    )

    # Data arguments
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to data file or Hugging Face dataset name",
    )
    parser.add_argument(
        "--text_column", type=str, default="text", help="Name of the text column"
    )
    parser.add_argument(
        "--label_column", type=str, default="label", help="Name of the label column"
    )
    parser.add_argument(
        "--is_huggingface_dataset",
        action="store_true",
        help="Whether the data is a Hugging Face dataset",
    )
    parser.add_argument(
        "--numeric_labels",
        action="store_true",
        help="Whether the labels are already numeric (0 or 1)",
    )
    parser.add_argument(
        "--positive_category",
        type=str,
        help="Positive category for binary classification",
    )

    # Model arguments
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to use"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the model and outputs",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--warmup_steps", type=int, default=500, help="Number of warmup steps"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )

    # Cross-validation arguments
    parser.add_argument(
        "--cross_validation",
        action="store_true",
        help="Whether to use cross-validation",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of splits for cross-validation",
    )

    # Additional arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, or None for auto-detection)",
    )

    return parser


def get_inference_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the inference command."""
    parser = argparse.ArgumentParser(
        description="Inference for technical debt classification"
    )

    # Model arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model_path", type=str, help="Path to a local model")
    model_group.add_argument(
        "--model_name", type=str, help="Name of a model on Hugging Face"
    )
    model_group.add_argument(
        "--model_paths", type=str, nargs="+", help="Paths to multiple local models"
    )
    model_group.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        help="Names of multiple models on Hugging Face",
    )

    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Text to classify")
    input_group.add_argument(
        "--input_file", type=str, help="Path to a file with texts (CSV or JSON)"
    )

    # Output arguments
    parser.add_argument("--output_file", type=str, help="Path to save predictions")
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Directory to store inference results and metrics",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text column in the input file",
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help=(
            "Device to use for inference. Default: auto — CPU unless a CUDA GPU "
            "with > 6 GB free VRAM is available (and, for the ONNX backend, only "
            "when onnxruntime-gpu / CUDAExecutionProvider is installed). "
            "Pass 'cuda' to force GPU or 'cpu' to force CPU; an explicit value "
            "always overrides the auto-detection. See also --gpu / --cpu."
        ),
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Convenience flag equivalent to --device cuda: force GPU inference "
            "even when free VRAM is below the auto-detection threshold. "
            "ONNX GPU requires onnxruntime-gpu (pip install 'tdsuite[gpu]')."
        ),
    )
    device_group.add_argument(
        "--cpu",
        action="store_true",
        help=(
            "Convenience flag equivalent to --device cpu: force CPU inference "
            "even when a capable CUDA GPU is available."
        ),
    )
    parser.add_argument(
        "--weights", type=float, nargs="+", help="Weights for ensemble models"
    )
    parser.add_argument(
        "--track_emissions",
        type=bool,
        default=True,
        help="Whether to track carbon emissions",
    )
    parser.add_argument(
        "--disable_progress_bar",
        action="store_true",
        help="Disable progress bar during inference",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        help=(
            "Path to a local .onnx model file. "
            "If omitted, model.onnx is auto-downloaded from Hugging Face Hub when "
            "--model_name is supplied. "
            "Export your own ONNX model with: python scripts/export_onnx.py"
        ),
    )
    parser.add_argument(
        "--use_torch",
        action="store_true",
        help=(
            "Force PyTorch-based inference instead of the default ONNX backend. "
            "Requires torch to be installed (pip install 'tdsuite[gpu]'). "
            "Use this for ensemble inference or when ONNX is not available."
        ),
    )

    return parser


def get_export_onnx_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the export-onnx script."""
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained transformer model to ONNX format for CPU inference. "
            "Requires: uv pip install onnx onnxruntime"
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--model_path",
        type=str,
        help="Path to a local model checkpoint directory",
    )
    source_group.add_argument(
        "--model_name",
        type=str,
        help="Hugging Face model name (e.g. karths/binary_classification_train_TD)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination .onnx file path (e.g. model.onnx)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Sequence length used for the export dummy input (default: 512)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    return parser


def get_split_data_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the split-data command."""
    parser = argparse.ArgumentParser(
        description="Split and save data for technical debt classification."
    )

    # Required arguments
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the data file or Hugging Face dataset name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the split data",
    )

    # Optional arguments
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    parser.add_argument(
        "--repo_column",
        type=str,
        default=None,
        help="Name of the column containing the repository information",
    )
    parser.add_argument(
        "--is_huggingface_dataset",
        action="store_true",
        help="Whether the data file is a Hugging Face dataset",
    )
    parser.add_argument(
        "--is_numeric_labels",
        action="store_true",
        help="Whether the labels are numeric (0/1) or categorical",
    )

    return parser


def get_fetch_issues_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the fetch-github-issues script."""
    parser = argparse.ArgumentParser(
        description=(
            "Fetch GitHub issues from a public repo into CSV. "
            "By default fetches the latest 100 issues. "
            "Use --all to paginate through every issue."
        )
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Repository in owner/repo format, e.g. 'torvalds/linux'",
    )
    parser.add_argument(
        "--output",
        default="issues.csv",
        help="Output CSV file path (default: issues.csv)",
    )
    parser.add_argument(
        "--state",
        choices=["open", "closed", "all"],
        default="all",
        help="Issue state to fetch (default: all)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "GitHub personal access token. Required when the unauthenticated "
            "rate limit (60 req/h) is exceeded. "
            "Create one at https://github.com/settings/tokens"
        ),
    )

    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--all",
        action="store_true",
        dest="fetch_all",
        help="Fetch every issue (paginates through all pages); slow for large repos.",
    )
    limit_group.add_argument(
        "--limit",
        type=int,
        default=100,
        metavar="N",
        help="Maximum number of issues to fetch, sorted newest-first (default: 100).",
    )
    return parser


def get_extract_issues_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the extract-issue-bodies script."""
    parser = argparse.ArgumentParser(
        description="Extract issue body text from a GitHub issues CSV"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV (output of fetch_github_issues.py)",
    )
    parser.add_argument(
        "--output",
        default="issue_texts.csv",
        help="Output CSV path (default: issue_texts.csv)",
    )
    parser.add_argument(
        "--body-column",
        default="body",
        dest="body_column",
        help="Name of the column containing issue body text (default: body)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        dest="min_length",
        help="Minimum body character length to keep (default: 20)",
    )
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        dest="keep_metadata",
        help="If set, also keep 'number' and 'title' columns in output",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        dest="drop_duplicates",
        help="Drop duplicate body texts",
    )
    return parser
