#!/usr/bin/env python
"""
Script for performing inference with trained technical debt classification models.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import torch

from tdsuite.utils import InferenceEngine, EnsembleInferenceEngine


def parse_args():
    """Parse command line arguments."""
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
    input_group.add_argument("--input_file", type=str, help="Path to a file with texts (CSV or JSON)")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, help="Path to save predictions")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column in the input file")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference (cuda, cpu)")
    parser.add_argument("--weights", type=float, nargs="+", help="Weights for ensemble models")
    
    return parser.parse_args()


def main():
    """Main function for inference."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if using ensemble
    if args.model_paths or args.model_names:
        # Create ensemble inference engine
        engine = EnsembleInferenceEngine(
            model_paths=args.model_paths or args.model_names,
            model_names=args.model_names,
            max_length=args.max_length,
            device=device,
            weights=args.weights,
        )
    else:
        # Create single model inference engine
        engine = InferenceEngine(
            model_path=args.model_path,
            model_name=args.model_name,
            max_length=args.max_length,
            device=device,
        )
    
    # Perform inference
    if args.text is not None:
        # Single text inference
        result = engine.predict_single(args.text)
        print(json.dumps(result, indent=2))
    else:
        # Batch inference from file
        df = engine.predict_from_file(
            args.input_file,
            output_file=args.output_file,
            text_column=args.text_column,
            batch_size=args.batch_size,
        )
        
        # Compute and save metrics if ground truth labels are available
        if "label" in df.columns:
            from tdsuite.utils.metrics import compute_metrics
            metrics_dir = os.path.dirname(args.output_file) if args.output_file else "outputs/metrics"
            metrics = compute_metrics(df, output_dir=metrics_dir, save_plots=True)
            print("\nMetrics:")
            print(json.dumps(metrics, indent=2))
        
        # Print results if no output file is specified
        if args.output_file is None:
            print(df.to_string())


if __name__ == "__main__":
    main() 