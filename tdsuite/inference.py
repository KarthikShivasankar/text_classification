#!/usr/bin/env python
"""
Script for performing inference with trained technical debt classification models.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from tqdm import tqdm  # Import tqdm for progress bars

# Disable oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import torch
from codecarbon import EmissionsTracker  # Import EmissionsTracker

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
    parser.add_argument("--results_dir", type=str, help="Directory to store inference results and metrics")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the text column in the input file")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference (cuda, cpu)")
    parser.add_argument("--weights", type=float, nargs="+", help="Weights for ensemble models")
    parser.add_argument("--track_emissions", type=bool, default=True, help="Whether to track carbon emissions")
    parser.add_argument("--disable_progress_bar", action="store_true", help="Disable progress bar during inference")
    
    return parser.parse_args()


def main():
    """Main function for inference."""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine the base directory for results
    if args.model_path:
        base_dir = os.path.dirname(args.model_path)
    elif args.model_paths:
        base_dir = os.path.dirname(args.model_paths[0])
    else:
        base_dir = "outputs"
    
    # Create results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join(base_dir, f"inference_{timestamp}")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize emissions tracker
    tracker = None
    if args.track_emissions:
        tracker_output_dir = os.path.join(results_dir, "emissions")
        os.makedirs(tracker_output_dir, exist_ok=True)
        tracker = EmissionsTracker(
            output_dir=tracker_output_dir,
            project_name="inference",
            output_file="inference_emissions.csv"
        )
        tracker.start()
        print("📊 Emissions tracking started for inference")
    
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
    
    # Set progress bar display
    if 'predict_batch' in dir(engine):
        engine.show_progress = not args.disable_progress_bar
    
    # Perform inference
    if args.text is not None:
        # Single text inference
        result = engine.predict_single(args.text)
        print(json.dumps(result, indent=2))
    else:
        # Batch inference from file
        # If output_file is not specified, use a default name in the results directory
        if args.output_file is None:
            input_filename = os.path.basename(args.input_file)
            output_filename = f"predictions_{input_filename}"
            args.output_file = os.path.join(results_dir, output_filename)
        
        print(f"🚀 Starting inference on {args.input_file}")
        df = engine.predict_from_file(
            args.input_file,
            output_file=args.output_file,
            text_column=args.text_column,
            batch_size=args.batch_size,
        )
        print(f"✅ Inference completed! Results saved to {args.output_file}")
        
        # Compute and save metrics if ground truth labels are available
        if "label" in df.columns:
            from tdsuite.utils.metrics import compute_metrics
            print("📊 Computing metrics...")
            metrics_dir = os.path.join(results_dir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            metrics = compute_metrics(df, output_dir=metrics_dir, save_plots=True)
            
            # Save metrics to JSON file
            metrics_file = os.path.join(metrics_dir, "metrics.json")
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            
            print("\n📊 Metrics:")
            print(json.dumps(metrics, indent=2))
        
        # Print results if no output file was specified
        if args.output_file is None:
            print(df.to_string())
    
    # Stop emissions tracking
    if args.track_emissions and tracker:
        try:
            emissions = tracker.stop()
            
            if emissions is None:
                # Handle case where emissions data is None
                emissions_json = {
                    "total_emissions": 0.0,
                    "unit": "kgCO2e",
                    "timestamp": datetime.now().isoformat(),
                    "error": "No emissions data was recorded"
                }
                print("\n⚠️ Warning: No emissions data was recorded")
            else:
                try:
                    # Try to convert emissions to float
                    emissions_value = float(emissions)
                    emissions_json = {
                        "total_emissions": emissions_value,
                        "unit": "kgCO2e",
                        "timestamp": datetime.now().isoformat()
                    }
                except (TypeError, ValueError):
                    # Handle case where emissions can't be converted to float
                    emissions_json = {
                        "total_emissions": 0.0,
                        "unit": "kgCO2e",
                        "timestamp": datetime.now().isoformat(),
                        "error": "Emissions data couldn't be converted to float"
                    }
                    print("\n⚠️ Warning: Emissions data couldn't be converted to float")
            
            # Create emissions directory if it doesn't exist
            emissions_dir = os.path.join(results_dir, "emissions")
            os.makedirs(emissions_dir, exist_ok=True)
            
            # Save emissions as JSON for easier access
            emissions_json_path = os.path.join(emissions_dir, "inference_emissions.json")
            with open(emissions_json_path, "w") as f:
                json.dump(emissions_json, f, indent=2)
            
            # Safely extract emissions value for display
            emissions_value = emissions_json.get("total_emissions", 0.0)
            if not isinstance(emissions_value, (int, float)):
                emissions_value = 0.0
                
            print(f"\n🌱 Carbon emissions: {emissions_value:.6f} kgCO2e")
            print(f"🌱 Emissions data saved to {emissions_json_path}")
        
        except Exception as e:
            # Catch any unexpected errors during emissions tracking
            print(f"\n⚠️ Warning: Error in emissions tracking: {str(e)}")


if __name__ == "__main__":
    main() 