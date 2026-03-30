#!/usr/bin/env python
"""
Script for performing inference with trained technical debt classification models.
"""

import json
import os
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Disable oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import torch
from codecarbon import EmissionsTracker  # Import EmissionsTracker

from tdsuite.utils import InferenceEngine, EnsembleInferenceEngine, OnnxInferenceEngine
from tdsuite.cli import get_inference_parser


def parse_args():
    """Parse command line arguments."""
    return get_inference_parser().parse_args()


def main():
    """Main function for inference."""
    # Parse arguments
    args = parse_args()
    
    print("Starting inference with arguments:")
    print(f"  model_path: {args.model_path}")
    print(f"  model_name: {args.model_name}")
    print(f"  model_paths: {args.model_paths}")
    print(f"  model_names: {args.model_names}")
    print(f"  input_file: {args.input_file}")
    print(f"  device: {args.device}")
    print(f"  weights: {args.weights}")
    
    # Set device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    print(f"Results will be saved to: {results_dir}")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Create emissions directory
    emissions_dir = os.path.join(results_dir, "emissions")
    os.makedirs(emissions_dir, exist_ok=True)
    
    # Initialize emissions tracker
    emissions_tracker = EmissionsTracker(
        output_dir=emissions_dir,
        project_name="inference",
        output_file="inference_emissions.csv",
        allow_multiple_runs=True,
        log_level="error",
        save_to_api=False,
    )
    emissions_tracker.start()
    print("📊 Emissions tracking started for inference")
    
    try:
        # ONNX path: CPU inference, no GPU needed
        if args.onnx_path:
            print(f"Using ONNX inference engine (CPU): {args.onnx_path}")
            engine = OnnxInferenceEngine(
                onnx_path=args.onnx_path,
                max_length=args.max_length,
                show_progress=not args.disable_progress_bar,
            )
        # Check if using ensemble
        elif args.model_paths or args.model_names:
            print("Creating ensemble inference engine")
            # Validate weights if provided
            if args.weights:
                num_models = max(len(args.model_paths or []), len(args.model_names or []))
                if len(args.weights) != num_models:
                    raise ValueError(f"Number of weights ({len(args.weights)}) must match number of models ({num_models})")

            # Create ensemble inference engine
            engine = EnsembleInferenceEngine(
                model_paths=args.model_paths,
                model_names=args.model_names,
                max_length=args.max_length,
                device=device,
                weights=args.weights,
            )
        else:
            print("Creating single model inference engine")
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
            print("Performing single text inference")
            result = engine.predict_single(args.text)
            print(json.dumps(result, indent=2))
        else:
            print(f"Starting batch inference on {args.input_file}")
            # If output_file is not specified, use a default name in the results directory
            if args.output_file is None:
                input_filename = os.path.basename(args.input_file)
                output_filename = f"predictions_{input_filename}"
                args.output_file = os.path.join(results_dir, output_filename)
            
            print(f"Output will be saved to: {args.output_file}")
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
    
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise
    
    finally:
        # Stop emissions tracking
        if args.track_emissions and emissions_tracker:
            try:
                emissions = emissions_tracker.stop()
                
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
                
                # Save emissions as JSON for easier access
                emissions_json_path = os.path.join(emissions_dir, "inference_emissions.json")
                with open(emissions_json_path, "w") as f:
                    json.dump(emissions_json, f, indent=2)
            
            except Exception as e:
                # Catch any unexpected errors during emissions tracking
                print(f"\n⚠️ Warning: Error in emissions tracking: {str(e)}")


if __name__ == "__main__":
    main() 