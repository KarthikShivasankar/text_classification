#!/usr/bin/env python
"""Inference entry-point for technical debt classification.

Default path  : ONNX (CPU or GPU via CUDAExecutionProvider) — no PyTorch needed.
Torch path    : use ``--use_torch`` to force PyTorch-based inference (GPU recommended).
"""

import json
import os
from datetime import datetime

# Disable oneDNN warning before any TF import
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd

from tdsuite.utils.onnx_inference import OnnxInferenceEngine
from tdsuite.cli import get_inference_parser


def parse_args():
    return get_inference_parser().parse_args()


def _requested_device(args) -> "str | None":
    """Resolve the explicitly-requested device from --device / --gpu / --cpu.

    Returns ``"cpu"``, ``"cuda"``, or ``None`` (no explicit request → auto).
    The convenience flags ``--gpu`` / ``--cpu`` map to ``cuda`` / ``cpu``.
    If both ``--device`` and a convenience flag are given they must agree,
    otherwise a clear ``ValueError`` is raised.
    """
    device = getattr(args, "device", None)
    if device:
        device = device.lower()

    flag_device = None
    if getattr(args, "gpu", False):
        flag_device = "cuda"
    elif getattr(args, "cpu", False):
        flag_device = "cpu"

    if device and flag_device and device != flag_device:
        raise ValueError(
            f"Conflicting device selection: --device {device} versus "
            f"--{'gpu' if flag_device == 'cuda' else 'cpu'} "
            f"(implies {flag_device}). Pass only one, or make them agree."
        )

    return device or flag_device


def _resolve_device(args, use_torch: bool) -> str:
    """Return the device string ('cpu' or 'cuda') to use.

    Defaults to CPU. CUDA is auto-selected only when a capable GPU with
    > 6 GB free VRAM is available for the active backend. An explicit
    ``--device`` / ``--gpu`` / ``--cpu`` always takes precedence.
    """
    from tdsuite.utils.onnx_inference import auto_select_device

    is_torch_backend = use_torch or bool(
        getattr(args, "model_paths", None) or getattr(args, "model_names", None)
    )
    backend = "torch" if is_torch_backend else "onnx"
    return auto_select_device(_requested_device(args), backend=backend)


def _build_torch_engine(args, device: str):
    """Build an InferenceEngine or EnsembleInferenceEngine (requires torch)."""
    try:
        from tdsuite.utils.inference import InferenceEngine, EnsembleInferenceEngine
    except ImportError as exc:
        raise ImportError(
            "PyTorch-based inference requires torch and transformers.\n"
            "Install with: pip install 'tdsuite[gpu]'"
        ) from exc

    if args.model_paths or args.model_names:
        if args.weights:
            num_models = max(len(args.model_paths or []), len(args.model_names or []))
            if len(args.weights) != num_models:
                raise ValueError(
                    f"Number of weights ({len(args.weights)}) must match number of models ({num_models})"
                )
        return EnsembleInferenceEngine(
            model_paths=args.model_paths,
            model_names=args.model_names,
            max_length=args.max_length,
            device=device,
            weights=args.weights,
        )

    return InferenceEngine(
        model_path=args.model_path,
        model_name=args.model_name,
        max_length=args.max_length,
        device=device,
    )


def _build_onnx_engine(args, device: str) -> OnnxInferenceEngine:
    """Build an OnnxInferenceEngine from a local path or HF Hub model ID."""
    show = not args.disable_progress_bar

    # Explicit ONNX path supplied
    if args.onnx_path:
        return OnnxInferenceEngine(
            onnx_path=args.onnx_path,
            max_length=args.max_length,
            show_progress=show,
            device=device,
        )

    # Auto-resolve from model_name or model_path
    source = args.model_name or args.model_path
    if source:
        # If source is a local directory that contains model.onnx, use it directly
        local_onnx = os.path.join(source, "model.onnx") if os.path.isdir(source) else None
        if local_onnx and os.path.exists(local_onnx):
            return OnnxInferenceEngine(
                onnx_path=local_onnx,
                tokenizer_path=source,
                max_length=args.max_length,
                show_progress=show,
                device=device,
            )
        # Otherwise try HF Hub download (works for both HF model IDs and local paths
        # that happen to have model.onnx on the Hub)
        return OnnxInferenceEngine.from_pretrained(
            model_id=source,
            device=device,
            max_length=args.max_length,
            show_progress=show,
        )

    raise ValueError(
        "Provide --model_name, --model_path, or --onnx_path for ONNX inference."
    )


def _start_emissions_tracker(emissions_dir: str):
    """Start CodeCarbon emissions tracker; return tracker or None if unavailable."""
    try:
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(
            output_dir=emissions_dir,
            project_name="inference",
            output_file="inference_emissions.csv",
            allow_multiple_runs=True,
            log_level="error",
            save_to_api=False,
        )
        tracker.start()
        print("Emissions tracking started")
        return tracker
    except Exception:
        return None


def _stop_emissions_tracker(tracker, emissions_dir: str):
    if tracker is None:
        return
    try:
        emissions = tracker.stop()
        value = float(emissions) if emissions is not None else 0.0
        emissions_json = {
            "total_emissions": value,
            "unit": "kgCO2e",
            "timestamp": datetime.now().isoformat(),
        }
        path = os.path.join(emissions_dir, "inference_emissions.json")
        with open(path, "w") as fh:
            json.dump(emissions_json, fh, indent=2)
    except Exception as exc:
        print(f"Warning: emissions tracking error: {exc}")


def main():
    args = parse_args()

    use_torch = getattr(args, "use_torch", False)
    try:
        device = _resolve_device(args, use_torch)
    except ValueError as exc:
        import sys

        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"Device : {device}")
    print(f"Backend: {'torch' if use_torch else 'onnx'}")

    # Results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.model_path:
        base_dir = os.path.dirname(args.model_path)
    elif args.model_paths:
        base_dir = os.path.dirname(args.model_paths[0])
    else:
        base_dir = "outputs"

    results_dir = args.results_dir or os.path.join(base_dir, f"inference_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    emissions_dir = os.path.join(results_dir, "emissions")
    os.makedirs(emissions_dir, exist_ok=True)

    tracker = None
    if args.track_emissions:
        tracker = _start_emissions_tracker(emissions_dir)

    try:
        # Build inference engine
        if use_torch:
            engine = _build_torch_engine(args, device)
        elif args.model_paths or args.model_names:
            # Ensemble: fall back to torch for now (ONNX ensemble not yet supported)
            print("Ensemble mode detected — using PyTorch backend.")
            engine = _build_torch_engine(args, device)
        else:
            engine = _build_onnx_engine(args, device)

        if hasattr(engine, "show_progress"):
            engine.show_progress = not args.disable_progress_bar

        # Run inference
        if args.text is not None:
            result = engine.predict_single(args.text)
            print(json.dumps(result, indent=2))
        else:
            if args.output_file is None:
                input_filename = os.path.basename(args.input_file)
                args.output_file = os.path.join(results_dir, f"predictions_{input_filename}")

            print(f"Input  : {args.input_file}")
            print(f"Output : {args.output_file}")

            df = engine.predict_from_file(
                args.input_file,
                output_file=args.output_file,
                text_column=args.text_column,
                batch_size=args.batch_size,
            )
            print(f"Done. {len(df)} rows saved to {args.output_file}")

            if "label" in df.columns:
                from tdsuite.utils.metrics import compute_metrics

                metrics_dir = os.path.join(results_dir, "metrics")
                os.makedirs(metrics_dir, exist_ok=True)
                metrics = compute_metrics(df, output_dir=metrics_dir, save_plots=True)
                with open(os.path.join(metrics_dir, "metrics.json"), "w") as fh:
                    json.dump(metrics, fh, indent=2)
                print("\nMetrics:")
                print(json.dumps(metrics, indent=2))

    except Exception as exc:
        print(f"Error during inference: {exc}")
        raise

    finally:
        _stop_emissions_tracker(tracker, emissions_dir)


if __name__ == "__main__":
    main()
