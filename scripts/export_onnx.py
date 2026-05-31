#!/usr/bin/env python
"""Export a trained tdsuite transformer model to ONNX format for CPU inference.

ONNX models run on CPU without a GPU and are typically 2-4x faster than PyTorch
on CPU for inference workloads.

Usage:
    # From a local checkpoint
    python scripts/export_onnx.py --model_path outputs/binary --output model.onnx

    # From a Hugging Face model
    python scripts/export_onnx.py \\
        --model_name karths/binary_classification_train_TD --output model.onnx

    # Then run inference with the exported model
    tdsuite-inference --onnx_path model.onnx --model_path outputs/binary \\
        --input_file issue_texts.csv

Requirements:
    uv pip install onnx onnxruntime
    # or:
    pip install onnx onnxruntime
"""

import os


def _get_parser():
    try:
        from tdsuite.cli import get_export_onnx_parser

        return get_export_onnx_parser()
    except ImportError:
        import argparse

        p = argparse.ArgumentParser(description="Export a transformer model to ONNX")
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--model_path", help="Path to a local model checkpoint")
        g.add_argument("--model_name", help="Hugging Face model name")
        p.add_argument("--output", required=True, help="Output .onnx file path")
        p.add_argument("--max_length", type=int, default=512)
        p.add_argument(
            "--opset", type=int, default=14, help="ONNX opset version (default: 14)"
        )
        return p


def export_to_onnx(model_path, model_name, output_path, max_length, opset):
    """Export a HuggingFace sequence-classification model to a self-contained ONNX file.

    Uses the shared TorchDynamo-based exporter (transformers>=5 compatible),
    which produces a single portable ``model.onnx`` (no external data sidecar).

    Args:
        model_path: Local path to the model directory (or None).
        model_name: HuggingFace model name (or None).
        output_path: Destination .onnx file.
        max_length: Sequence length to use for the dummy input.
        opset: ONNX opset version.
    """
    from tdsuite.utils.onnx_inference import export_transformer_to_onnx

    source = model_path or model_name
    print(f"Loading model from: {source}")
    export_transformer_to_onnx(
        model_source=source,
        output_path=output_path,
        max_length=max_length,
        opset=opset,
    )

    onnx_dir = os.path.dirname(os.path.abspath(output_path))
    print(f"Tokenizer saved to {onnx_dir}/")
    print(
        f"\nRun CPU inference:\n"
        f"  tdsuite-inference --onnx_path {output_path} --input_file issue_texts.csv"
    )


def main():
    args = _get_parser().parse_args()
    export_to_onnx(
        model_path=args.model_path,
        model_name=args.model_name,
        output_path=args.output,
        max_length=args.max_length,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
