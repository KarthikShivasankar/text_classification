"""Utility functions for technical debt classification."""

from .data_utils import load_dataset, preprocess_text
from .logging import setup_logging
from .metrics import compute_metrics
from .onnx_inference import OnnxEnsembleInferenceEngine, OnnxInferenceEngine


def __getattr__(name):
    """Lazy-load torch-dependent engines only when explicitly requested."""
    if name in ("InferenceEngine", "EnsembleInferenceEngine"):
        try:
            from .inference import (  # noqa: F401
                EnsembleInferenceEngine,
                InferenceEngine,
            )
        except ImportError as exc:
            raise ImportError(
                "The PyTorch inference engines require torch and transformers. "
                "Install them with: pip install 'tdsuite[gpu]'"
            ) from exc

        globals()["InferenceEngine"] = InferenceEngine
        globals()["EnsembleInferenceEngine"] = EnsembleInferenceEngine
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "compute_metrics",
    "setup_logging",
    "load_dataset",
    "preprocess_text",
    "OnnxInferenceEngine",
    "OnnxEnsembleInferenceEngine",
    "InferenceEngine",
    "EnsembleInferenceEngine",
]
