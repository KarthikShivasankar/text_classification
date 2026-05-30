"""Utility functions for technical debt classification."""

from .metrics import compute_metrics
from .logging import setup_logging
from .data_utils import load_dataset, preprocess_text
from .onnx_inference import OnnxEnsembleInferenceEngine, OnnxInferenceEngine


def __getattr__(name):
    """Lazy-load torch-dependent engines only when explicitly requested."""
    if name in ("InferenceEngine", "EnsembleInferenceEngine"):
        from .inference import InferenceEngine, EnsembleInferenceEngine  # noqa: F401
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