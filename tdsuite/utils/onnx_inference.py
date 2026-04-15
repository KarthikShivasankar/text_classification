"""ONNX-based inference engine — CPU (default) and GPU via CUDAExecutionProvider."""

import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


def _require_onnxruntime():
    try:
        import onnxruntime as ort

        return ort
    except ImportError:
        print(
            "Error: 'onnxruntime' is not installed.\n"
            "Install it with:\n"
            "  pip install onnxruntime          # CPU\n"
            "  pip install onnxruntime-gpu      # GPU (CUDA)\n"
            "  # or via tdsuite extras:\n"
            "  pip install 'tdsuite[onnx]'      # CPU\n"
            "  pip install 'tdsuite[gpu]'       # GPU",
            file=sys.stderr,
        )
        sys.exit(1)


class OnnxInferenceEngine:
    """Inference engine that runs a tdsuite model exported to ONNX.

    Supports CPU (default) and GPU via CUDAExecutionProvider (requires
    onnxruntime-gpu).  This is the recommended inference path — no PyTorch
    required.

    Load from a local .onnx file:
        engine = OnnxInferenceEngine("model.onnx")

    Auto-download from Hugging Face Hub:
        engine = OnnxInferenceEngine.from_pretrained("karths/binary_classification_train_TD")
        engine = OnnxInferenceEngine.from_pretrained("karths/binary_classification_train_TD", device="cuda")
    """

    def __init__(
        self,
        onnx_path: str,
        tokenizer_path: Optional[str] = None,
        max_length: int = 512,
        show_progress: bool = True,
        device: str = "cpu",
    ):
        """
        Args:
            onnx_path: Path to the .onnx model file.
            tokenizer_path: Local directory or Hugging Face model ID for the
                tokenizer.  Defaults to the directory containing onnx_path.
            max_length: Maximum token sequence length.
            show_progress: Whether to show tqdm progress bars.
            device: ``"cpu"`` (default) or ``"cuda"`` for GPU inference.
                GPU requires ``onnxruntime-gpu`` to be installed.
        """
        ort = _require_onnxruntime()

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.onnx_path = onnx_path
        self.max_length = max_length
        self.show_progress = show_progress
        self.device = device.lower()

        tok_source = tokenizer_path or os.path.dirname(os.path.abspath(onnx_path))
        self.tokenizer = self._load_tokenizer(tok_source)

        # Build provider list based on device
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=providers,
        )

        # Warn if CUDA was requested but fell back to CPU
        active = self.session.get_providers()
        if self.device == "cuda" and "CUDAExecutionProvider" not in active:
            print(
                "Warning: CUDAExecutionProvider not available — running on CPU.\n"
                "Install onnxruntime-gpu for GPU support: pip install onnxruntime-gpu",
                file=sys.stderr,
            )

        self._input_names = [inp.name for inp in self.session.get_inputs()]

    # ------------------------------------------------------------------
    # Class-method constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str = "cpu",
        max_length: int = 512,
        show_progress: bool = True,
        token: Optional[str] = None,
    ) -> "OnnxInferenceEngine":
        """Download ``model.onnx`` from a Hugging Face Hub repository and load it.

        If ``model.onnx`` is not present in the Hub repo, the method falls back
        to exporting the model from its safetensors/PyTorch weights using
        ``torch.onnx.export`` (requires ``torch`` and ``onnx`` to be installed).

        Args:
            model_id: Hugging Face model ID, e.g.
                ``"karths/binary_classification_train_TD"``.
            device: ``"cpu"`` (default) or ``"cuda"`` for GPU inference.
            max_length: Maximum token sequence length.
            show_progress: Whether to show tqdm progress bars.
            token: Optional Hugging Face API token for private repos.

        Returns:
            Loaded :class:`OnnxInferenceEngine` instance.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print(
                "Error: 'huggingface_hub' is not installed.\n"
                "Install with: pip install huggingface-hub",
                file=sys.stderr,
            )
            sys.exit(1)

        # Try to download pre-exported model.onnx from the Hub
        onnx_path = None
        try:
            print(f"Downloading model.onnx from {model_id} …")
            onnx_path = hf_hub_download(
                repo_id=model_id,
                filename="model.onnx",
                token=token,
            )
        except Exception as hub_err:
            print(
                f"model.onnx not found in {model_id} ({hub_err}). "
                "Falling back to torch.onnx.export …",
                file=sys.stderr,
            )

        if onnx_path is None:
            onnx_path = cls._export_to_onnx(
                model_id=model_id,
                max_length=max_length,
                token=token,
            )

        return cls(
            onnx_path=onnx_path,
            tokenizer_path=model_id,  # AutoTokenizer handles HF Hub download
            max_length=max_length,
            show_progress=show_progress,
            device=device,
        )

    @staticmethod
    def _export_to_onnx(
        model_id: str,
        max_length: int = 512,
        token: Optional[str] = None,
    ) -> str:
        """Export a HF model to ONNX using ``torch.onnx.export``.

        The resulting file is saved under the HF Hub cache so it can be reused
        across calls.  Requires ``torch`` and ``onnx`` to be installed.

        Args:
            model_id: Hugging Face model ID.
            max_length: Maximum token sequence length (used for dummy input).
            token: Optional Hugging Face API token.

        Returns:
            Path to the exported ``.onnx`` file.
        """
        try:
            import torch
        except ImportError:
            print(
                "Error: exporting a model to ONNX requires 'torch'.\n"
                "Install with:  pip install torch\n"
                "Or use a model that already has model.onnx on the Hub.",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            import onnx  # noqa: F401 — presence check only
        except ImportError:
            print(
                "Error: exporting a model to ONNX requires the 'onnx' package.\n"
                "Install with:  pip install onnx\n"
                "Or: pip install 'tdsuite[onnx]'",
                file=sys.stderr,
            )
            sys.exit(1)

        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import tempfile, pathlib

        print(f"Exporting {model_id} → ONNX (this may take a moment) …")

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, token=token
        )
        model.eval()

        dummy = tokenizer(
            "dummy input text",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=min(max_length, 128),
        )
        input_ids = dummy["input_ids"]
        attention_mask = dummy["attention_mask"]

        # Determine which inputs the model actually expects
        try:
            sig = model.forward.__code__.co_varnames
            has_token_type = "token_type_ids" in sig and "token_type_ids" in dummy
        except Exception:
            has_token_type = False

        if has_token_type and "token_type_ids" in dummy:
            model_inputs = (input_ids, attention_mask, dummy["token_type_ids"])
            input_names = ["input_ids", "attention_mask", "token_type_ids"]
            dynamic_axes = {
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            }
        else:
            model_inputs = (input_ids, attention_mask)
            input_names = ["input_ids", "attention_mask"]
            dynamic_axes = {
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            }

        # Cache alongside HF Hub snapshots so we don't re-export every time
        try:
            from huggingface_hub import snapshot_download

            cache_dir = pathlib.Path(
                snapshot_download(
                    model_id,
                    token=token,
                    ignore_patterns=["*.safetensors", "*.bin", "flax_model*"],
                )
            )
        except Exception:
            cache_dir = pathlib.Path(tempfile.mkdtemp(prefix="tdsuite_onnx_"))

        onnx_path = str(cache_dir / "model.onnx")

        with torch.no_grad():
            torch.onnx.export(
                model,
                model_inputs,
                onnx_path,
                opset_version=17,
                input_names=input_names,
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
            )

        print(f"ONNX model saved to {onnx_path}")
        return onnx_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_tokenizer(self, tokenizer_path: str):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path)

    def _tokenize(self, texts: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        return {k: enc[k] for k in self._input_names if k in enc}

    def _run_session(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        logits = self.session.run(None, inputs)[0]  # (batch, num_labels)
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Public API (drop-in for InferenceEngine)
    # ------------------------------------------------------------------

    def predict_single(self, text: str) -> Dict[str, Union[str, float, List[float]]]:
        """Predict the class for a single text string."""
        inputs = self._tokenize(text)
        probs = self._run_session(inputs)[0]
        predicted_class = int(np.argmax(probs))
        return {
            "text": text,
            "predicted_class": predicted_class,
            "predicted_probability": float(probs[predicted_class]),
            "class_probabilities": probs.tolist(),
        }

    def predict_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Dict[str, Union[str, float, List[float]]]]:
        """Predict classes for a list of texts."""
        predictions = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        batches = range(0, len(texts), batch_size)
        if self.show_progress:
            batches = tqdm(batches, total=num_batches, desc="ONNX inference", unit="batch")

        for i in batches:
            batch_texts = texts[i : i + batch_size]
            inputs = self._tokenize(batch_texts)
            probs = self._run_session(inputs)
            for text, prob_row in zip(batch_texts, probs):
                predicted_class = int(np.argmax(prob_row))
                predictions.append(
                    {
                        "text": text,
                        "predicted_class": predicted_class,
                        "predicted_probability": float(prob_row[predicted_class]),
                        "class_probabilities": prob_row.tolist(),
                    }
                )
        return predictions

    def predict_from_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        text_column: str = "text",
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """Run inference on every row of a CSV/JSON file.

        Args:
            input_file: Path to input CSV or JSON file.
            output_file: Path to save predictions CSV (optional).
            text_column: Name of the text column.
            batch_size: Batch size.

        Returns:
            DataFrame with original columns plus ``predicted_class``,
            ``predicted_probability``, and ``class_probabilities``.
        """
        ext = os.path.splitext(input_file)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(input_file)
        elif ext in (".json", ".jsonl"):
            df = pd.read_json(input_file, lines=ext == ".jsonl")
        else:
            raise ValueError(f"Unsupported file format: {input_file}")

        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found. Available: {list(df.columns)}"
            )

        texts = df[text_column].fillna("").tolist()
        predictions = self.predict_batch(texts, batch_size=batch_size)

        df["predicted_class"] = [p["predicted_class"] for p in predictions]
        df["predicted_probability"] = [p["predicted_probability"] for p in predictions]
        df["class_probabilities"] = [str(p["class_probabilities"]) for p in predictions]

        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            df.to_csv(output_file, index=False)

        return df
