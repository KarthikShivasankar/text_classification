"""ONNX-based inference engine for CPU deployment without a GPU."""

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
            "  uv pip install onnx onnxruntime\n"
            "  # or: pip install onnx onnxruntime",
            file=sys.stderr,
        )
        sys.exit(1)


class OnnxInferenceEngine:
    """Inference engine that runs a tdsuite model exported to ONNX on CPU.

    Produces the same output format as InferenceEngine so it is a drop-in
    replacement for CPU environments without a GPU.

    Usage:
        engine = OnnxInferenceEngine(onnx_path="model.onnx")
        result = engine.predict_single("This code has serious design flaws")
        df = engine.predict_from_file("issue_texts.csv", output_file="preds.csv")
    """

    def __init__(
        self,
        onnx_path: str,
        tokenizer_path: Optional[str] = None,
        max_length: int = 512,
        show_progress: bool = True,
    ):
        """
        Args:
            onnx_path: Path to the .onnx model file.
            tokenizer_path: Directory containing the tokenizer files.
                Defaults to the directory of onnx_path.
            max_length: Maximum token sequence length.
            show_progress: Whether to show tqdm progress bars.
        """
        ort = _require_onnxruntime()

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.onnx_path = onnx_path
        self.max_length = max_length
        self.show_progress = show_progress

        # Tokenizer lives next to the .onnx file by default (saved there by export_onnx.py)
        tok_dir = tokenizer_path or os.path.dirname(os.path.abspath(onnx_path))
        self.tokenizer = self._load_tokenizer(tok_dir)

        # Create ONNX Runtime session — uses all available CPU threads by default
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = [inp.name for inp in self.session.get_inputs()]

    def _load_tokenizer(self, tokenizer_path: str):
        from transformers import AutoTokenizer

        if not os.path.isdir(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer directory not found: {tokenizer_path}\n"
                f"Re-run export_onnx.py — it saves the tokenizer alongside the .onnx file."
            )
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
        # Keep only the inputs the ONNX model actually expects
        return {k: enc[k] for k in self._input_names if k in enc}

    def _run_session(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        logits = self.session.run(None, inputs)[0]  # shape (batch, num_labels)
        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

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
        batches = range(0, len(texts), batch_size)
        num_batches = (len(texts) + batch_size - 1) // batch_size
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
            DataFrame with original columns plus 'predicted_class',
            'predicted_probability', and 'class_probabilities'.
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
