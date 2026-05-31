"""ONNX-based inference engine — CPU (default) and GPU via CUDAExecutionProvider.

Device policy: inference defaults to CPU. CUDA is selected automatically only
when the ONNX CUDAExecutionProvider is available *and* a GPU exposes more than
``GPU_VRAM_THRESHOLD_GB`` of free VRAM. An explicit ``device`` always wins.
"""

import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Default opset for the TorchDynamo-based exporter (transformers>=5 compatible).
ONNX_OPSET = 18

# CUDA is only auto-selected when a GPU exposes more free VRAM than this (GiB).
GPU_VRAM_THRESHOLD_GB = 6.0


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


def _max_free_vram_gb() -> float:
    """Return the free VRAM (GiB) of the most-free CUDA GPU, or 0.0 if none.

    Tries torch first (no CUDA context is created by ``mem_get_info`` beyond a
    lightweight device query), then falls back to parsing ``nvidia-smi``.
    """
    try:
        import torch

        if torch.cuda.is_available():
            best = 0.0
            for i in range(torch.cuda.device_count()):
                free, _total = torch.cuda.mem_get_info(i)
                best = max(best, free / (1024**3))
            return best
    except Exception:
        pass

    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            vals = []
            for line in out.stdout.splitlines():
                line = line.strip()
                if line:
                    try:
                        vals.append(float(line))
                    except ValueError:
                        continue
            if vals:
                return max(vals) / 1024.0  # MiB -> GiB
    except Exception:
        pass

    return 0.0


def auto_select_device(
    requested: Optional[str] = None,
    backend: str = "onnx",
    vram_threshold_gb: float = GPU_VRAM_THRESHOLD_GB,
) -> str:
    """Resolve the inference device string ('cpu' or 'cuda').

    Args:
        requested: Explicit device ('cpu'/'cuda'). When given it is honored
            verbatim. When ``None``/empty the device is auto-detected.
        backend: ``"onnx"`` checks for the ONNX CUDAExecutionProvider;
            ``"torch"`` checks ``torch.cuda.is_available()``.
        vram_threshold_gb: CUDA is only chosen when free VRAM exceeds this.

    Returns:
        ``"cuda"`` only when a capable GPU with > ``vram_threshold_gb`` free
        VRAM is available for the given backend; otherwise ``"cpu"``.
    """
    if requested:
        return requested.lower()

    if backend == "torch":
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    else:
        try:
            import onnxruntime as ort

            if "CUDAExecutionProvider" not in ort.get_available_providers():
                return "cpu"
        except Exception:
            return "cpu"

    return "cuda" if _max_free_vram_gb() > vram_threshold_gb else "cpu"


def export_transformer_to_onnx(
    model_source: str,
    output_path: str,
    max_length: int = 512,
    token: Optional[str] = None,
    opset: int = ONNX_OPSET,
):
    """Export a HF sequence-classification model to a single self-contained ONNX file.

    Uses the TorchDynamo-based exporter (``torch.onnx.export(dynamo=True)``),
    which is compatible with ``transformers>=5`` where the legacy TorchScript
    exporter fails on the new attention-mask code path. The exporter writes
    weights to an external ``.onnx.data`` sidecar; this function consolidates
    everything back into one ``model.onnx`` (no sidecar) so the file is portable
    and safe to upload to the Hugging Face Hub.

    Requires ``torch``, ``onnx`` and ``onnxscript``.

    Args:
        model_source: Local model directory or HF model ID.
        output_path: Destination ``.onnx`` file path.
        max_length: Sequence length for the dummy tracing input.
        token: Optional HF token for private/gated repos.
        opset: ONNX opset version.

    Returns:
        Tuple ``(output_path, tokenizer)``.
    """
    import shutil
    import tempfile

    try:
        import torch
    except ImportError:
        print(
            "Error: exporting a model to ONNX requires 'torch'.\n"
            "Install with: pip install torch\n"
            "Or use a model that already has a working model.onnx on the Hub.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import onnx
    except ImportError:
        print(
            "Error: exporting a model to ONNX requires the 'onnx' package.\n"
            "Install with: pip install onnx  (or: pip install 'tdsuite[onnx]')",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import onnxscript  # noqa: F401 — required by the dynamo exporter
    except ImportError:
        print(
            "Error: the ONNX dynamo exporter requires 'onnxscript'.\n"
            "Install with: pip install onnxscript  (or: pip install 'tdsuite[onnx]')",
            file=sys.stderr,
        )
        sys.exit(1)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # The dynamo exporter requires a recent opset; clamp older requests up.
    if opset < ONNX_OPSET:
        opset = ONNX_OPSET

    print(f"Exporting {model_source} -> ONNX (dynamo, opset {opset}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_source, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source, token=token
    )
    model.eval()

    dummy = tokenizer(
        "example technical debt text for export",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=min(max_length, 128),
    )
    inputs = (dummy["input_ids"], dummy["attention_mask"])
    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"},
    }

    tmp_dir = tempfile.mkdtemp(prefix="tdsuite_onnx_")
    tmp_onnx = os.path.join(tmp_dir, "model.onnx")
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                inputs,
                tmp_onnx,
                input_names=input_names,
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                dynamo=True,
            )

        # Consolidate external weight data into a single self-contained file.
        loaded = onnx.load(tmp_onnx)  # pulls .onnx.data into memory
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        onnx.save_model(loaded, output_path, save_as_external_data=False)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    tokenizer.save_pretrained(os.path.dirname(os.path.abspath(output_path)))

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported self-contained ONNX ({size_mb:.1f} MB) -> {output_path}")
    return output_path, tokenizer


class OnnxInferenceEngine:
    """Inference engine that runs a tdsuite model exported to ONNX.

    Supports CPU (default) and GPU via CUDAExecutionProvider (requires
    onnxruntime-gpu).  This is the recommended inference path — no PyTorch
    required.

    Load from a local .onnx file:
        engine = OnnxInferenceEngine("model.onnx")

    Auto-download from Hugging Face Hub:
        model_id = "karths/binary_classification_train_TD"
        engine = OnnxInferenceEngine.from_pretrained(model_id)
        engine = OnnxInferenceEngine.from_pretrained(model_id, device="cuda")
    """

    def __init__(
        self,
        onnx_path: str,
        tokenizer_path: Optional[str] = None,
        max_length: int = 512,
        show_progress: bool = True,
        device: Optional[str] = None,
    ):
        """
        Args:
            onnx_path: Path to the .onnx model file.
            tokenizer_path: Local directory or Hugging Face model ID for the
                tokenizer.  Defaults to the directory containing onnx_path.
            max_length: Maximum token sequence length.
            show_progress: Whether to show tqdm progress bars.
            device: ``"cpu"`` or ``"cuda"``. When ``None`` (default) the device
                is auto-detected: CPU unless a CUDA GPU with > 6 GB free VRAM is
                available. GPU requires ``onnxruntime-gpu`` to be installed.
        """
        ort = _require_onnxruntime()

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.onnx_path = onnx_path
        self.max_length = max_length
        self.show_progress = show_progress
        self.device = auto_select_device(device, backend="onnx")

        tok_source = tokenizer_path or os.path.dirname(os.path.abspath(onnx_path))
        self.tokenizer = self._load_tokenizer(tok_source)

        # Build provider list based on device
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Silence benign "could not constant fold CastLike" warnings emitted by
        # dynamo-exported graphs (severity: 3 = error and above only).
        sess_opts.log_severity_level = 3

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
        device: Optional[str] = None,
        max_length: int = 512,
        show_progress: bool = True,
        token: Optional[str] = None,
    ) -> "OnnxInferenceEngine":
        """Download ``model.onnx`` from a Hugging Face Hub repository and load it.

        If ``model.onnx`` is absent — or present but fails to load (e.g. it
        references a missing external ``.onnx.data`` sidecar) — this method
        falls back to exporting the model from its safetensors/PyTorch weights
        (requires ``torch``, ``onnx`` and ``onnxscript``).

        Args:
            model_id: Hugging Face model ID, e.g.
                ``"karths/binary_classification_train_TD"``.
            device: ``"cpu"`` or ``"cuda"``. When ``None`` (default) the device
                is auto-detected (CPU unless a CUDA GPU with > 6 GB free VRAM).
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
            print(f"Downloading model.onnx from {model_id} ...")
            onnx_path = hf_hub_download(
                repo_id=model_id,
                filename="model.onnx",
                token=token,
            )
        except Exception as hub_err:
            print(
                f"model.onnx not found in {model_id} ({hub_err}). "
                "Falling back to ONNX export ...",
                file=sys.stderr,
            )

        # Try loading the downloaded file; if it is broken (missing external
        # data, etc.), re-export a self-contained model instead.
        if onnx_path is not None:
            try:
                return cls(
                    onnx_path=onnx_path,
                    tokenizer_path=model_id,
                    max_length=max_length,
                    show_progress=show_progress,
                    device=device,
                )
            except Exception as load_err:
                print(
                    f"Downloaded model.onnx from {model_id} failed to load "
                    f"({load_err}). Re-exporting a self-contained ONNX ...",
                    file=sys.stderr,
                )
                onnx_path = None

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
        """Export a HF model to a single self-contained ONNX file.

        Delegates to :func:`export_transformer_to_onnx` (TorchDynamo exporter,
        transformers>=5 compatible). The result is cached in a temp directory so
        repeated calls within a session reuse it. Requires ``torch``, ``onnx``
        and ``onnxscript``.

        Args:
            model_id: Hugging Face model ID.
            max_length: Maximum token sequence length (used for dummy input).
            token: Optional Hugging Face API token.

        Returns:
            Path to the exported ``.onnx`` file.
        """
        import hashlib
        import tempfile

        # Deterministic cache location per model id so a session reuses it.
        digest = hashlib.md5(model_id.encode("utf-8")).hexdigest()[:12]
        cache_dir = os.path.join(tempfile.gettempdir(), f"tdsuite_onnx_{digest}")
        os.makedirs(cache_dir, exist_ok=True)
        onnx_path = os.path.join(cache_dir, "model.onnx")

        if os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0:
            print(f"Reusing cached ONNX export: {onnx_path}")
            return onnx_path

        export_transformer_to_onnx(
            model_source=model_id,
            output_path=onnx_path,
            max_length=max_length,
            token=token,
        )
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
            batches = tqdm(
                batches, total=num_batches, desc="ONNX inference", unit="batch"
            )

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


def _build_onnx_member(
    source: str,
    device: Optional[str] = None,
    max_length: int = 512,
    show_progress: bool = True,
    token: Optional[str] = None,
) -> "OnnxInferenceEngine":
    """Build a single :class:`OnnxInferenceEngine` from a path or HF model ID.

    Mirrors the resolution logic used for single-model ONNX inference: a local
    directory containing ``model.onnx`` is used directly, otherwise the source
    is treated as a Hugging Face model ID (or local path with ``model.onnx`` on
    the Hub) and resolved via :meth:`OnnxInferenceEngine.from_pretrained`.
    """
    if source.lower().endswith(".onnx") and os.path.exists(source):
        return OnnxInferenceEngine(
            onnx_path=source,
            max_length=max_length,
            show_progress=show_progress,
            device=device,
        )

    local_onnx = os.path.join(source, "model.onnx") if os.path.isdir(source) else None
    if local_onnx and os.path.exists(local_onnx):
        return OnnxInferenceEngine(
            onnx_path=local_onnx,
            tokenizer_path=source,
            max_length=max_length,
            show_progress=show_progress,
            device=device,
        )

    return OnnxInferenceEngine.from_pretrained(
        model_id=source,
        device=device,
        max_length=max_length,
        show_progress=show_progress,
        token=token,
    )


class OnnxEnsembleInferenceEngine:
    """Ensemble inference over several ONNX models — no PyTorch required.

    Holds a list of :class:`OnnxInferenceEngine` members (one per model) and
    combines their per-class probabilities with weighted averaging, using the
    same semantics as the PyTorch ``EnsembleInferenceEngine`` (weights are
    normalised to sum to 1, the weighted mean of the softmax probabilities is
    taken, and the predicted label is the argmax).

    Supports CPU (default) and GPU via ``CUDAExecutionProvider`` (requires
    ``onnxruntime-gpu``). Each member is built with
    :meth:`OnnxInferenceEngine.from_pretrained` (for HF model IDs) or directly
    from a local ``model.onnx`` / model directory.

    Example::

        engine = OnnxEnsembleInferenceEngine(
            model_names=["karths/model_a", "karths/model_b"],
            weights=[0.6, 0.4],
        )
        engine.predict_single("some text about technical debt")
    """

    def __init__(
        self,
        model_paths: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
        max_length: int = 512,
        show_progress: bool = True,
        device: Optional[str] = None,
        weights: Optional[List[float]] = None,
        token: Optional[str] = None,
    ):
        """
        Args:
            model_paths: Local model directories or ``.onnx`` file paths.
            model_names: Hugging Face model IDs (``model.onnx`` auto-downloaded).
            max_length: Maximum token sequence length.
            show_progress: Whether to show a tqdm progress bar for batches.
            device: ``"cpu"`` or ``"cuda"``. When ``None`` it is auto-detected
                (CPU unless a CUDA GPU with > 6 GB free VRAM and the ONNX
                CUDAExecutionProvider are available).
            weights: Optional per-model weights (normalised to sum to 1).
            token: Optional Hugging Face API token for private repos.
        """
        self.model_paths = model_paths or []
        self.model_names = model_names or []
        self.max_length = max_length
        self.show_progress = show_progress
        self.device = auto_select_device(device, backend="onnx")

        if not self.model_paths and not self.model_names:
            raise ValueError(
                "At least one of model_paths or model_names must be provided"
            )

        # Build one ONNX engine per model. Member progress bars are suppressed
        # so only the ensemble-level bar is shown.
        self.engines: List[OnnxInferenceEngine] = []
        for source in [*self.model_paths, *self.model_names]:
            engine = _build_onnx_member(
                source,
                device=self.device,
                max_length=max_length,
                show_progress=False,
                token=token,
            )
            self.engines.append(engine)

        if not self.engines:
            raise ValueError("No models were successfully loaded")

        # Normalise weights to sum to 1 (equal weighting by default).
        if weights is None:
            self.weights = [1.0 / len(self.engines)] * len(self.engines)
        else:
            if len(weights) != len(self.engines):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match number of "
                    f"models ({len(self.engines)})"
                )
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _weighted_probabilities(self, texts: List[str]) -> np.ndarray:
        """Return the weighted-mean softmax probabilities for a list of texts.

        Shape: ``(len(texts), num_classes)``.
        """
        weighted = None
        for engine, weight in zip(self.engines, self.weights):
            inputs = engine._tokenize(texts)
            probs = engine._run_session(inputs)  # (batch, num_classes)
            if weighted is None:
                weighted = np.zeros_like(probs)
            weighted += probs * weight
        return weighted

    # ------------------------------------------------------------------
    # Public API (drop-in for EnsembleInferenceEngine / OnnxInferenceEngine)
    # ------------------------------------------------------------------

    def predict_single(self, text: str) -> Dict[str, Union[str, float, List[float]]]:
        """Predict the class for a single text using the ONNX ensemble."""
        probs = self._weighted_probabilities([text])[0]
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
        """Predict classes for a list of texts using the ONNX ensemble."""
        predictions = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        batches = range(0, len(texts), batch_size)
        if self.show_progress:
            batches = tqdm(
                batches, total=num_batches, desc="ONNX ensemble inference", unit="batch"
            )

        for i in batches:
            batch_texts = texts[i : i + batch_size]
            probs = self._weighted_probabilities(batch_texts)
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
        """Run ensemble inference on every row of a CSV/JSON file.

        Output columns mirror :meth:`OnnxInferenceEngine.predict_from_file`:
        the original columns plus ``predicted_class``, ``predicted_probability``
        and ``class_probabilities``.
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
