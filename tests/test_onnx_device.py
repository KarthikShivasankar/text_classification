"""Tests for device selection and CLI --device/--gpu/--cpu reconciliation.

Covers:
- ``tdsuite.utils.onnx_inference.auto_select_device`` (ONNX and torch backends)
- ``tdsuite.inference._requested_device`` (CLI flag reconciliation)

All branches are made deterministic by monkeypatching the external probes
(``onnxruntime.get_available_providers``, ``torch.cuda``, and
``tdsuite.utils.onnx_inference._max_free_vram_gb``) so the tests never depend on
the actual machine's GPU and never touch the network.
"""

import sys
import types

import pytest

from tdsuite.utils.onnx_inference import (
    GPU_VRAM_THRESHOLD_GB,
    auto_select_device,
)
from tdsuite.cli import get_inference_parser
from tdsuite.inference import _requested_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install_fake_onnxruntime(monkeypatch, providers):
    """Install a fake ``onnxruntime`` module exposing ``get_available_providers``."""
    fake = types.ModuleType("onnxruntime")
    fake.get_available_providers = lambda: list(providers)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    return fake


def _install_fake_torch(monkeypatch, cuda_available):
    """Install a fake ``torch`` module with a controllable ``cuda.is_available``."""
    fake = types.ModuleType("torch")
    cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    fake.cuda = cuda
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


# ---------------------------------------------------------------------------
# auto_select_device — explicit requests are honored verbatim
# ---------------------------------------------------------------------------

class TestAutoSelectExplicit:
    def test_explicit_cpu(self):
        assert auto_select_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        assert auto_select_device("cuda") == "cuda"

    def test_explicit_is_lowercased(self):
        assert auto_select_device("CUDA") == "cuda"
        assert auto_select_device("CPU") == "cpu"

    def test_explicit_wins_for_torch_backend(self):
        # Even with an explicit request the backend probes must not matter.
        assert auto_select_device("cuda", backend="torch") == "cuda"


# ---------------------------------------------------------------------------
# auto_select_device — ONNX backend auto-detection
# ---------------------------------------------------------------------------

class TestAutoSelectOnnx:
    def test_no_cuda_provider_returns_cpu(self, monkeypatch):
        _install_fake_onnxruntime(monkeypatch, ["CPUExecutionProvider"])
        # VRAM probe should never decide the outcome here, but pin it high anyway.
        monkeypatch.setattr(
            "tdsuite.utils.onnx_inference._max_free_vram_gb", lambda: 99.0
        )
        assert auto_select_device(None, backend="onnx") == "cpu"

    def test_cuda_provider_high_vram_returns_cuda(self, monkeypatch):
        _install_fake_onnxruntime(
            monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        monkeypatch.setattr(
            "tdsuite.utils.onnx_inference._max_free_vram_gb",
            lambda: GPU_VRAM_THRESHOLD_GB + 2.0,
        )
        assert auto_select_device(None, backend="onnx") == "cuda"

    def test_cuda_provider_low_vram_returns_cpu(self, monkeypatch):
        _install_fake_onnxruntime(
            monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        monkeypatch.setattr(
            "tdsuite.utils.onnx_inference._max_free_vram_gb",
            lambda: GPU_VRAM_THRESHOLD_GB - 1.0,
        )
        assert auto_select_device(None, backend="onnx") == "cpu"

    def test_onnxruntime_missing_returns_cpu(self, monkeypatch):
        # Simulate onnxruntime not being importable.
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        assert auto_select_device(None, backend="onnx") == "cpu"


# ---------------------------------------------------------------------------
# auto_select_device — torch backend auto-detection
# ---------------------------------------------------------------------------

class TestAutoSelectTorch:
    def test_torch_cuda_unavailable_returns_cpu(self, monkeypatch):
        _install_fake_torch(monkeypatch, cuda_available=False)
        monkeypatch.setattr(
            "tdsuite.utils.onnx_inference._max_free_vram_gb", lambda: 99.0
        )
        assert auto_select_device(None, backend="torch") == "cpu"

    def test_torch_cuda_available_high_vram_returns_cuda(self, monkeypatch):
        _install_fake_torch(monkeypatch, cuda_available=True)
        monkeypatch.setattr(
            "tdsuite.utils.onnx_inference._max_free_vram_gb",
            lambda: GPU_VRAM_THRESHOLD_GB + 2.0,
        )
        assert auto_select_device(None, backend="torch") == "cuda"

    def test_torch_missing_returns_cpu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        assert auto_select_device(None, backend="torch") == "cpu"

    def test_custom_threshold_respected(self, monkeypatch):
        _install_fake_torch(monkeypatch, cuda_available=True)
        monkeypatch.setattr(
            "tdsuite.utils.onnx_inference._max_free_vram_gb", lambda: 4.0
        )
        # 4 GB free with a 2 GB threshold -> cuda
        assert auto_select_device(None, backend="torch", vram_threshold_gb=2.0) == "cuda"
        # 4 GB free with a 8 GB threshold -> cpu
        assert auto_select_device(None, backend="torch", vram_threshold_gb=8.0) == "cpu"


# ---------------------------------------------------------------------------
# _requested_device — CLI flag reconciliation
# ---------------------------------------------------------------------------

class TestRequestedDevice:
    def _parse(self, extra):
        base = ["--model_name", "m", "--text", "t"]
        return get_inference_parser().parse_args(base + extra)

    def test_neither_returns_none(self):
        ns = self._parse([])
        assert _requested_device(ns) is None

    def test_gpu_flag_returns_cuda(self):
        ns = self._parse(["--gpu"])
        assert _requested_device(ns) == "cuda"

    def test_cpu_flag_returns_cpu(self):
        ns = self._parse(["--cpu"])
        assert _requested_device(ns) == "cpu"

    def test_device_cuda_returns_cuda(self):
        ns = self._parse(["--device", "cuda"])
        assert _requested_device(ns) == "cuda"

    def test_device_cpu_returns_cpu(self):
        ns = self._parse(["--device", "cpu"])
        assert _requested_device(ns) == "cpu"

    def test_device_and_matching_flag_agree(self):
        ns = self._parse(["--device", "cuda", "--gpu"])
        assert _requested_device(ns) == "cuda"

    def test_conflicting_device_and_flag_raises(self):
        ns = self._parse(["--device", "cpu", "--gpu"])
        with pytest.raises(ValueError):
            _requested_device(ns)

    def test_gpu_and_cpu_flags_mutually_exclusive(self):
        # The parser itself rejects passing both convenience flags.
        with pytest.raises(SystemExit):
            self._parse(["--gpu", "--cpu"])
