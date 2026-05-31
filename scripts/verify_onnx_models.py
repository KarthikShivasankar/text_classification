#!/usr/bin/env python
"""Strictly verify ONNX CPU inference for all TDSuite category models.

For each model this:
  1. (optionally) clears the locally cached model.onnx so a fresh copy is pulled
  2. downloads model.onnx from the Hub (with retries for flaky networks)
  3. loads it DIRECTLY with onnxruntime on CPU — no torch, no export fallback,
     so a broken/incomplete Hub file fails loudly instead of being masked
  4. runs a sample prediction

Run:  python scripts/verify_onnx_models.py [--fresh] [--model <id>]
"""

import argparse
import os
import shutil
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TDSUITE_MODELS = [
    "karths/binary_classification_train_TD",
    "karths/binary_classification_train_architecture",
    "karths/binary_classification_train_code",
    "karths/binary_classification_train_defect",
    "karths/binary_classification_train_infrastructure",
    "karths/binary_classification_train_perf",
    "karths/binary_classification_train_requirement",
    "karths/binary_classification_train_design",
    "karths/binary_classification_train_secu",
    "karths/binary_classification_train_usab",
    "karths/binary_classification_train_reli",
    "karths/binary_classification_train_process",
    "karths/binary_classification_train_build",
    "karths/binary_classification_train_main",
    "karths/binary_classification_train_automation",
    "karths/binary_classification_train_people",
    "karths/binary_classification_train_port",
]

SAMPLE = (
    "The authentication module has no rate limiting and stores passwords in plain text."
)


def _clear_cached_onnx(model_id: str):
    """Delete any locally cached model.onnx blobs/snapshots for a fresh pull."""
    from huggingface_hub.constants import HF_HUB_CACHE

    repo_dir = os.path.join(HF_HUB_CACHE, "models--" + model_id.replace("/", "--"))
    shutil.rmtree(repo_dir, ignore_errors=True)


def _download_with_retries(model_id: str, retries: int = 6) -> str:
    last = None
    for attempt in range(1, retries + 1):
        try:
            try:
                from huggingface_hub.utils._http import close_session

                close_session()
            except Exception:
                pass
            return hf_hub_download(model_id, "model.onnx")
        except Exception as exc:
            last = exc
            wait = min(30, 3 * attempt)
            print(f"    download attempt {attempt} failed: {str(exc)[:120]}")
            if attempt < retries:
                time.sleep(wait)
    raise RuntimeError(f"download failed after {retries} attempts: {last}")


def verify_one(model_id: str, fresh: bool) -> dict:
    short = model_id.split("/")[-1].replace("binary_classification_train_", "")
    entry = {"model": short, "status": "FAIL", "detail": ""}
    try:
        if fresh:
            _clear_cached_onnx(model_id)

        onnx_path = _download_with_retries(model_id)
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

        # Direct standalone load on CPU — no torch involved at all.
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )

        tok = AutoTokenizer.from_pretrained(model_id)
        enc = tok(SAMPLE, return_tensors="np", truncation=True, max_length=512)
        names = [i.name for i in sess.get_inputs()]
        feeds = {k: enc[k] for k in names if k in enc}
        logits = sess.run(None, feeds)[0][0]
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        pred = int(np.argmax(probs))

        entry["status"] = "OK"
        entry["detail"] = f"{size_mb:6.1f}MB  class={pred}  p={probs[pred]:.3f}"
    except Exception as exc:
        entry["detail"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    mark = "OK  " if entry["status"] == "OK" else "FAIL"
    print(f"[{mark}] {entry['model']:16} {entry['detail']}")
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Clear cache and re-download each model.onnx",
    )
    ap.add_argument("--model", help="Verify a single model ID")
    args = ap.parse_args()

    models = [args.model] if args.model else TDSUITE_MODELS
    print("=" * 64)
    print(
        f"Verifying ONNX CPU inference for {len(models)} model(s) "
        f"(fresh={args.fresh})"
    )
    print("=" * 64)

    results = [verify_one(m, args.fresh) for m in models]
    ok = sum(1 for r in results if r["status"] == "OK")
    fail = len(results) - ok

    print("=" * 64)
    print(f"Summary: {ok}/{len(results)} passed, {fail} failed")
    if fail:
        print("Failed:")
        for r in results:
            if r["status"] != "OK":
                print(f"  {r['model']}: {r['detail']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
