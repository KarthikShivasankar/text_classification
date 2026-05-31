#!/usr/bin/env python
"""Export all karths/binary_classification_train_* models to ONNX and upload model.onnx
back to the same Hugging Face Hub repository.

Reads HF_TOKEN from the .env file in the project root.

Usage:
    python scripts/export_and_upload_onnx.py
    python scripts/export_and_upload_onnx.py --dry-run          # list models only
    python scripts/export_and_upload_onnx.py \\
        --model karths/binary_classification_train_TD
    python scripts/export_and_upload_onnx.py --opset 17 --max-length 256

Requirements:
    uv pip install onnx onnxruntime
    # onnxruntime already installed via: uv pip install -e ".[onnx]"
"""

import argparse
import os
import sys
import tempfile
import time

# Enable robust, resumable large-file transfers before huggingface_hub loads.
# hf_transfer/hf_xet chunk uploads, which survives flaky connections far better
# than the default single-stream upload (avoids WinError 10054 resets).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_env_token(env_path: str = None) -> str:
    """Read HF_TOKEN from the .env file (key=value format)."""
    if env_path is None:
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
        )

    if not os.path.exists(env_path):
        raise FileNotFoundError(f".env file not found: {env_path}")

    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()

    raise ValueError("HF_TOKEN not found in .env file")


# The 17 TDSuite pre-trained models listed in the README.
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


def list_td_models(api, prefixes: list = None) -> list:
    """Return TDSuite model IDs. If prefixes supplied, filter by those instead."""
    if prefixes:
        all_models = list(api.list_models(author="karths"))
        return sorted(
            m.id for m in all_models if any(m.id.startswith(p) for p in prefixes)
        )
    return list(TDSUITE_MODELS)


def has_onnx(api, repo_id: str) -> bool:
    """Return True if model.onnx already exists in the repo."""
    try:
        files = list(api.list_repo_files(repo_id))
        return "model.onnx" in files
    except Exception:
        return False


def onnx_is_fixed(api, repo_id: str, min_mb: float = 10.0, retries: int = 3) -> bool:
    """Return True if model.onnx exists and is a real, self-contained model.

    The old broken stubs are tiny (~0.9 MB) because their weights lived in a
    missing external sidecar. A correctly exported self-contained model is
    hundreds of MB, so a size threshold reliably distinguishes the two.
    Retries because the Hub tree call can hit transient connection resets.
    """
    for attempt in range(retries):
        try:
            for entry in api.list_repo_tree(repo_id):
                if getattr(entry, "path", None) == "model.onnx":
                    size = getattr(entry, "size", 0) or 0
                    return size > min_mb * 1024 * 1024
            return False
        except Exception:
            if attempt < retries - 1:
                time.sleep(3)
    return False


def export_model_to_onnx(
    model_id: str, output_path: str, max_length: int, opset: int, token: str = None
):
    """Download a HF model and export it to a single self-contained ONNX file.

    Uses the shared TorchDynamo-based exporter (transformers>=5 compatible) and
    consolidates external weight data into one ``model.onnx`` (no sidecar), so
    the uploaded file is portable and loads standalone on CPU.
    """
    from tdsuite.utils.onnx_inference import export_transformer_to_onnx

    print(f"  Loading + exporting: {model_id}")
    _, tokenizer = export_transformer_to_onnx(
        model_source=model_id,
        output_path=output_path,
        max_length=max_length,
        token=token,
        opset=opset,
    )
    return tokenizer


def upload_onnx_to_repo(
    api, repo_id: str, onnx_path: str, tokenizer, commit_msg: str, max_retries: int = 4
):
    """Upload model.onnx (and refresh tokenizer files) to the HF repo.

    Retries on transient network failures (e.g. connection resets) with
    exponential backoff. hf_transfer/hf_xet make the large upload resumable so
    retries resume rather than restart.
    """
    import shutil

    with tempfile.TemporaryDirectory() as tmp:
        # Save tokenizer alongside so the repo has everything needed for CPU inference
        tokenizer.save_pretrained(tmp)

        # Copy onnx file into the temp directory
        dest = os.path.join(tmp, "model.onnx")
        shutil.copy2(onnx_path, dest)

        last_err = None
        for attempt in range(1, max_retries + 1):
            # Force a fresh HTTP client each attempt. After a connection reset
            # huggingface_hub's cached httpx client is closed and every later
            # request raises "client has been closed"; resetting it lets the
            # retry actually reconnect.
            try:
                from huggingface_hub.utils._http import close_session

                close_session()
            except Exception:
                pass

            try:
                print(
                    f"  Uploading model.onnx to {repo_id} "
                    f"(attempt {attempt}/{max_retries}) ..."
                )
                api.upload_folder(
                    folder_path=tmp,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=commit_msg,
                    # Only upload onnx + tokenizer files; leave the rest as-is
                    allow_patterns=[
                        "model.onnx",
                        "tokenizer*",
                        "special_tokens_map.json",
                        "vocab*",
                        "merges.txt",
                    ],
                )
                print(f"  Uploaded -> https://huggingface.co/{repo_id}")
                return
            except Exception as exc:
                last_err = exc
                wait = min(60, 5 * (2 ** (attempt - 1)))
                print(f"  Upload attempt {attempt} failed: {exc}")
                if attempt < max_retries:
                    print(f"  Retrying in {wait}s ...")
                    time.sleep(wait)

        raise RuntimeError(
            f"Upload to {repo_id} failed after {max_retries} attempts: {last_err}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export all binary_classification_train_* models to ONNX "
            "and upload to HF Hub"
        )
    )
    parser.add_argument(
        "--model",
        help="Export a single model ID instead of all TDSuite classifier models",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        dest="prefixes",
        default=None,
        help=(
            "Model ID prefix to filter. Can be specified multiple times. "
            "Default: both karths/binary_classification_train_ and "
            "karths/modernbertbase-binary-10IQR-"
        ),
    )
    parser.add_argument(
        "--opset", type=int, default=14, help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Sequence length for export dummy input (default: 512)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip models that already have model.onnx in the repo (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-export and re-upload even if model.onnx already exists",
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help=(
            "Process each model in its own subprocess. This gives every model a "
            "fresh HTTP client, so a connection reset on one model cannot poison "
            "the rest (recommended on flaky networks)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List models that would be processed without exporting or uploading",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Path to .env file (default: <project_root>/.env)",
    )
    args = parser.parse_args()

    # Load token
    token = load_env_token(args.env)

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Determine model list
    if args.model:
        model_ids = [args.model]
    else:
        prefixes = args.prefixes or None  # None → use defaults
        model_ids = list_td_models(api, prefixes=prefixes)

    print(f"Found {len(model_ids)} model(s) to process:")
    for mid in model_ids:
        print(f"  {mid}")

    if args.dry_run:
        print("\nDry run — exiting without exporting.")
        return

    # Isolated mode: run each model in its own subprocess so a poisoned HTTP
    # client (after a connection reset) cannot cascade to the other models.
    if args.isolate:
        import subprocess

        ok, failed, skipped = [], [], []
        for model_id in model_ids:
            if args.skip_existing and onnx_is_fixed(api, model_id):
                print(f"SKIP (already fixed): {model_id}")
                skipped.append(model_id)
                continue
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--model",
                model_id,
                "--no-skip-existing",
                "--opset",
                str(args.opset),
                "--max-length",
                str(args.max_length),
            ]
            if args.env:
                cmd += ["--env", args.env]
            print(f"\n{'='*60}\n>>> Subprocess upload: {model_id}")
            rc = subprocess.run(cmd).returncode
            (ok if rc == 0 else failed).append(model_id)

        print(f"\n{'='*60}\nIsolated run summary:")
        print(f"  Uploaded : {len(ok)}")
        print(f"  Skipped  : {len(skipped)}")
        print(f"  Failed   : {len(failed)}")
        if failed:
            print("Failed models (re-run to retry):")
            for mid in failed:
                print(f"  {mid}")
            sys.exit(1)
        return

    results = {"skipped": [], "exported": [], "failed": []}

    with tempfile.TemporaryDirectory() as workdir:
        for model_id in model_ids:
            print(f"\n{'='*60}")
            print(f"Processing: {model_id}")

            if args.skip_existing and onnx_is_fixed(api, model_id):
                print(
                    "  model.onnx already fixed — skipping "
                    "(use --no-skip-existing to force)"
                )
                results["skipped"].append(model_id)
                continue

            onnx_path = os.path.join(workdir, model_id.replace("/", "__") + ".onnx")

            try:
                tokenizer = export_model_to_onnx(
                    model_id=model_id,
                    output_path=onnx_path,
                    max_length=args.max_length,
                    opset=args.opset,
                    token=token,
                )
                upload_onnx_to_repo(
                    api=api,
                    repo_id=model_id,
                    onnx_path=onnx_path,
                    tokenizer=tokenizer,
                    commit_msg="Add ONNX export for CPU inference (no GPU required)",
                )
                results["exported"].append(model_id)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                results["failed"].append((model_id, str(exc)))

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Exported & uploaded : {len(results['exported'])}")
    print(f"  Skipped (existing)  : {len(results['skipped'])}")
    print(f"  Failed              : {len(results['failed'])}")
    if results["failed"]:
        print("\nFailed models:")
        for mid, err in results["failed"]:
            print(f"  {mid}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
