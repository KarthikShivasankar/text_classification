#!/usr/bin/env python
"""Export all karths/binary_classification_train_* models to ONNX and upload model.onnx
back to the same Hugging Face Hub repository.

Reads HF_TOKEN from the .env file in the project root.

Usage:
    python scripts/export_and_upload_onnx.py
    python scripts/export_and_upload_onnx.py --dry-run          # list models only
    python scripts/export_and_upload_onnx.py --model karths/binary_classification_train_TD
    python scripts/export_and_upload_onnx.py --opset 17 --max-length 256

Requirements:
    uv pip install onnx onnxruntime
    # onnxruntime already installed via: uv pip install -e ".[onnx]"
"""

import argparse
import os
import sys
import tempfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env_token(env_path: str = None) -> str:
    """Read HF_TOKEN from the .env file (key=value format)."""
    if env_path is None:
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")

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


def export_model_to_onnx(model_id: str, output_path: str, max_length: int, opset: int):
    """Download a HF model and export it to ONNX."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        print(
            "Error: 'onnx' package not installed.\n"
            "Install with: uv pip install onnx onnxruntime",
            file=sys.stderr,
        )
        sys.exit(1)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"  Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()

    dummy_inputs = tokenizer(
        "example technical debt text for export",
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )

    input_names = list(dummy_inputs.keys())
    dynamic_axes = {n: {0: "batch_size", 1: "sequence_length"} for n in input_names}
    dynamic_axes["logits"] = {0: "batch_size"}

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"  Exporting to ONNX (opset {opset}) → {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(dummy_inputs[k] for k in input_names),
            output_path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Exported ({size_mb:.1f} MB)")
    return tokenizer


def upload_onnx_to_repo(api, repo_id: str, onnx_path: str, tokenizer, commit_msg: str):
    """Upload model.onnx (and refresh tokenizer files) to the HF repo."""
    with tempfile.TemporaryDirectory() as tmp:
        # Save tokenizer alongside so the repo has everything needed for CPU inference
        tokenizer.save_pretrained(tmp)

        # Copy onnx file into the temp directory
        import shutil
        dest = os.path.join(tmp, "model.onnx")
        shutil.copy2(onnx_path, dest)

        print(f"  Uploading model.onnx to {repo_id} …")
        api.upload_folder(
            folder_path=tmp,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg,
            # Only upload onnx + tokenizer files; leave the rest of the repo untouched
            allow_patterns=["model.onnx", "tokenizer*", "special_tokens_map.json", "vocab*", "merges.txt"],
        )
    print(f"  Uploaded → https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export all binary_classification_train_* models to ONNX and upload to HF Hub"
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
            "Default: both karths/binary_classification_train_ and karths/modernbertbase-binary-10IQR-"
        ),
    )
    parser.add_argument(
        "--opset", type=int, default=14, help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="Sequence length for export dummy input (default: 512)"
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

    results = {"skipped": [], "exported": [], "failed": []}

    with tempfile.TemporaryDirectory() as workdir:
        for model_id in model_ids:
            print(f"\n{'='*60}")
            print(f"Processing: {model_id}")

            if args.skip_existing and has_onnx(api, model_id):
                print("  model.onnx already exists — skipping (use --no-skip-existing to force)")
                results["skipped"].append(model_id)
                continue

            onnx_path = os.path.join(workdir, model_id.replace("/", "__") + ".onnx")

            try:
                tokenizer = export_model_to_onnx(
                    model_id=model_id,
                    output_path=onnx_path,
                    max_length=args.max_length,
                    opset=args.opset,
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


if __name__ == "__main__":
    main()
