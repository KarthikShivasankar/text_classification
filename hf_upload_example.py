#!/usr/bin/env python
"""
Example script demonstrating how to upload a model to Hugging Face Hub.
"""

import os
import subprocess
import sys


def main():
    """Run an example of uploading a model to Hugging Face Hub."""
    # Get user input for token
    print("=" * 80)
    print("Hugging Face Upload Example")
    print("=" * 80)
    print("\nThis script will guide you through uploading a model to Hugging Face Hub.")
    print("You will need a Hugging Face account and an access token with write permissions.")
    print("\nYou can create a token at: https://huggingface.co/settings/tokens")
    
    # Get token
    token = input("\nEnter your Hugging Face token: ").strip()
    if not token:
        print("❌ Token cannot be empty. Please try again.")
        return
    
    # Get repository name
    repo_name = input("\nEnter repository name (format: username/repo-name): ").strip()
    if not repo_name or "/" not in repo_name:
        print("❌ Invalid repository name. Format should be 'username/repo-name'")
        return
    
    # Get model path
    model_path = input("\nEnter path to the model directory: ").strip()
    if not model_path or not os.path.exists(model_path):
        print(f"❌ Model path '{model_path}' does not exist")
        return
    
    # Ask for visibility
    visibility = input("\nMake repository private? (y/n, default: n): ").strip().lower()
    repo_visibility = "private" if visibility == "y" else "public"
    
    # Ask for model card generation
    generate_card = input("\nGenerate model card? (y/n, default: y): ").strip().lower()
    generate_card_flag = "--generate_card" if generate_card != "n" else ""
    
    # Build command
    cmd = [
        sys.executable,
        "tdsuite/upload_to_hf.py",
        "--token", token,
        "--repo_name", repo_name,
        "--model_path", model_path,
        "--repo_visibility", repo_visibility,
        "--create_if_not_exists"
    ]
    
    if generate_card_flag:
        cmd.append(generate_card_flag)
    
    # Print command (without token)
    safe_cmd = cmd.copy()
    token_index = safe_cmd.index("--token") + 1
    safe_cmd[token_index] = "****"  # Hide token
    print("\nRunning command:")
    print(" ".join(safe_cmd))
    
    # Confirm
    confirm = input("\nProceed with upload? (y/n): ").strip().lower()
    if confirm != "y":
        print("❌ Upload cancelled")
        return
    
    # Execute command
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Upload process completed")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during upload: {e}")
        return


if __name__ == "__main__":
    main() 