#!/usr/bin/env python
"""
Setup script for installing dependencies required for the Hugging Face upload script.
"""

import subprocess
import sys
import os


def install_dependencies():
    """Install dependencies required for the Hugging Face upload script."""
    dependencies = [
        "huggingface_hub>=0.19.0",
        "transformers>=4.35.0",
        "torch>=2.0.0"
    ]
    
    print("Installing dependencies for Hugging Face upload script...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + dependencies)
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)


if __name__ == "__main__":
    install_dependencies()
    
    # Check if the upload script exists
    upload_script_path = os.path.join("tdsuite", "upload_to_hf.py")
    if not os.path.exists(upload_script_path):
        print(f"❌ Upload script not found at {upload_script_path}")
        sys.exit(1)
    
    print("\n🎉 Setup completed. You can now use the upload script with:")
    print(f"\npython {upload_script_path} --token YOUR_HF_TOKEN --repo_name username/repo --model_path path/to/model --create_if_not_exists --generate_card\n")
    print("For more options, run:")
    print(f"\npython {upload_script_path} --help\n") 