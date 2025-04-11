#!/usr/bin/env python
"""
Script for uploading trained technical debt classification models to Hugging Face Hub.
"""

import os
import argparse
import json
from typing import Dict, Optional
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Upload a technical debt classification model to Hugging Face Hub")
    
    # Authentication arguments
    parser.add_argument("--token", type=str, required=True,
                      help="Hugging Face API token")
    
    # Repository arguments
    parser.add_argument("--repo_name", type=str, required=True,
                      help="Name of the repository to create or update (format: username/repo)")
    parser.add_argument("--repo_type", type=str, default="model", choices=["model", "dataset", "space"],
                      help="Type of repository to create")
    parser.add_argument("--repo_visibility", type=str, default="public", choices=["public", "private"],
                      help="Visibility of the repository")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to the trained model directory")
    parser.add_argument("--commit_message", type=str, default="Upload TD-Classifier model",
                      help="Commit message for the upload")
    
    # Additional arguments
    parser.add_argument("--create_if_not_exists", action="store_true",
                      help="Create the repository if it doesn't exist")
    parser.add_argument("--generate_card", action="store_true", 
                      help="Generate and upload a model card based on training config")
    
    return parser.parse_args()


def read_config_file(model_path: str, file_name: str) -> Optional[Dict]:
    """Read a configuration file from the model directory."""
    config_path = os.path.join(model_path, file_name)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return None


def generate_model_card(model_path: str) -> str:
    """Generate a model card based on the training configuration."""
    # Try to read configuration files
    training_config = read_config_file(model_path, "training_config.json")
    metrics = read_config_file(model_path, "metrics.json")
    
    # Default model card content
    card_content = "---\n"
    card_content += "language: en\n"
    card_content += "tags:\n"
    card_content += "- technical-debt\n"
    card_content += "- text-classification\n"
    card_content += "- transformers\n"
    card_content += "license: mit\n"
    card_content += "---\n\n"
    card_content += "# TD-Classifier Model\n\n"
    card_content += "This model was trained using the TD-Classifier Suite for technical debt classification.\n\n"
    
    # Add model information if available
    if training_config:
        card_content += "## Model Information\n\n"
        card_content += f"- Base Model: {training_config.get('model_name', 'Unknown')}\n"
        card_content += f"- Training Dataset: {training_config.get('data_file', 'Unknown')}\n"
        card_content += f"- Epochs: {training_config.get('num_epochs', 'Unknown')}\n"
        card_content += f"- Batch Size: {training_config.get('batch_size', 'Unknown')}\n"
        card_content += f"- Learning Rate: {training_config.get('learning_rate', 'Unknown')}\n"
        card_content += f"- Max Sequence Length: {training_config.get('max_length', 'Unknown')}\n"
        
        if training_config.get('cross_validation'):
            card_content += f"- Cross-Validation: Yes ({training_config.get('n_splits', 5)} folds)\n"
        else:
            card_content += "- Cross-Validation: No\n"
        
        card_content += "\n"
    
    # Add performance metrics if available
    if metrics:
        card_content += "## Performance Metrics\n\n"
        card_content += f"- Accuracy: {metrics.get('accuracy', 'Unknown'):.4f}\n"
        card_content += f"- F1 Score: {metrics.get('f1', 'Unknown'):.4f}\n"
        card_content += f"- Precision: {metrics.get('precision', 'Unknown'):.4f}\n"
        card_content += f"- Recall: {metrics.get('recall', 'Unknown'):.4f}\n"
        card_content += f"- ROC AUC: {metrics.get('roc_auc', 'Unknown'):.4f}\n"
        card_content += "\n"
    
    # Add usage information
    card_content += "## Usage\n\n"
    card_content += "```python\n"
    card_content += "from transformers import AutoTokenizer, AutoModelForSequenceClassification\n"
    card_content += "import torch\n\n"
    card_content += "# Load model and tokenizer\n"
    card_content += "tokenizer = AutoTokenizer.from_pretrained(\"REPO_NAME\")\n"
    card_content += "model = AutoModelForSequenceClassification.from_pretrained(\"REPO_NAME\")\n\n"
    card_content += "# Prepare text\n"
    card_content += "text = \"This code is a mess and needs refactoring.\"\n"
    card_content += "inputs = tokenizer(text, return_tensors=\"pt\", padding=True, truncation=True, max_length=512)\n\n"
    card_content += "# Make prediction\n"
    card_content += "with torch.no_grad():\n"
    card_content += "    outputs = model(**inputs)\n"
    card_content += "    logits = outputs.logits\n"
    card_content += "    probabilities = torch.softmax(logits, dim=1)\n"
    card_content += "    prediction = torch.argmax(probabilities, dim=1).item()\n"
    card_content += "    confidence = probabilities[0][prediction].item()\n\n"
    card_content += "print(f\"Prediction: {'Technical Debt' if prediction == 1 else 'Not Technical Debt'}\")\n"
    card_content += "print(f\"Confidence: {confidence:.4f}\")\n"
    card_content += "```\n\n"
    
    # Add citation information
    card_content += "## Citation\n\n"
    card_content += "If you use this model in your research, please cite:\n\n"
    card_content += "```\n"
    card_content += "@misc{TD-Classifier,\n"
    card_content += "  author = {TD-Classifier Suite},\n"
    card_content += "  title = {Technical Debt Classification Model},\n"
    card_content += "  year = {2024},\n"
    card_content += "  publisher = {Hugging Face},\n"
    card_content += "  howpublished = {\\url{https://huggingface.co/REPO_NAME}},\n"
    card_content += "}\n"
    card_content += "```\n"
    
    return card_content


def main():
    """Main function for uploading models to Hugging Face Hub."""
    args = parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path '{args.model_path}' does not exist")
    
    # Initialize Hugging Face API
    api = HfApi(token=args.token)
    
    print(f"🔐 Authenticating with Hugging Face using the provided token...")
    
    # Create repository if needed
    if args.create_if_not_exists:
        try:
            print(f"🔍 Checking if repository '{args.repo_name}' exists...")
            api.repo_info(repo_id=args.repo_name, repo_type=args.repo_type)
            print(f"✅ Repository '{args.repo_name}' already exists")
        except Exception:
            print(f"🛠️ Creating new {args.repo_visibility} repository: '{args.repo_name}'...")
            create_repo(
                repo_id=args.repo_name,
                token=args.token,
                private=args.repo_visibility == "private",
                repo_type=args.repo_type
            )
            print(f"✅ Repository created successfully")
    
    # Load model and tokenizer from local path to verify they can be loaded
    print(f"🔍 Verifying model can be loaded from '{args.model_path}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        print(f"✅ Model and tokenizer loaded successfully")
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")
    
    # Generate model card if requested
    if args.generate_card:
        print(f"📝 Generating model card...")
        card_content = generate_model_card(args.model_path)
        card_content = card_content.replace("REPO_NAME", args.repo_name)
        
        # Write model card to temporary file
        card_path = os.path.join(args.model_path, "README.md")
        with open(card_path, "w") as f:
            f.write(card_content)
        print(f"✅ Model card saved to {card_path}")
    
    # Upload model to Hugging Face Hub
    print(f"🚀 Uploading model to Hugging Face Hub...")
    api.upload_folder(
        folder_path=args.model_path,
        repo_id=args.repo_name,
        repo_type=args.repo_type,
        commit_message=args.commit_message
    )
    
    print(f"✅ Model successfully uploaded to https://huggingface.co/{args.repo_name}")
    print(f"🎉 You can now use it with: AutoModelForSequenceClassification.from_pretrained('{args.repo_name}')")


if __name__ == "__main__":
    main() 