# Hugging Face Upload Script

This tool allows you to upload your trained TD-Classifier models to the Hugging Face Hub, making them easily accessible for inference and sharing with the community.

## Setup

Before using the upload script, you need to install the required dependencies:

```bash
python setup_upload_script.py
```

This will install the necessary packages for interacting with the Hugging Face Hub.

## Getting a Hugging Face Token

1. Sign up or log in to [Hugging Face](https://huggingface.co/)
2. Go to your profile settings and navigate to the "Access Tokens" section
3. Create a new token with "write" permissions
4. Copy the token for use with the upload script

## Usage

After training a model with the TD-Classifier Suite, you can upload it to Hugging Face with:

```bash
python tdsuite/upload_to_hf.py \
    --token YOUR_HF_TOKEN \
    --repo_name username/repository_name \
    --model_path path/to/trained/model \
    --create_if_not_exists \
    --generate_card
```

### Required Arguments

- `--token`: Your Hugging Face API token with write access
- `--repo_name`: Name of the repository to create or update (format: username/repo)
- `--model_path`: Path to the trained model directory

### Optional Arguments

- `--repo_type`: Type of repository to create (choices: "model", "dataset", "space"; default: "model")
- `--repo_visibility`: Visibility of the repository (choices: "public", "private"; default: "public")
- `--commit_message`: Commit message for the upload (default: "Upload TD-Classifier model")
- `--create_if_not_exists`: Create the repository if it doesn't exist (flag)
- `--generate_card`: Generate and upload a model card based on training config (flag)

## Examples

### Upload a model with a custom repository name

```bash
python tdsuite/upload_to_hf.py \
    --token YOUR_HF_TOKEN \
    --repo_name username/td-classifier-v1 \
    --model_path outputs/binary \
    --create_if_not_exists
```

### Upload a model as a private repository

```bash
python tdsuite/upload_to_hf.py \
    --token YOUR_HF_TOKEN \
    --repo_name username/td-classifier-private \
    --model_path outputs/binary \
    --repo_visibility private \
    --create_if_not_exists
```

### Upload a model with a specific commit message

```bash
python tdsuite/upload_to_hf.py \
    --token YOUR_HF_TOKEN \
    --repo_name username/td-classifier \
    --model_path outputs/binary \
    --commit_message "Add model trained on new dataset" \
    --create_if_not_exists
```

## After Uploading

Once your model is uploaded, you can use it directly from Hugging Face in any application:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("username/repository_name")
model = AutoModelForSequenceClassification.from_pretrained("username/repository_name")

# Use the model for inference
# ...
```

## Troubleshooting

### Token Issues

If you encounter authentication errors, verify that:
- Your token has write permissions
- Your token is entered correctly (no extra spaces)
- Your token is still valid (not expired)

### Repository Naming

- Repository names should follow Hugging Face's naming conventions
- If you get a "repository already exists" error, either use a different name or remove the `--create_if_not_exists` flag

### Connection Issues

If you encounter connection problems:
- Check your internet connection
- Ensure firewall settings allow outbound connections to huggingface.co
- Try again later, as Hugging Face might be experiencing service interruptions 