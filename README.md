# Text Classifier

A Python package for fine-tuning transformer models for text classification tasks.

## Features

- Binary classification for single categories
- Binary classification for multiple categories (one-vs-rest approach)
- Multi-class classification
- K-fold cross-validation training
- Comprehensive metrics calculation and visualization
- Class balancing for imbalanced datasets
- Top repository extraction for evaluation
- Integration with Weights & Biases for experiment tracking

## Installation

```bash
pip install -e .
```

## Package Structure

```
text_classifier/
├── config/             # Configuration classes
├── data/               # Data processing utilities
├── models/             # Model definitions
├── trainers/           # Training utilities
├── utils/              # Utility functions
├── train_binary.py     # Script for binary classification
├── train_binary_multiple.py  # Script for multiple binary classifications
├── train_multiclass.py # Script for multi-class classification
├── setup.py            # Package installation
└── README.md           # Documentation
```

## Usage

### Binary Classification (Single Category)

```bash
# Using command-line arguments
train-binary --data_file path/to/data.csv --output_dir ./output --model_name bert-base-uncased --category your_category --num_train_epochs 3

# Using a configuration file
train-binary --config path/to/config.json
```

### Binary Classification (Multiple Categories)

```bash
# Using command-line arguments
train-binary-multiple --data_file path/to/data.csv --output_dir ./output --model_name bert-base-uncased --categories category1 category2 category3 --num_train_epochs 3

# Using a configuration file
train-binary-multiple --config path/to/config.json

# Using all categories in the dataset
train-binary-multiple --data_file path/to/data.csv --output_dir ./output --model_name bert-base-uncased --use_all_categories
```

### Multi-class Classification

```bash
# Using command-line arguments
train-multiclass --data_file path/to/data.csv --output_dir ./output --model_name bert-base-uncased --num_train_epochs 3

# Using a configuration file
train-multiclass --config path/to/config.json
```

## Configuration

You can specify model and training parameters using a JSON configuration file:

```json
{
  "model_name": "bert-base-uncased",
  "num_train_epochs": 3,
  "learning_rate": 2e-5,
  "per_device_train_batch_size": 16,
  "per_device_eval_batch_size": 64,
  "warmup_steps": 500,
  "weight_decay": 0.01,
  "logging_dir": "./logs",
  "logging_steps": 10,
  "evaluation_strategy": "epoch",
  "save_strategy": "epoch",
  "load_best_model_at_end": true,
  "metric_for_best_model": "f1",
  "greater_is_better": true,
  "seed": 42
}
```

## Weights & Biases Integration

To enable Weights & Biases logging, add the `--wandb` flag:

```bash
train-binary --data_file path/to/data.csv --category your_category --wandb
```

## Output

After training, the following files will be generated in the output directory:

- Model checkpoints
- Training logs
- Evaluation metrics
- Confusion matrix visualization
- Precision-recall curves
- ROC curves

## Requirements

- Python 3.7+
- PyTorch 1.7+
- Transformers 4.5+
- scikit-learn 0.24+
- pandas 1.1+
- numpy 1.19+
- matplotlib 3.3+
- seaborn 0.11+
- Weights & Biases 0.12+ (optional)

## License

MIT
