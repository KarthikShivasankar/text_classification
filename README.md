# TD-Classifier Suite

A comprehensive suite for technical debt classification using transformer models. This package provides tools for training, evaluating, and deploying models to classify technical debt in software development contexts.

## Features

- **Classification Support**: Both binary and multi-class classification capabilities
- **State-of-the-art Models**: Built on transformer models like BERT, RoBERTa, and DistilBERT
- **Data Processing**: Flexible pipeline for handling various data formats and sources
- **Advanced Training**: Cross-validation, early stopping, and class weighting for imbalanced datasets
- **Carbon Tracking**: Monitor carbon emissions during model training
- **Visualization**: Comprehensive metrics visualization including ROC curves, confusion matrices, and more
- **CLI Interface**: Easy-to-use command-line tools for training and inference
- **Batch Processing**: Efficient batch inference for large datasets
- **Flexible Model Loading**: Support for both local models and Hugging Face models
- **Multiple Data Sources**: Load data from local files or Hugging Face datasets
- **Ensemble Inference**: Combine predictions from multiple models with optional weighting for improved accuracy
- **Data Splitting**: Split data into training and test sets with balanced classes and extract top repositories

## Installation

```bash
# Clone the repository
git clone "our_repo"
cd our_repo

# Install the package
pip install -e .
```

For development installation with additional tools:
```bash
pip install -e ".[dev]"
```

## Usage

### Data Preparation

The package supports loading data from both local files and Hugging Face datasets:

#### Local Files
The package expects data in CSV, JSON, or JSONL format with at least two columns:
- `text`: The text content to classify
- `label`: The class label

Example CSV data format with text labels:
```csv
text,label
"This code has complex logic that should be simplified",code_smell
"The documentation is outdated and needs to be updated",documentation_debt
"The architecture violates the single responsibility principle",design_debt
```

Example CSV data format with numeric labels (0 or 1):
```csv
text,label
"This code has complex logic that should be simplified",1
"The documentation is outdated and needs to be updated",0
"The architecture violates the single responsibility principle",1
```


#### Data Splitting

You can split your data into training and test sets with balanced classes, and extract top repositories:

```bash
# Using a Hugging Face dataset with numeric labels
tdsuite-split-data --data_file "karths/binary-10IQR-TD" \
                   --output_dir "data/split" \
                   --is_numeric_labels \
                   --repo_column "repo" \
                   --is_huggingface_dataset
```

This will:
1. Load the data from the Hugging Face dataset
2. Use the existing numeric labels (0/1) directly
3. Balance the classes in the dataset
4. Split the data into training and test sets
5. Extract the top repositories with the most positive class samples
6. Save the split data to the specified output directory

The output directory will contain:
- `train.csv`: Training data with balanced classes
- `test.csv`: Test data with balanced classes
- `top_repos.csv`: Data from the top repositories with balanced classes
- `label_mappings.json`: Mapping from numeric labels to text labels (if text labels are used)

### Training

#### Binary Classification

Train a model to classify text as either technical debt or not:

```bash
# Using a Hugging Face dataset with text labels
tdsuite-train \
    --data_file "karths/binary-10IQR-people" \
    --model_name "distilbert-base-uncased" \
    --classification_type binary \
    --positive_category "people" \
    --output_dir "outputs/binary"

# Using a local data file with text labels
tdsuite-train \
    --data_file "data/split/train.csv" \
    --model_name "distilbert-base-uncased" \
    --classification_type binary \
    --positive_category "TD" \
    --output_dir "outputs/binary"
```

#### Multi-class Classification

Train a model to classify text into multiple technical debt categories:

```bash
# Using a Hugging Face dataset with numeric labels
tdsuite-train \
    --data_file "eevvgg/ad-hominem-multiclass" \
    --model_name "distilbert-base-uncased" \
    --classification_type multi \
    --numeric_labels \
    --num_classes 6 \
    --output_dir "outputs/multiclass"
```

# Using a Hugging Face dataset with text labels

```bash
tdsuite-train \
    --data_file "karths/Multi_class_debt_types" \
    --model_name "distilbert-base-uncased" \
    --classification_type multi \
    --categories "code" "documentation" "design" \
    --output_dir "outputs/multiclass"
```
### Inference

#### Batch Inference

Process multiple texts from a file:

```bash
# Using a local model
tdsuite-inference \
    --model_path "outputs/binary" \
    --input_file "data/split/test.csv" \
    --output_file "outputs/test_predictions.csv"

# Using a Hugging Face model
tdsuite-inference \
    --model_name "karths/TD_model_Deberta \
    --input_file "data/split/test.csv" \
    --output_file "outputs/test_predictions.csv"
```

### Output Files

After training, the following files will be saved in the output directory:

- `pytorch_model.bin`: The trained model weights
- `model_config.json`: Model configuration
- `training_config.json`: Training configuration
- `data_config.json`: Data processing configuration
- `metrics.json`: Training and evaluation metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve visualization (for binary classification)
- `carbon_emissions.json`: Carbon emissions data from training

After inference with ground truth labels, the following files will be saved:

- `predictions.csv`: Model predictions
- `metrics.json`: Evaluation metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve visualization (for binary classification)

## Project Structure

```
tdsuite/
├── config/             # Configuration management
│   ├── __init__.py
│   └── config.py       # Configuration classes
├── data/               # Data processing and dataset classes
│   ├── __init__.py
│   ├── dataset.py      # Dataset and processor classes
│   └── data_splitter.py # Data splitting utilities
├── models/             # Model implementations
│   ├── __init__.py
│   ├── base.py         # Base model class
│   └── transformer.py  # Transformer model implementation
├── trainers/           # Training utilities
│   ├── __init__.py
│   ├── base.py         # Base trainer class
│   └── td_trainer.py   # Technical debt trainer
├── utils/              # Utility functions
│   ├── __init__.py
│   └── inference.py    # Inference engine
├── train.py            # Training script
├── inference.py        # Inference script
└── split_data.py       # Data splitting script
```

## Advanced Usage

### Cross-Validation

Cross-validation helps evaluate model performance more robustly:

```bash
tdsuite-train --data_file "karths/binary-10IQR-port" \
              --model_name "distilbert-base-uncased" \
              --classification_type binary \
              --positive_category "port" \
              --output_dir "outputs/cv" \
              --cross_validation \
              --n_splits 10
```

This will:
1. Split the data into 10 folds
2. Train a model on each fold
3. Evaluate on the held-out fold
4. Aggregate the results across all folds
5. Generate visualizations of the cross-validation results

### Early Stopping

Early stopping prevents overfitting by stopping training when the model stops improving:

```bash
tdsuite-train --data_file "karths/binary-10IQR-port" \
              --model_name "distilbert-base-uncased" \
              --classification_type binary \
              --positive_category "port" \
              --output_dir "outputs/early_stopping" \
              --early_stopping \
              --patience 5
```

This will:
1. Monitor the evaluation loss during training
2. Stop training if the loss doesn't improve for 5 epochs
3. Restore the model from the best checkpoint

### Carbon Emissions Tracking

Track the carbon emissions of your training process:

```bash
tdsuite-train --data_file "karths/binary-10IQR-port" \
              --model_name "distilbert-base-uncased" \
              --classification_type binary \
              --positive_category "port" \
              --output_dir "outputs/emissions" \
              --track_emissions
```

This will:
1. Track energy consumption during training
2. Calculate carbon emissions based on your location
3. Save the emissions data to a CSV file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
