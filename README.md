# TD-Classifier Suite

A comprehensive suite for technical debt classification using transformer models. This package provides tools for training, evaluating, and deploying models to classify technical debt in software development contexts.

## Features

- **Binary Classification**: Classify text as either technical debt or not
- **State-of-the-art Models**: Built on transformer models like BERT, RoBERTa, DistilBERT and DeBERTav3 
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

## Hugging Face Integration

Our suite is fully compatible with Hugging Face's ecosystem, supporting both datasets and pre-trained models from Hugging Face Hub.

### Available Datasets

We provide a comprehensive collection of technical debt datasets on Hugging Face, covering various aspects of software development:

- **General Technical Debt**: [karths/binary-10IQR-TD](https://huggingface.co/datasets/karths/binary-10IQR-TD)
- **Architecture**: [karths/binary-10IQR-architecture](https://huggingface.co/datasets/karths/binary-10IQR-architecture)
- **Code Quality**: [karths/binary-10IQR-code](https://huggingface.co/datasets/karths/binary-10IQR-code)
- **Defects**: [karths/binary-10IQR-defect](https://huggingface.co/datasets/karths/binary-10IQR-defect)
- **Infrastructure**: [karths/binary-10IQR-infrastructure](https://huggingface.co/datasets/karths/binary-10IQR-infrastructure)
- **Performance**: [karths/binary-10IQR-perf](https://huggingface.co/datasets/karths/binary-10IQR-perf)
- **Requirements**: [karths/binary-10IQR-requirement](https://huggingface.co/datasets/karths/binary-10IQR-requirement)
- **Design**: [karths/binary-10IQR-design](https://huggingface.co/datasets/karths/binary-10IQR-design)
- **Security**: [karths/binary-10IQR-secu](https://huggingface.co/datasets/karths/binary-10IQR-secu)
- **Usability**: [karths/binary-10IQR-usab](https://huggingface.co/datasets/karths/binary-10IQR-usab)
- **Compatibility**: [karths/binary-10IQR-comp](https://huggingface.co/datasets/karths/binary-10IQR-comp)
- **Reliability**: [karths/binary-10IQR-reli](https://huggingface.co/datasets/karths/binary-10IQR-reli)
- **Process**: [karths/binary-10IQR-process](https://huggingface.co/datasets/karths/binary-10IQR-process)
- **Build**: [karths/binary-10IQR-build](https://huggingface.co/datasets/karths/binary-10IQR-build)
- **Maintenance**: [karths/binary-10IQR-main](https://huggingface.co/datasets/karths/binary-10IQR-main)
- **Automation**: [karths/binary-10IQR-automation](https://huggingface.co/datasets/karths/binary-10IQR-automation)
- **People**: [karths/binary-10IQR-people](https://huggingface.co/datasets/karths/binary-10IQR-people)
- **Portability**: [karths/binary-10IQR-port](https://huggingface.co/datasets/karths/binary-10IQR-port)

### Pre-trained Models

We provide pre-trained models for each category, available on Hugging Face Hub:

- **General Technical Debt**: [karths/binary_classification_train_TD](https://huggingface.co/karths/binary_classification_train_TD)
- **Architecture**: [karths/binary_classification_train_architecture](https://huggingface.co/karths/binary_classification_train_architecture)
- **Code Quality**: [karths/binary_classification_train_code](https://huggingface.co/karths/binary_classification_train_code)
- **Defects**: [karths/binary_classification_train_defect](https://huggingface.co/karths/binary_classification_train_defect)
- **Infrastructure**: [karths/binary_classification_train_infrastructure](https://huggingface.co/karths/binary_classification_train_infrastructure)
- **Performance**: [karths/binary_classification_train_perf](https://huggingface.co/karths/binary_classification_train_perf)
- **Requirements**: [karths/binary_classification_train_requirement](https://huggingface.co/karths/binary_classification_train_requirement)
- **Design**: [karths/binary_classification_train_design](https://huggingface.co/karths/binary_classification_train_design)
- **Security**: [karths/binary_classification_train_secu](https://huggingface.co/karths/binary_classification_train_secu)
- **Usability**: [karths/binary_classification_train_usab](https://huggingface.co/karths/binary_classification_train_usab)
- **Reliability**: [karths/binary_classification_train_reli](https://huggingface.co/karths/binary_classification_train_reli)
- **Process**: [karths/binary_classification_train_process](https://huggingface.co/karths/binary_classification_train_process)
- **Build**: [karths/binary_classification_train_build](https://huggingface.co/karths/binary_classification_train_build)
- **Maintenance**: [karths/binary_classification_train_main](https://huggingface.co/karths/binary_classification_train_main)
- **Automation**: [karths/binary_classification_train_automation](https://huggingface.co/karths/binary_classification_train_automation)
- **People**: [karths/binary_classification_train_people](https://huggingface.co/karths/binary_classification_train_people)
- **Portability**: [karths/binary_classification_train_port](https://huggingface.co/karths/binary_classification_train_port)

### Using Hugging Face Datasets and Models

You can use these datasets and models directly with our suite:

```bash
# Using a Hugging Face dataset
tdsuite-train \
    --data_file "karths/binary-10IQR-TD" \
    --model_name "distilbert-base-uncased" \
    --positive_category "TD" \
    --output_dir "outputs/binary" \
    --is_huggingface_dataset

# Using a pre-trained model from Hugging Face
tdsuite-inference \
    --model_name "karths/binary_classification_train_TD" \
    --input_file "data/split/test.csv"
```

## Usage

### Data Preparation

The package supports loading data from both local files and Hugging Face datasets:

#### Local Files
The package expects data in CSV, JSON, or JSONL format with at least two columns:
- `text`: The text content to classify
- `label`: The class label (0 or 1 for binary classification)

Example CSV data format with numeric labels:
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

### Training

Train a model to classify text as either technical debt or not:

```bash
# Using a Hugging Face dataset with text labels
tdsuite-train \
    --data_file "karths/binary-10IQR-people" \
    --model_name "distilbert-base-uncased" \
    --positive_category "people" \
    --output_dir "outputs/binary" \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --num_epochs 5 \
    --warmup_steps 1000 \
    --cross_validation \
    --n_splits 5

# Using a local data file with text labels
tdsuite-train \
    --data_file "data/split/train.csv" \
    --model_name "distilbert-base-uncased" \
    --positive_category "TD" \
    --output_dir "outputs/binary" \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --num_epochs 5 \
    --warmup_steps 1000
```

#### Training Arguments

The training script supports the following arguments:

- **Data Arguments**:
  - `--data_file`: Path to data file or Hugging Face dataset name (required)
  - `--text_column`: Name of the text column (default: "text")
  - `--label_column`: Name of the label column (default: "label")
  - `--is_huggingface_dataset`: Flag to indicate if data is from Hugging Face
  - `--numeric_labels`: Flag to indicate if labels are already numeric (0/1)
  - `--positive_category`: Positive category for binary classification

- **Model Arguments**:
  - `--model_name`: Name of the model to use (required)
  - `--max_length`: Maximum sequence length (default: 512)

- **Training Arguments**:
  - `--output_dir`: Directory to save the model and outputs (required)
  - `--num_epochs`: Number of training epochs (default: 3)
  - `--batch_size`: Batch size (default: 16)
  - `--learning_rate`: Learning rate (default: 2e-5)
  - `--weight_decay`: Weight decay (default: 0.01)
  - `--warmup_steps`: Number of warmup steps (default: 500)
  - `--gradient_accumulation_steps`: Number of gradient accumulation steps (default: 1)

- **Cross-validation Arguments**:
  - `--cross_validation`: Flag to enable cross-validation
  - `--n_splits`: Number of splits for cross-validation (default: 5)

- **Additional Arguments**:
  - `--seed`: Random seed (default: 42)
  - `--device`: Device to use (cuda, cpu, or None for auto-detection)

#### Cross-Validation

Cross-validation helps evaluate model performance more robustly:

```bash
tdsuite-train --data_file "karths/binary-10IQR-port" \
              --model_name "distilbert-base-uncased" \
              --positive_category "port" \
              --output_dir "outputs/cv" \
              --batch_size 16 \
              --learning_rate 1e-5 \
              --num_epochs 5 \
              --warmup_steps 1000 \
              --cross_validation \
              --n_splits 5
```

This will:
1. Split the data into 5 folds (or specified number of splits)
2. Train a model on each fold
3. Evaluate on the held-out fold
4. Aggregate the results across all folds
5. Generate visualizations of the cross-validation results
6. Save metrics and model checkpoints for each fold

The cross-validation output directory will contain:
- `fold_0/` to `fold_N/`: Individual fold results
  - `metrics.json`: Training and evaluation metrics
  - `confusion_matrix.png`: Confusion matrix visualization
  - `roc_curve.png`: ROC curve visualization
- `cross_validation_results.json`: Aggregated results across all folds
- `cross_validation_visualization.png`: Visualization of cross-validation performance

### Inference

#### Batch Inference

Process multiple texts from a file:

```bash
# Using a local model
tdsuite-inference \
    --model_path "outputs/binary" \
    --input_file "data/split/test.csv"

# Using a Hugging Face model
tdsuite-inference \
    --model_name "karths/binary_classification_train_TD" \
    --input_file "data/split/test.csv"

# Using a custom results directory
tdsuite-inference \
    --model_path "outputs/binary" \
    --input_file "data/split/test.csv" \
    --results_dir "custom_results"
```

The inference script automatically creates a timestamped directory for each run inside the model's output folder. For example, if you run inference with a model in `outputs/binary`, it will create a directory like `outputs/binary/inference_20240411_074955/` containing:

- `predictions_test.csv`: Model predictions
- `metrics/`: Directory containing evaluation results
  - `metrics.json`: Evaluation metrics
  - `confusion_matrix.png`: Confusion matrix visualization
  - `roc_curve.png`: ROC curve visualization

#### Ensemble Inference

You can combine multiple models into an ensemble for improved prediction accuracy:

```bash
# Using multiple local models
tdsuite-inference \
    --model_paths "outputs/model1" "outputs/model2" "outputs/model3" \
    --input_file "data/split/test.csv" \
    --weights 0.5 0.3 0.2

# Using multiple Hugging Face models
tdsuite-inference \
    --model_names "karths/binary_classification_train_TD" "binary_classification_train_architecture" \
    --input_file "data/split/test.csv" \
    --weights 0.6 0.4
    

```

Key features of ensemble inference:
- Combine multiple models for more robust predictions
- Assign custom weights to each model (optional)
- Mix local and Hugging Face models in a single ensemble
- Equal weights are applied automatically if not specified

The ensemble approach works by:
1. Loading each model independently
2. Running inference with each model for the input text
3. Combining the probability outputs using weighted averaging
4. Determining the final prediction based on the combined probabilities

#### Inference Arguments

The inference script supports the following arguments:

- **Model Arguments**:
  - `--model_path`: Path to a local model (required if not using model_name)
  - `--model_name`: Name of a model on Hugging Face (required if not using model_path)
  - `--model_paths`: Paths to multiple local models for ensemble
  - `--model_names`: Names of multiple Hugging Face models for ensemble
  - `--weights`: Weights for ensemble models (optional)

- **Input Arguments**:
  - `--text`: Single text to classify
  - `--input_file`: Path to a file with texts (CSV or JSON)
  - `--text_column`: Name of the text column (default: "text")

- **Output Arguments**:
  - `--output_file`: Path to save predictions (optional)
  - `--results_dir`: Custom directory to store results (optional)
  - `--max_length`: Maximum sequence length (default: 512)
  - `--batch_size`: Batch size for inference (default: 32)
  - `--device`: Device to use for inference (cuda, cpu, or None for auto-detection)

#### Results Organization

Each inference run creates a new timestamped directory with the following structure:

```
outputs/model_name/
└── inference_YYYYMMDD_HHMMSS/
    ├── predictions_[input_filename].csv
    └── metrics/
        ├── metrics.json
        ├── confusion_matrix.png
        └── roc_curve.png
```

This organization allows you to:
- Keep track of different inference runs chronologically
- Compare results across different runs
- Maintain a clean separation between model files and inference results
- Optionally specify a custom results directory with `--results_dir`

### Output Files

After training, the following files will be saved in the output directory:

- `pytorch_model.bin`: The trained model weights
- `model_config.json`: Model configuration
- `training_config.json`: Training configuration
- `data_config.json`: Data processing configuration
- `metrics.json`: Training and evaluation metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve visualization
- `carbon_emissions.json`: Carbon emissions data from training

After inference with ground truth labels, the following files will be saved:

- `predictions.csv`: Model predictions
- `metrics.json`: Evaluation metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve visualization

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

### Early Stopping

Early stopping prevents overfitting by stopping training when the model stops improving:

```bash
tdsuite-train --data_file "karths/binary-10IQR-port" \
              --model_name "karths/binary_classification_train_port" \
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
