"""Dataset classes for technical debt classification."""

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Dict, List, Tuple, Optional, Union, Any
from datasets import load_dataset
from transformers import PreTrainedTokenizer


class TDDataset(Dataset):
    """Base dataset class for technical debt classification tasks."""

    def __init__(self, encodings, labels):
        """
        Initialize the dataset.

        Args:
            encodings: The tokenized inputs
            labels: The corresponding labels
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        item = {}
        for key, val in self.encodings.items():
            if isinstance(val[idx], torch.Tensor):
                item[key] = val[idx].clone().detach()
            else:
                item[key] = torch.tensor(val[idx])
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.labels)


class TDProcessor:
    """Base processor for technical debt classification data."""

    def __init__(self, tokenizer, max_length=512):
        """
        Initialize the processor.

        Args:
            tokenizer: The tokenizer to use
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def create_dataset(self, data, text_column, label_column):
        """
        Create a dataset from the data.

        Args:
            data: The data to process
            text_column: The column containing the text
            label_column: The column containing the labels

        Returns:
            TDDataset
        """
        # Tokenize the texts
        encodings = self.tokenizer(
            data[text_column].tolist(),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Convert labels to integers
        labels = data[label_column].tolist()

        # Create dataset
        dataset = TDDataset(encodings, labels)
        return dataset

    @classmethod
    def load_data(cls, data_source, text_column="text", label_column="label", **kwargs):
        """
        Load data from a source (local file or Hugging Face dataset).
        
        Args:
            data_source: Path to a local file or name of a Hugging Face dataset
            text_column: The column containing the text
            label_column: The column containing the labels
            **kwargs: Additional arguments for loading the data
            
        Returns:
            DataFrame containing the data
        """
        # Check if data_source is a local file
        if os.path.exists(data_source):
            # Load from local file
            if data_source.endswith('.csv'):
                data = pd.read_csv(data_source)
            elif data_source.endswith('.json'):
                data = pd.read_json(data_source)
            elif data_source.endswith('.jsonl'):
                data = pd.read_json(data_source, lines=True)
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
        else:
            # Load from Hugging Face
            try:
                # Try to load as a dataset
                dataset = load_dataset(data_source, **kwargs)
                
                # Convert to pandas DataFrame
                if isinstance(dataset, dict):
                    # If it's a dict of datasets, use the first one
                    data = dataset[list(dataset.keys())[0]].to_pandas()
                else:
                    data = dataset.to_pandas()
            except Exception as e:
                raise ValueError(f"Failed to load dataset from Hugging Face: {e}")
        
        # Verify required columns exist
        if text_column not in data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")
        
        return data


class BinaryTDProcessor(TDProcessor):
    """Processor for binary technical debt classification."""

    def prepare_binary_data(self, data, positive_category=None, text_column="text", label_column="label", numeric_labels=False):
        """
        Prepare data for binary classification.

        Args:
            data: The data to process
            positive_category: The category to consider as positive (ignored if numeric_labels=True)
            text_column: The column containing the text
            label_column: The column containing the labels
            numeric_labels: Whether the labels are already numeric (0 or 1)

        Returns:
            DataFrame with binary labels
        """
        # Create a copy of the data
        df = data.copy()

        if numeric_labels:
            # If labels are already numeric, just ensure they're integers
            df["label_idx"] = df[label_column].astype(int)
        else:
            # Convert text labels to binary
            if positive_category is None:
                raise ValueError("positive_category must be specified when numeric_labels=False")
            df["label_idx"] = (df[label_column] == positive_category).astype(int)

        return df


class MultiTDProcessor(TDProcessor):
    """Processor for multi-class technical debt classification."""

    def prepare_multi_class_data(self, data, categories=None, text_column="text", label_column="label", numeric_labels=False, num_classes=None):
        """
        Prepare data for multi-class classification.

        Args:
            data: The data to process
            categories: List of categories to include (ignored if numeric_labels=True)
            text_column: The column containing the text
            label_column: The column containing the labels
            numeric_labels: Whether the labels are already numeric (0, 1, 2, etc.)
            num_classes: Number of classes (required if numeric_labels=True)

        Returns:
            DataFrame with numeric labels
        """
        # Create a copy of the data
        df = data.copy()

        if numeric_labels:
            # If labels are already numeric, just ensure they're integers
            if num_classes is None:
                raise ValueError("num_classes must be specified when numeric_labels=True")
            
            # Ensure labels are integers
            df["label_idx"] = df[label_column].astype(int)
            
            # Create dummy mappings for compatibility
            cat2idx = {str(i): i for i in range(num_classes)}
            idx2cat = {i: str(i) for i in range(num_classes)}
        else:
            # Filter data to include only the specified categories
            if categories is None:
                raise ValueError("categories must be specified when numeric_labels=False")
            
            df = df[df[label_column].isin(categories)]

            # Create a mapping from categories to indices
            cat2idx = {cat: idx for idx, cat in enumerate(categories)}
            idx2cat = {idx: cat for idx, cat in enumerate(categories)}

            # Convert labels to indices
            df["label_idx"] = df[label_column].map(cat2idx)

        return df, cat2idx, idx2cat

    def prepare_data(
        self,
        data: Union[pd.DataFrame, str],
        text_col: str = "text",
        label_col: str = "label",
        test_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[TDDataset, TDDataset]:
        """
        Prepare data for training and evaluation.

        Args:
            data: Either a DataFrame or a Hugging Face dataset name
            text_col: Name of the text column
            label_col: Name of the label column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if isinstance(data, str):
            df = self.load_data(data, text_col, label_col)
        else:
            df = data

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df[label_col]
        )

        train_dataset = self.create_dataset(train_df, text_col, label_col)
        test_dataset = self.create_dataset(test_df, text_col, label_col)

        return train_dataset, test_dataset

    def prepare_k_fold(
        self,
        data: Union[pd.DataFrame, str],
        text_col: str = "text",
        label_col: str = "label",
        n_splits: int = 5,
        random_state: int = 42,
    ) -> List[Tuple[TDDataset, TDDataset]]:
        """
        Prepare data for k-fold cross-validation.

        Args:
            data: Either a DataFrame or a Hugging Face dataset name
            text_col: Name of the text column
            label_col: Name of the label column
            n_splits: Number of folds
            random_state: Random seed for reproducibility

        Returns:
            List of (train_dataset, val_dataset) tuples for each fold
        """
        if isinstance(data, str):
            df = self.load_data(data, text_col, label_col)
        else:
            df = data

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = []

        for train_idx, val_idx in skf.split(df[text_col], df[label_col]):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            train_dataset = self.create_dataset(train_df, text_col, label_col)
            val_dataset = self.create_dataset(val_df, text_col, label_col)

            folds.append((train_dataset, val_dataset))

        return folds

    def extract_top_repo(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        repo_col: str = "repo",
        positive_label: int = 1,
    ):
        """
        Extract data from the top repository for a given label.

        Args:
            df: DataFrame containing the data
            label_col: Name of the column containing the labels
            repo_col: Name of the column containing the repository
            positive_label: The positive label value

        Returns:
            Tuple of (remaining_data, top_repo_data)
        """
        # Filter for positive class
        pos_df = df[df[label_col] == positive_label].copy()

        # Get repository counts
        repo_counts = pos_df[repo_col].value_counts()

        if len(repo_counts) == 0:
            return df, pd.DataFrame()

        # Get top repository
        top_repo = repo_counts.index[0]

        # Split data
        top_repo_data = pos_df[pos_df[repo_col] == top_repo].copy()
        remaining_data = df[df[repo_col] != top_repo].copy()

        return remaining_data, top_repo_data

    def extract_top_repos_by_category(
        self, df: pd.DataFrame, label_idx_col: str = "label_idx", repo_col: str = "repo"
    ):
        """
        Extract data from the top repository for each category.

        Args:
            df: DataFrame containing the data
            label_idx_col: Name of the column containing the label indices
            repo_col: Name of the column containing the repository

        Returns:
            Tuple of (remaining_data, top_repo_data)
        """
        top_repo_data = pd.DataFrame()
        remaining_data = df.copy()

        for label_idx in df[label_idx_col].unique():
            # Filter for current category
            category_df = df[df[label_idx_col] == label_idx].copy()

            # Get repository counts
            repo_counts = category_df[repo_col].value_counts()

            if len(repo_counts) == 0:
                continue

            # Get top repository
            top_repo = repo_counts.index[0]

            # Split data
            category_top_repo = category_df[category_df[repo_col] == top_repo].copy()
            category_remaining = category_df[category_df[repo_col] != top_repo].copy()

            # Update dataframes
            top_repo_data = pd.concat([top_repo_data, category_top_repo], ignore_index=True)
            remaining_data = pd.concat([remaining_data, category_remaining], ignore_index=True)

        return remaining_data, top_repo_data 