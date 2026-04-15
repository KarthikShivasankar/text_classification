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
    """Base class for technical debt data processing."""

    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, data_file, text_column="text", label_column="label"):
        """Load data from file or Hugging Face dataset."""
        if isinstance(data_file, str) and os.path.exists(data_file):
            # Load from local file
            if data_file.endswith(".csv"):
                df = pd.read_csv(data_file)
            elif data_file.endswith(".json") or data_file.endswith(".jsonl"):
                df = pd.read_json(data_file, lines=data_file.endswith(".jsonl"))
            else:
                raise ValueError(f"Unsupported file format: {data_file}")
        else:
            # Load from Hugging Face dataset
            dataset = load_dataset(data_file)
            df = pd.DataFrame(dataset["train"])
        
        return df

    def create_dataset(self, data, text_column="text", label_column="label"):
        """Create a PyTorch dataset from the data."""
        texts = data[text_column].tolist()
        labels = data[label_column].tolist()
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create dataset
        class TDDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)
        
        return TDDataset(encodings, labels)


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
            DataFrame with numeric labels (0 or 1)
        """
        # Create a copy of the data
        df = data.copy()

        if numeric_labels:
            # If labels are already numeric, just ensure they're integers
            df["label_idx"] = df[label_column].astype(int)
        else:
            if positive_category is None:
                raise ValueError("positive_category must be specified when numeric_labels=False")
            
            # Convert labels to binary (0 or 1)
            df["label_idx"] = (df[label_column] == positive_category).astype(int)

        return df

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