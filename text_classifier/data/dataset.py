"""Dataset classes for text classification."""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Dict, List, Tuple, Optional, Union, Any


class TextClassificationDataset(Dataset):
    """Base dataset class for text classification tasks."""

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
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.labels)


class DataProcessor:
    """Class to handle data processing for text classification."""

    def __init__(self, tokenizer, max_length=512):
        """
        Initialize the data processor.

        Args:
            tokenizer: The tokenizer to use
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def prepare_data(
        self,
        df: pd.DataFrame,
        text_col: str,
        label_col: str,
        test_size: float = 0.15,
        random_state: int = 42,
    ):
        """
        Prepare data for training and testing.

        Args:
            df: DataFrame containing the data
            text_col: Name of the column containing the text
            label_col: Name of the column containing the labels
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_data, test_data)
        """
        # Split the data
        train_data, test_data = train_test_split(
            df, test_size=test_size, stratify=df[label_col], random_state=random_state
        )

        return train_data, test_data

    def create_dataset(
        self, data: pd.DataFrame, text_col: str, label_col: str
    ) -> TextClassificationDataset:
        """
        Create a dataset from a DataFrame.

        Args:
            data: DataFrame containing the data
            text_col: Name of the column containing the text
            label_col: Name of the column containing the labels

        Returns:
            TextClassificationDataset
        """
        # Tokenize the texts
        encodings = self.tokenizer(
            data[text_col].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create the dataset
        dataset = TextClassificationDataset(encodings, data[label_col].values)

        return dataset

    def prepare_k_fold(
        self,
        data: pd.DataFrame,
        text_col: str,
        label_col: str,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        """
        Prepare data for k-fold cross-validation.

        Args:
            data: DataFrame containing the data
            text_col: Name of the column containing the text
            label_col: Name of the column containing the labels
            n_splits: Number of folds
            random_state: Random seed for reproducibility

        Returns:
            Generator of (train_data, val_data) for each fold
        """
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

        for train_index, val_index in skf.split(data[text_col], data[label_col]):
            train_data = data.iloc[train_index]
            val_data = data.iloc[val_index]

            yield train_data, val_data


class BinaryClassificationProcessor(DataProcessor):
    """Data processor for binary classification tasks."""

    def prepare_binary_data(
        self,
        df: pd.DataFrame,
        category: str,
        label_col: str = "label",
        text_col: str = "text",
        balance: bool = True,
    ):
        """
        Prepare data for binary classification.

        Args:
            df: DataFrame containing the data
            category: The positive category
            label_col: Name of the column containing the labels
            text_col: Name of the column containing the text
            balance: Whether to balance the classes

        Returns:
            DataFrame with binary labels
        """
        # Extract positive class
        pos_class_df = df[df[label_col] == category].copy()

        # Extract negative class
        neg_class_df = df[df[label_col] != category].copy()

        if balance:
            # Determine the number of negative instances to sample
            available_neg_instances = len(neg_class_df)
            num_neg_instances = min(len(pos_class_df), available_neg_instances)

            # Randomly sample negative instances
            neg_class_df = neg_class_df.sample(n=num_neg_instances, random_state=42)

        # Assign binary labels
        pos_class_df.loc[:, "binary_label"] = 1
        neg_class_df.loc[:, "binary_label"] = 0

        # Rename negative class
        neg_class_df.loc[:, label_col] = f"non_{category}"

        # Concatenate and shuffle
        binary_df = pd.concat([pos_class_df, neg_class_df], ignore_index=True)
        binary_df = binary_df.sample(frac=1, random_state=42).reset_index(drop=True)

        return binary_df

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
        filtered_data = df[df[label_col] == positive_label]

        # Get the count of each repository
        top_repos = filtered_data[repo_col].value_counts()

        # Find the top repository name
        top_repo_name = top_repos.idxmax()

        # Handle a tie situation
        if len(top_repos) > 1 and top_repos.max() == top_repos.iloc[1]:
            top_repo_name = top_repos.sample(1).index[0]

        # Extract top repo data
        top_repo_data = df[df[repo_col] == top_repo_name].copy()

        # Remove top repo data from main data
        remaining_data = df[df[repo_col] != top_repo_name].copy()

        return remaining_data, top_repo_data


class MultiClassificationProcessor(DataProcessor):
    """Data processor for multi-class classification tasks."""

    def prepare_multi_class_data(
        self,
        df: pd.DataFrame,
        categories: List[str],
        label_col: str = "label",
        text_col: str = "text",
    ):
        """
        Prepare data for multi-class classification.

        Args:
            df: DataFrame containing the data
            categories: List of categories
            label_col: Name of the column containing the labels
            text_col: Name of the column containing the text

        Returns:
            DataFrame with numerical labels
        """
        # Filter data to include only the specified categories
        filtered_df = df[df[label_col].isin(categories)].copy()

        # Create category to index mapping
        cat2idx = {category: idx for idx, category in enumerate(categories)}

        # Convert labels to indices
        filtered_df["label_idx"] = filtered_df[label_col].map(lambda x: cat2idx[x])

        return filtered_df, cat2idx

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
        # Identify top repositories for each category
        top_repos = (
            df.groupby([label_idx_col, repo_col]).size().reset_index(name="counts")
        )
        top_repos = top_repos.loc[top_repos.groupby(label_idx_col)["counts"].idxmax()]

        # Extract top repos and remove them from main data
        top_repo_data = pd.DataFrame()
        remaining_data = df.copy()

        for _, row in top_repos.iterrows():
            top_repo = row[repo_col]
            label_idx = row[label_idx_col]

            # Extract entries for the top repo
            top_repo_entries = remaining_data[
                (remaining_data[repo_col] == top_repo)
                & (remaining_data[label_idx_col] == label_idx)
            ]

            top_repo_data = pd.concat(
                [top_repo_data, top_repo_entries], ignore_index=True
            )

            # Remove top repo entries from main data
            remaining_data = remaining_data[
                ~(
                    (remaining_data[repo_col] == top_repo)
                    & (remaining_data[label_idx_col] == label_idx)
                )
            ]

        return remaining_data, top_repo_data
