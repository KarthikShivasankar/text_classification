"""Data splitting and saving utilities for technical debt classification."""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset
import json


class DataSplitter:
    """Class for splitting and saving data for technical debt classification."""

    def __init__(
        self,
        data_file: str,
        output_dir: str,
        test_size: float = 0.2,
        random_state: int = 42,
        text_column: str = "text",
        label_column: str = "label",
        repo_column: Optional[str] = None,
        is_huggingface_dataset: bool = False,
        is_numeric_labels: bool = False,
        positive_category: Optional[str] = None,
    ):
        """
        Initialize the data splitter.

        Args:
            data_file: Path to the data file or Hugging Face dataset name
            output_dir: Directory to save the split data
            test_size: Proportion of the dataset to include in the test split
            random_state: Random state for reproducibility
            text_column: Name of the column containing the text
            label_column: Name of the column containing the labels
            repo_column: Name of the column containing the repository information
            is_huggingface_dataset: Whether the data file is a Hugging Face dataset
            is_numeric_labels: Whether the labels are numeric (0/1) or categorical
            positive_category: The category to consider as positive for binary classification
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state
        self.text_column = text_column
        self.label_column = label_column
        self.repo_column = repo_column
        self.is_huggingface_dataset = is_huggingface_dataset
        self.is_numeric_labels = is_numeric_labels
        self.positive_category = positive_category

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def load_data(self):
        """Load data from file or Hugging Face dataset."""
        if self.is_huggingface_dataset:
            try:
                from datasets import load_dataset
                dataset = load_dataset(self.data_file)
                if isinstance(dataset, dict):
                    # If dataset is a dict, use the first split
                    split_name = list(dataset.keys())[0]
                    df = dataset[split_name].to_pandas()
                else:
                    # If dataset is a Dataset, convert directly
                    df = dataset.to_pandas()
                return df
            except Exception as e:
                raise ValueError(f"Failed to load Hugging Face dataset: {str(e)}")
        
        # Check if the file exists
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"File not found: {self.data_file}")
        
        # Load from local file
        file_ext = os.path.splitext(self.data_file)[1].lower()
        if file_ext == '.csv':
            return pd.read_csv(self.data_file)
        elif file_ext == '.json':
            return pd.read_json(self.data_file)
        elif file_ext == '.jsonl':
            return pd.read_json(self.data_file, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {self.data_file}")

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data.

        Args:
            df: DataFrame containing the data

        Returns:
            Preprocessed DataFrame
        """
        # Convert labels to numeric if needed
        if not self.is_numeric_labels and self.positive_category is not None:
            # Binary classification with categorical labels
            df[self.label_column] = (df[self.label_column] == self.positive_category).astype(int)
        elif not self.is_numeric_labels:
            # Multi-class classification with categorical labels
            unique_labels = df[self.label_column].unique()
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            df[self.label_column] = df[self.label_column].map(label_map)

        return df

    def balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the classes in the dataset.

        Args:
            df: DataFrame containing the data

        Returns:
            Balanced DataFrame
        """
        # Get the class counts
        class_counts = df[self.label_column].value_counts()
        
        # Find the minimum class count
        min_count = class_counts.min()
        
        # Balance the classes
        balanced_dfs = []
        for label in class_counts.index:
            label_df = df[df[self.label_column] == label]
            if len(label_df) > min_count:
                # Randomly sample to match the minimum count
                label_df = label_df.sample(n=min_count, random_state=self.random_state)
            balanced_dfs.append(label_df)
        
        # Concatenate the balanced dataframes
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Shuffle the balanced dataframe
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        return balanced_df

    def split_and_save(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Split the data into training and test sets, and extract top repositories.
        First identifies the top repo with most positive samples, extracts all its data,
        removes it from the main dataset, then splits the remaining data into train and test.
        Finally, balances the top repository data by copying negative samples from test.csv if needed.

        Returns:
            Tuple of (train_df, test_df, top_repo_df)
        """
        # Load the data
        df = self.load_data()
        
        # Preprocess the data
        df = self.preprocess_data(df)
        
        # Identify the top repository with most positive samples
        top_repo = None
        top_repo_df = None
        if self.repo_column is not None:
            # Get the positive class label
            positive_label = 1 if self.is_numeric_labels else self.positive_category
            
            # Filter for positive class samples
            positive_df = df[df[self.label_column] == positive_label]
            
            # Count the number of positive samples per repository
            repo_counts = positive_df[self.repo_column].value_counts()
            
            # Get the single top repository (the one with the most positive samples)
            if not repo_counts.empty:
                top_repo = repo_counts.index[0]
                print(f"Top repository with most positive samples: {top_repo}")
                
                # Extract all samples from the top repository
                top_repo_df = df[df[self.repo_column] == top_repo].copy()
                
                # Remove the top repository data from the main dataset
                df = df[df[self.repo_column] != top_repo].copy()
                
                print(f"Extracted {len(top_repo_df)} samples from top repository")
                print(f"Remaining dataset has {len(df)} samples")
        
        # Balance the classes in the remaining dataset
        balanced_df = self.balance_classes(df)
        
        # Split into training and test sets
        train_df, test_df = train_test_split(
            balanced_df, test_size=self.test_size, random_state=self.random_state, stratify=balanced_df[self.label_column]
        )
        
        # Save the initial data
        train_df.to_csv(os.path.join(self.output_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(self.output_dir, "test.csv"), index=False)
        
        # Balance the top repository data if needed
        if top_repo_df is not None:
            # Get the positive class label
            positive_label = 1 if self.is_numeric_labels else self.positive_category
            
            # Count positive and negative samples in top_repo_df
            positive_count = len(top_repo_df[top_repo_df[self.label_column] == positive_label])
            negative_count = len(top_repo_df[top_repo_df[self.label_column] != positive_label])
            
            print(f"Top repository data: {positive_count} positive samples, {negative_count} negative samples")
            
            # If the top repository data is not balanced, copy negative samples from test.csv
            if positive_count > negative_count:
                # Get negative samples from test.csv
                test_negative = test_df[test_df[self.label_column] != positive_label].copy()
                
                # Calculate how many negative samples we need
                needed_negative = positive_count - negative_count
                
                # Sample negative samples from test.csv
                if len(test_negative) >= needed_negative:
                    additional_negative = test_negative.sample(n=needed_negative, random_state=self.random_state)
                    
                    # Add the additional negative samples to top_repo_df
                    top_repo_df = pd.concat([top_repo_df, additional_negative], ignore_index=True)
                    
                    # Remove the additional negative samples from test_df
                    test_df = test_df[~test_df.index.isin(additional_negative.index)]
                    
                    print(f"Added {needed_negative} negative samples from test.csv to top_repos.csv")
                    print(f"Updated test.csv has {len(test_df)} samples")
                else:
                    print(f"Warning: Not enough negative samples in test.csv to balance top_repos.csv")
            
            # Save the balanced top repository data
            top_repo_df.to_csv(os.path.join(self.output_dir, "top_repos.csv"), index=False)
            
            # Save the updated test data
            test_df.to_csv(os.path.join(self.output_dir, "test.csv"), index=False)
        
        # Save the label mappings
        if not self.is_numeric_labels:
            unique_labels = df[self.label_column].unique()
            label_map = {idx: label for idx, label in enumerate(unique_labels)}
            with open(os.path.join(self.output_dir, "label_mappings.json"), "w") as f:
                json.dump(label_map, f, indent=4)
        
        return train_df, test_df, top_repo_df


def split_data(
    data_file: str,
    output_dir: str,
    is_numeric_labels: bool = False,
    repo_column: Optional[str] = None,
    is_huggingface_dataset: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Split data into training and test sets with balanced classes.
    
    Args:
        data_file: Path to data file or Hugging Face dataset name
        output_dir: Directory to save split data
        is_numeric_labels: Whether the labels are already numeric (0 or 1)
        repo_column: Name of the repository column (optional)
        is_huggingface_dataset: Whether the data is a Hugging Face dataset
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    """
    # Load data
    if is_huggingface_dataset:
        dataset = load_dataset(data_file)
        df = pd.DataFrame(dataset["train"])
    else:
        if data_file.endswith(".csv"):
            df = pd.read_csv(data_file)
        elif data_file.endswith(".json") or data_file.endswith(".jsonl"):
            df = pd.read_json(data_file, lines=data_file.endswith(".jsonl"))
        else:
            raise ValueError(f"Unsupported file format: {data_file}")
    
    # Ensure labels are numeric
    if not is_numeric_labels:
        # Find the positive category (the one that doesn't start with "non_")
        unique_labels = df["label"].unique()
        if len(unique_labels) != 2:
            raise ValueError("Binary classification requires exactly 2 unique labels")
        
        positive_label = None
        non_label = None
        for label in unique_labels:
            if str(label).startswith("non_"):
                non_label = label
            else:
                positive_label = label
        
        # If we couldn't identify the labels by prefix, use the first one as non_ and second as positive
        if non_label is None or positive_label is None:
            sorted_labels = sorted(unique_labels)
            non_label = sorted_labels[0]
            positive_label = sorted_labels[1]
        
        # Create mapping: non_ -> 0, positive -> 1
        label_map = {non_label: 0, positive_label: 1}
        df["label"] = df["label"].map(label_map)
    
    # Split data
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]
    )
    
    # Extract top repositories if repo_column is provided
    top_repos_df = None
    if repo_column:
        # Count positive samples per repository
        repo_counts = df[df["label"] == 1].groupby(repo_column).size()
        top_repos = repo_counts.nlargest(10).index.tolist()
        
        # Filter data to include only top repositories
        top_repos_df = df[df[repo_column].isin(top_repos)]
        
        # Balance classes in top repositories data
        pos_samples = top_repos_df[top_repos_df["label"] == 1]
        neg_samples = top_repos_df[top_repos_df["label"] == 0]
        
        # Get the minimum number of samples between positive and negative
        min_samples = min(len(pos_samples), len(neg_samples))
        
        if min_samples > 0:
            # Sample equal number of positive and negative samples
            pos_samples = pos_samples.sample(n=min_samples, random_state=random_state)
            neg_samples = neg_samples.sample(n=min_samples, random_state=random_state)
            top_repos_df = pd.concat([pos_samples, neg_samples])
        else:
            # If one class is empty, just use the available samples
            top_repos_df = pd.concat([pos_samples, neg_samples])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save split data
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    if top_repos_df is not None:
        top_repos_df.to_csv(os.path.join(output_dir, "top_repos.csv"), index=False)
    
    return train_df, test_df, top_repos_df 