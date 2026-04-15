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


def split_and_save_data(
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
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Split and save data for technical debt classification.

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

    Returns:
        Tuple of (train_df, test_df, top_repo_df)
    """
    splitter = DataSplitter(
        data_file=data_file,
        output_dir=output_dir,
        test_size=test_size,
        random_state=random_state,
        text_column=text_column,
        label_column=label_column,
        repo_column=repo_column,
        is_huggingface_dataset=is_huggingface_dataset,
        is_numeric_labels=is_numeric_labels,
        positive_category=positive_category,
    )
    
    return splitter.split_and_save() 