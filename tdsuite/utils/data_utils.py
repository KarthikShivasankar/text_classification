"""Data utilities for technical debt classification."""

import pandas as pd
import os
from typing import Union, List, Dict, Any, Optional
from datasets import load_dataset


def load_dataset_from_file(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a file.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        DataFrame containing the dataset
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the dataset based on the file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext == '.json':
        return pd.read_json(file_path)
    elif file_ext == '.jsonl':
        return pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def load_dataset_from_huggingface(dataset_name: str, split: str = "train") -> pd.DataFrame:
    """
    Load a dataset from Hugging Face.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        split: Split of the dataset to load (default: "train")
        
    Returns:
        DataFrame containing the dataset
    """
    try:
        dataset = load_dataset(dataset_name, split=split)
        return dataset.to_pandas()
    except Exception as e:
        raise ValueError(f"Failed to load Hugging Face dataset: {str(e)}")


def load_dataset(data_source: str, split: str = "train") -> pd.DataFrame:
    """
    Load a dataset from a file or Hugging Face.
    
    Args:
        data_source: Path to the dataset file or name of the dataset on Hugging Face
        split: Split of the dataset to load (default: "train")
        
    Returns:
        DataFrame containing the dataset
    """
    # Check if the data source is a file path
    if os.path.exists(data_source):
        return load_dataset_from_file(data_source)
    else:
        # Assume it's a Hugging Face dataset
        return load_dataset_from_huggingface(data_source, split)


def preprocess_text(text: str, max_length: int = 512) -> str:
    """
    Preprocess text for model input.
    
    Args:
        text: Text to preprocess
        max_length: Maximum length of the text (default: 512)
        
    Returns:
        Preprocessed text
    """
    # Truncate text if it's too long
    if len(text) > max_length:
        text = text[:max_length]
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text 