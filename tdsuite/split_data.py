#!/usr/bin/env python
"""Script to split and save data for technical debt classification."""

import argparse
import os
from tdsuite.data.data_splitter import split_and_save_data


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Split and save data for technical debt classification.")
    
    # Required arguments
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file or Hugging Face dataset name")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the split data")
    
    # Optional arguments
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the column containing the text")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the column containing the labels")
    parser.add_argument("--repo_column", type=str, default=None, help="Name of the column containing the repository information")
    parser.add_argument("--is_huggingface_dataset", action="store_true", help="Whether the data file is a Hugging Face dataset")
    parser.add_argument("--is_numeric_labels", action="store_true", help="Whether the labels are numeric (0/1) or categorical")
    parser.add_argument("--positive_category", type=str, default=None, help="The category to consider as positive for binary classification")
    
    args = parser.parse_args()
    
    # Split and save the data
    train_df, test_df, top_repo_df = split_and_save_data(
        data_file=args.data_file,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        text_column=args.text_column,
        label_column=args.label_column,
        repo_column=args.repo_column,
        is_huggingface_dataset=args.is_huggingface_dataset,
        is_numeric_labels=args.is_numeric_labels,
        positive_category=args.positive_category,
    )
    
    # Print summary
    print(f"Data split and saved to {args.output_dir}")
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    if top_repo_df is not None:
        print(f"Top repositories: {len(top_repo_df)} samples")
    
    # Print class distribution
    print("\nClass distribution in training set:")
    print(train_df[args.label_column].value_counts(normalize=True))
    
    print("\nClass distribution in test set:")
    print(test_df[args.label_column].value_counts(normalize=True))
    
    if top_repo_df is not None:
        print("\nClass distribution in top repositories:")
        print(top_repo_df[args.label_column].value_counts(normalize=True))


if __name__ == "__main__":
    main() 