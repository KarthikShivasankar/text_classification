#!/usr/bin/env python
"""Script to split and save data for technical debt classification."""

from tdsuite.cli import get_split_data_parser
from tdsuite.data.data_splitter import split_data


def main():
    """Main function."""
    args = get_split_data_parser().parse_args()

    # Split and save the data
    train_df, test_df, top_repo_df = split_data(
        data_file=args.data_file,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        repo_column=args.repo_column,
        is_huggingface_dataset=args.is_huggingface_dataset,
        is_numeric_labels=args.is_numeric_labels,
    )

    # Print summary
    print(f"Data split and saved to {args.output_dir}")
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    if top_repo_df is not None:
        print(f"Top repositories: {len(top_repo_df)} samples")

    # Print class distribution
    print("\nClass distribution in training set:")
    print(train_df["label"].value_counts(normalize=True))

    print("\nClass distribution in test set:")
    print(test_df["label"].value_counts(normalize=True))

    if top_repo_df is not None:
        print("\nClass distribution in top repositories:")
        print(top_repo_df["label"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
