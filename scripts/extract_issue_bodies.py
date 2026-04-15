#!/usr/bin/env python
"""Extract and clean issue body text from a GitHub issues CSV.

Produces a CSV with a 'text' column ready for tdsuite-inference:

    python scripts/extract_issue_bodies.py --input issues.csv --output issue_texts.csv
    tdsuite-inference --model_name karths/binary_classification_train_TD \\
                      --input_file issue_texts.csv

Options:
    --min-length   Drop issues whose body is shorter than N characters.
    --keep-metadata  Also keep 'number' and 'title' columns for traceability.
    --drop-duplicates  Remove duplicate body texts.
"""

import sys

import pandas as pd
from tqdm import tqdm


def _get_parser():
    try:
        from tdsuite.cli import get_extract_issues_parser
        return get_extract_issues_parser()
    except ImportError:
        import argparse

        p = argparse.ArgumentParser(
            description="Extract issue body text from a GitHub issues CSV"
        )
        p.add_argument("--input", required=True)
        p.add_argument("--output", default="issue_texts.csv")
        p.add_argument("--body-column", default="body", dest="body_column")
        p.add_argument("--min-length", type=int, default=20, dest="min_length")
        p.add_argument("--keep-metadata", action="store_true", dest="keep_metadata")
        p.add_argument("--drop-duplicates", action="store_true", dest="drop_duplicates")
        return p


def main():
    args = _get_parser().parse_args()

    print(f"Reading {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.body_column not in df.columns:
        print(
            f"Error: column '{args.body_column}' not found. "
            f"Available columns: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    total_before = len(df)

    # Drop rows with no body text
    df = df[df[args.body_column].notna()]
    df = df[df[args.body_column].str.strip().str.len() >= args.min_length]
    tqdm.write(
        f"Rows: {total_before} → {len(df)} after filtering "
        f"(min length {args.min_length} chars)"
    )

    if args.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=[args.body_column])
        tqdm.write(f"Rows: {before} → {len(df)} after deduplication")

    # Build output columns
    out_data = {"text": df[args.body_column].str.strip().values}
    if args.keep_metadata:
        for col in ("number", "title"):
            if col in df.columns:
                out_data[col] = df[col].values

    out_df = pd.DataFrame(out_data)

    # Show progress while iterating
    rows = [row for row in tqdm(out_df.itertuples(index=False), total=len(out_df), desc="Processing")]
    out_df = pd.DataFrame(rows, columns=out_df.columns)

    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")
    print(
        f"\nNext step — run inference:\n"
        f"  tdsuite-inference \\\n"
        f"    --model_name karths/binary_classification_train_TD \\\n"
        f"    --input_file {args.output}"
    )


if __name__ == "__main__":
    main()
