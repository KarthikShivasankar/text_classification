#!/usr/bin/env python
"""Extract and clean issue body text from a GitHub issues CSV.

Produces a CSV with a 'text' column ready for tdsuite-inference:

    python scripts/extract_issue_bodies.py --input issues.csv --output issue_texts.csv
    tdsuite-inference --model_name karths/binary_classification_train_TD \\
                      --input_file issue_texts.csv

Options:
    --min-length      Drop issues whose cleaned text is shorter than N characters.
    --keep-metadata   Also keep 'number' and 'title' columns for traceability.
    --drop-duplicates Remove duplicate body texts.
"""

import re
import sys
import unicodedata

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


def clean_text(text: str) -> str:
    """Aggressively clean a GitHub issue body down to human-written prose only."""
    if not isinstance(text, str):
        return ""

    # 1. Remove fenced code blocks (``` ... ``` and ~~~ ... ~~~), including language tag
    text = re.sub(r"```[\w]*\n?.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"~~~[\w]*\n?.*?~~~", " ", text, flags=re.DOTALL)

    # 2. Remove inline code (`...`)
    text = re.sub(r"`[^`\n]+`", " ", text)

    # 3. Remove HTML tags and HTML comments
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)

    # 4. Remove HTML entities (&amp; &lt; &gt; &#123; etc.)
    text = re.sub(r"&[a-zA-Z]+;|&#\d+;", " ", text)

    # 5. Remove blockquotes (lines starting with >)
    text = re.sub(r"^\s*>+.*$", "", text, flags=re.MULTILINE)

    # 6. Remove markdown images and links, keep only link text
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)  # ![alt](url)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [text](url) → text
    text = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", text)  # [text][ref] → text

    # 7. Remove raw URLs
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)

    # 8. Remove markdown headings markers (keep the heading text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # 9. Remove markdown emphasis (* ** _ __ ~~ ~~)
    text = re.sub(r"(\*{1,3}|_{1,3}|~{2})(.*?)\1", r"\2", text)

    # 10. Remove markdown horizontal rules
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # 11. Remove markdown checkboxes (- [ ] / - [x])
    text = re.sub(r"^\s*-\s*\[[ xX]\]\s*", "", text, flags=re.MULTILINE)

    # 12. Remove markdown list bullets and numbered list markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # 13. Remove emoji characters (covers all Unicode emoji / pictograph ranges)
    text = re.sub(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map
        "\U0001f1e0-\U0001f1ff"  # flags
        "\U00002700-\U000027bf"  # dingbats
        "\U0001f900-\U0001f9ff"  # supplemental symbols
        "\U00002600-\U000026ff"  # misc symbols
        "\U0001fa00-\U0001fa6f"  # chess / other
        "\U0001fa70-\U0001faff"  # food / drink
        "\U00002300-\U000023ff"  # misc technical
        "\U0001f700-\U0001f77f"  # alchemical
        "]+",
        " ",
        text,
    )

    # 14. Remove Unicode variation selectors and zero-width characters
    text = re.sub(r"[\uFE00-\uFE0F\u200B-\u200F\u00AD\uFEFF]", "", text)

    # 15. Normalise unicode (e.g. fancy quotes → ASCII)
    text = unicodedata.normalize("NFKC", text)

    # 16. Remove non-printable / control characters (keep newlines and tabs)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uD7FF\uE000-\uFFFD]", " ", text)

    # 17. Collapse runs of blank lines to a single newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 18. Collapse multiple spaces/tabs on a single line
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 19. Strip leading/trailing whitespace on each line and globally
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return text.strip()


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
    df = df[df[args.body_column].notna()].copy()

    # Clean text
    tqdm.write("Cleaning issue bodies...")
    df["text"] = [
        clean_text(body)
        for body in tqdm(df[args.body_column], desc="Cleaning", unit="issue")
    ]

    # Filter by minimum length after cleaning
    df = df[df["text"].str.len() >= args.min_length]
    tqdm.write(
        f"Rows: {total_before} → {len(df)} after cleaning "
        f"(min length {args.min_length} chars)"
    )

    if args.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["text"])
        tqdm.write(f"Rows: {before} → {len(df)} after deduplication")

    # Build output columns
    out_data: dict = {"text": df["text"].values}
    if args.keep_metadata:
        for col in ("number", "title"):
            if col in df.columns:
                out_data[col] = df[col].values

    out_df = pd.DataFrame(out_data)
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
