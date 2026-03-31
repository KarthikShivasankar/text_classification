"""Tests for scripts/extract_issue_bodies.py — clean_text() and pipeline logic."""

import sys
import os

import pandas as pd
import pytest

# Make scripts/ importable without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from extract_issue_bodies import clean_text


# ---------------------------------------------------------------------------
# clean_text — code removal
# ---------------------------------------------------------------------------

class TestCleanTextCodeBlocks:
    def test_fenced_code_block_removed(self):
        text = "Some text\n```python\nx = 1\n```\nMore text"
        result = clean_text(text)
        assert "x = 1" not in result
        assert "Some text" in result
        assert "More text" in result

    def test_inline_code_removed(self):
        result = clean_text("Use `os.path.join()` for paths")
        assert "os.path.join()" not in result
        assert "for paths" in result

    def test_tilde_code_block_removed(self):
        text = "Intro\n~~~\ncode here\n~~~\nOutro"
        result = clean_text(text)
        assert "code here" not in result


# ---------------------------------------------------------------------------
# clean_text — HTML / markdown
# ---------------------------------------------------------------------------

class TestCleanTextHtml:
    def test_html_tags_removed(self):
        result = clean_text("<b>bold</b> text")
        assert "<b>" not in result
        assert "bold" in result

    def test_html_comment_removed(self):
        result = clean_text("<!-- This is a comment -->visible")
        assert "This is a comment" not in result
        assert "visible" in result

    def test_html_entities_removed(self):
        result = clean_text("A &amp; B &lt;C&gt;")
        assert "&amp;" not in result
        assert "&lt;" not in result


class TestCleanTextMarkdown:
    def test_image_removed(self):
        result = clean_text("Look at this: ![screenshot](http://example.com/img.png)")
        assert "screenshot" not in result
        assert "http" not in result

    def test_link_text_preserved(self):
        result = clean_text("See [the docs](https://docs.example.com) for details")
        assert "the docs" in result
        assert "https://" not in result

    def test_raw_url_removed(self):
        result = clean_text("Go to https://example.com for more info")
        assert "https://" not in result
        assert "for more info" in result

    def test_heading_markers_removed(self):
        result = clean_text("## Section Title\nSome content")
        assert "##" not in result
        assert "Section Title" in result

    def test_bold_markers_removed(self):
        result = clean_text("This is **important** text")
        assert "**" not in result
        assert "important" in result

    def test_list_bullets_removed(self):
        result = clean_text("- item one\n- item two")
        assert "- item" not in result
        assert "item one" in result

    def test_numbered_list_markers_removed(self):
        result = clean_text("1. First\n2. Second")
        assert "1." not in result
        assert "First" in result

    def test_blockquote_removed(self):
        result = clean_text("> Quoted text\nNormal text")
        assert "Quoted text" not in result
        assert "Normal text" in result

    def test_horizontal_rule_removed(self):
        result = clean_text("Before\n---\nAfter")
        assert "---" not in result

    def test_checkbox_removed(self):
        result = clean_text("- [x] Done\n- [ ] Pending")
        assert "[x]" not in result
        assert "[ ]" not in result
        assert "Done" in result


# ---------------------------------------------------------------------------
# clean_text — emoji / whitespace
# ---------------------------------------------------------------------------

class TestCleanTextEmojiAndWhitespace:
    def test_emoji_removed(self):
        result = clean_text("Great work \U0001F600 keep going")
        assert "\U0001F600" not in result
        assert "Great work" in result

    def test_multiple_blank_lines_collapsed(self):
        result = clean_text("Line one\n\n\n\n\nLine two")
        assert "\n\n\n" not in result

    def test_extra_spaces_collapsed(self):
        result = clean_text("word1   word2    word3")
        assert "  " not in result

    def test_non_string_returns_empty(self):
        assert clean_text(None) == ""
        assert clean_text(42) == ""

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_leading_trailing_whitespace_stripped(self):
        result = clean_text("   hello world   ")
        assert result == result.strip()


# ---------------------------------------------------------------------------
# clean_text — plain prose is preserved
# ---------------------------------------------------------------------------

class TestCleanTextProsePreservation:
    def test_plain_prose_preserved(self):
        prose = "This module has high technical debt because it lacks proper abstractions."
        result = clean_text(prose)
        assert "technical debt" in result
        assert "abstractions" in result

    def test_multi_sentence_preserved(self):
        text = "The bug occurs on Windows. It crashes when the path has spaces."
        result = clean_text(text)
        assert "bug occurs" in result
        assert "crashes" in result


# ---------------------------------------------------------------------------
# Pipeline integration — CSV in / CSV out
# ---------------------------------------------------------------------------

class TestExtractPipeline:
    def test_clean_pipeline_produces_text_column(self, tmp_path):
        """Simulate the main() pipeline without subprocess."""
        df = pd.DataFrame({
            "body": [
                "This is a **real** issue with `inline code`.",
                "## Bug Report\nThe app crashes when clicking submit.",
                None,
                "short",  # below default min_length=20 → filtered out
            ]
        })
        csv_in = str(tmp_path / "issues.csv")
        df.to_csv(csv_in, index=False)

        # Replicate the core pipeline steps
        df_loaded = pd.read_csv(csv_in)
        df_loaded = df_loaded[df_loaded["body"].notna()]
        df_loaded["text"] = [clean_text(b) for b in df_loaded["body"]]
        df_loaded = df_loaded[df_loaded["text"].str.len() >= 20]

        assert "text" in df_loaded.columns
        # The None row and "short" row should be filtered
        assert len(df_loaded) == 2
        # Markdown should be cleaned
        for text in df_loaded["text"]:
            assert "**" not in text
            assert "`" not in text
            assert "##" not in text

    def test_drop_duplicates(self, tmp_path):
        df = pd.DataFrame({
            "body": ["Duplicate issue text here."] * 3 + ["Unique issue text here."]
        })
        df["text"] = [clean_text(b) for b in df["body"]]
        deduped = df.drop_duplicates(subset=["text"])
        assert len(deduped) == 2
