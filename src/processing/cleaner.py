"""Text cleaning and normalization module preserving raw review text for lineage."""

import html
import re
import unicodedata
from typing import Any, Optional
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("processing.cleaner")

# Regex to detect unprintable control characters (excluding standard whitespace)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Regex to collapse consecutive whitespace into a single space
_WHITESPACE_RE = re.compile(r"\s+")


class TextCleaner:
    """Normalizes review text, strips invisible control codes, unescapes HTML, and preserves originals."""

    def __init__(self, strip_html: bool = True, normalize_unicode: bool = True):
        self.strip_html = strip_html
        self.normalize_unicode = normalize_unicode

    def clean_text(self, text: Any) -> str:
        """Cleans and normalizes a single string.

        Args:
            text: Input text (or raw object).

        Returns:
            str: Cleaned and normalized text, or empty string if input is invalid/empty.
        """
        if text is None or pd.isna(text):
            return ""

        raw_str = str(text)

        # 1. Unicode NFKC normalization
        if self.normalize_unicode:
            raw_str = unicodedata.normalize("NFKC", raw_str)

        # 2. Decode HTML entities (e.g. &amp;, &quot;, &#39;)
        if self.strip_html:
            raw_str = html.unescape(raw_str)

        # 3. Strip non-printable control characters
        raw_str = _CONTROL_CHAR_RE.sub(" ", raw_str)

        # 4. Collapse multiple whitespace and strip edges
        cleaned = _WHITESPACE_RE.sub(" ", raw_str).strip()

        return cleaned

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        cleaned_column: str = "cleaned_text",
        original_column: str = "original_text",
    ) -> pd.DataFrame:
        """Normalizes text in a DataFrame batch while maintaining full audit lineage.

        Args:
            df: Input DataFrame.
            text_column: Name of the source text column.
            cleaned_column: Target column for cleaned text output.
            original_column: Target column preserving raw unmodified text.

        Returns:
            pd.DataFrame: DataFrame containing both preserved raw text and cleaned text.
        """
        if df.empty:
            logger.debug("Empty DataFrame provided to TextCleaner")
            result = df.copy()
            if text_column in result.columns:
                result[original_column] = result[text_column]
                result[cleaned_column] = result[text_column]
            return result

        if text_column not in df.columns:
            logger.warning("Target text column '%s' not found in DataFrame", text_column)
            return df.copy()

        df_out = df.copy()

        # Preserve original unadulterated text for audit and validation lineage
        if original_column not in df_out.columns:
            df_out[original_column] = df_out[text_column].copy()

        # Vectorized string cleaning apply
        df_out[cleaned_column] = df_out[text_column].apply(self.clean_text)

        # Keep primary 'text' column aligned with cleaned text
        df_out[text_column] = df_out[cleaned_column]

        logger.debug("Cleaned %d records in text column '%s'", len(df_out), text_column)
        return df_out

