"""Unit tests for the processing layer: ReviewValidator, TextCleaner, and ReviewDeduplicator."""

import numpy as np
import pandas as pd
import pytest

from src.processing.cleaner import TextCleaner
from src.processing.deduplicator import ReviewDeduplicator
from src.processing.validator import ReviewValidator


# ============================================================================
# ReviewValidator Tests
# ============================================================================


def test_validator_all_valid_records():
    df = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "business_id": "b1",
                "stars": 5.0,
                "text": "Excellent food and service!",
                "date": "2023-01-15 14:30:00",
            },
            {
                "review_id": "r2",
                "business_id": "b2",
                "stars": 2,
                "text": "Food was cold.",
                "date": "2023-02-10",
            },
        ]
    )
    validator = ReviewValidator()
    result = validator.validate_batch(df)

    assert result.total_records == 2
    assert result.valid_count == 2
    assert result.invalid_count == 0
    assert result.is_fully_valid is True
    assert len(result.valid_df) == 2
    assert len(result.invalid_df) == 0


def test_validator_missing_required_columns():
    df = pd.DataFrame([{"review_id": "r1", "text": "Incomplete schema"}])
    validator = ReviewValidator()
    result = validator.validate_batch(df)

    assert result.valid_count == 0
    assert result.invalid_count == 1
    assert "Missing required columns" in result.invalid_df.iloc[0]["validation_errors"][0]


def test_validator_rating_alias_for_stars():
    df = pd.DataFrame(
        [
            {
                "review_id": "r1",
                "business_id": "b1",
                "rating": 4.5,  # uses 'rating' instead of 'stars'
                "text": "Great place!",
                "date": "2023-05-01",
            }
        ]
    )
    validator = ReviewValidator()
    result = validator.validate_batch(df)

    assert result.valid_count == 1
    assert result.invalid_count == 0
    assert result.valid_df.iloc[0]["stars"] == 4.5


def test_validator_quarantines_invalid_fields():
    df = pd.DataFrame(
        [
            # Invalid review_id
            {
                "review_id": "",
                "business_id": "b1",
                "stars": 4,
                "text": "Valid text",
                "date": "2023-01-01",
            },
            # Invalid business_id (null)
            {
                "review_id": "r2",
                "business_id": None,
                "stars": 4,
                "text": "Valid text",
                "date": "2023-01-01",
            },
            # Invalid stars (> 5.0)
            {
                "review_id": "r3",
                "business_id": "b1",
                "stars": 6.5,
                "text": "Valid text",
                "date": "2023-01-01",
            },
            # Invalid stars (negative/zero)
            {
                "review_id": "r4",
                "business_id": "b1",
                "stars": 0,
                "text": "Valid text",
                "date": "2023-01-01",
            },
            # Empty text
            {
                "review_id": "r5",
                "business_id": "b1",
                "stars": 3,
                "text": "    ",
                "date": "2023-01-01",
            },
            # Unparseable date
            {
                "review_id": "r6",
                "business_id": "b1",
                "stars": 3,
                "text": "Valid text",
                "date": "not-a-real-date",
            },
            # Multiple simultaneous failures
            {
                "review_id": "nan",
                "business_id": "  ",
                "stars": "not_numeric",
                "text": None,
                "date": None,
            },
        ]
    )
    validator = ReviewValidator()
    result = validator.validate_batch(df)

    assert result.total_records == 7
    assert result.valid_count == 0
    assert result.invalid_count == 7

    # Check multiple failure reasons on the last record
    last_record_errors = result.invalid_df.iloc[6]["validation_errors"]
    assert len(last_record_errors) >= 4

    # Verify failure breakdown metrics
    assert result.failure_breakdown.get("invalid_stars", 0) >= 3
    assert result.failure_breakdown.get("empty_text", 0) >= 2


def test_validator_empty_dataframe():
    validator = ReviewValidator()
    result = validator.validate_batch(pd.DataFrame())
    assert result.total_records == 0
    assert result.valid_count == 0
    assert result.invalid_count == 0


# ============================================================================
# TextCleaner Tests
# ============================================================================


def test_clean_text_normalizations():
    cleaner = TextCleaner()

    # 1. HTML entities
    assert cleaner.clean_text("Fish &amp; chips &quot;delight&quot;!") == 'Fish & chips "delight"!'

    # 2. Whitespace and newline collapsing
    assert cleaner.clean_text("   Great    food \n\n  and \t service!   ") == "Great food and service!"

    # 3. Control characters
    assert cleaner.clean_text("Clean\x00\x08text\x1fhere") == "Clean text here"

    # 4. Null and NaN handling
    assert cleaner.clean_text(None) == ""
    assert cleaner.clean_text(np.nan) == ""
    assert cleaner.clean_text(12345) == "12345"


def test_clean_dataframe_preserves_original_text():
    df = pd.DataFrame(
        [
            {"review_id": "r1", "text": "   Raw &amp; Unclean   "},
            {"review_id": "r2", "text": "Normal text"},
            {"review_id": "r3", "text": None},
        ]
    )
    cleaner = TextCleaner()
    cleaned_df = cleaner.clean_dataframe(df)

    assert "original_text" in cleaned_df.columns
    assert "cleaned_text" in cleaned_df.columns

    # Verify original text preserved exactly
    assert cleaned_df.iloc[0]["original_text"] == "   Raw &amp; Unclean   "
    # Verify cleaned column normalized
    assert cleaned_df.iloc[0]["cleaned_text"] == "Raw & Unclean"
    # Verify primary 'text' updated with cleaned text
    assert cleaned_df.iloc[0]["text"] == "Raw & Unclean"
    # Verify None safely converted to empty string
    assert cleaned_df.iloc[2]["cleaned_text"] == ""


def test_clean_dataframe_empty():
    cleaner = TextCleaner()
    res = cleaner.clean_dataframe(pd.DataFrame())
    assert res.empty


# ============================================================================
# ReviewDeduplicator Tests
# ============================================================================


def test_deduplicator_in_batch():
    df = pd.DataFrame(
        [
            {"review_id": "r1", "text": "First instance", "stars": 5},
            {"review_id": "r2", "text": "Unique record", "stars": 3},
            {"review_id": "r1", "text": "Second duplicate instance", "stars": 1},
        ]
    )
    deduplicator = ReviewDeduplicator(stateful=False)
    result = deduplicator.deduplicate_batch(df, keep="first")

    assert result.total_records == 3
    assert result.unique_count == 2
    assert result.duplicate_count == 1

    # Deterministically preserved the first occurrence
    assert result.deduplicated_df.iloc[0]["text"] == "First instance"
    assert result.deduplicated_df.iloc[1]["review_id"] == "r2"

    # Isolated duplicate in duplicates_df
    assert len(result.duplicates_df) == 1
    assert result.duplicates_df.iloc[0]["text"] == "Second duplicate instance"


def test_deduplicator_cross_batch_stateful():
    deduplicator = ReviewDeduplicator(stateful=True)

    batch_1 = pd.DataFrame(
        [
            {"review_id": "r100", "text": "B1 text"},
            {"review_id": "r101", "text": "B1 text"},
        ]
    )
    res_1 = deduplicator.deduplicate_batch(batch_1)
    assert res_1.unique_count == 2
    assert res_1.duplicate_count == 0
    assert deduplicator.seen_count == 2

    # Batch 2 contains a repeat of r100 from batch 1 and a new r102
    batch_2 = pd.DataFrame(
        [
            {"review_id": "r100", "text": "Repeat of B1"},
            {"review_id": "r102", "text": "B2 text"},
        ]
    )
    res_2 = deduplicator.deduplicate_batch(batch_2)
    assert res_2.unique_count == 1
    assert res_2.duplicate_count == 1
    assert res_2.deduplicated_df.iloc[0]["review_id"] == "r102"
    assert res_2.duplicates_df.iloc[0]["review_id"] == "r100"
    assert deduplicator.total_duplicates_detected == 1


def test_deduplicator_reset_state():
    deduplicator = ReviewDeduplicator(stateful=True)
    df = pd.DataFrame([{"review_id": "r1"}, {"review_id": "r2"}])
    deduplicator.deduplicate_batch(df)
    assert deduplicator.seen_count == 2

    deduplicator.reset_state()
    assert deduplicator.seen_count == 0
    assert deduplicator.total_duplicates_detected == 0


def test_deduplicator_missing_id_column():
    deduplicator = ReviewDeduplicator(id_column="review_id")
    df = pd.DataFrame([{"other_col": 1}, {"other_col": 2}])
    result = deduplicator.deduplicate_batch(df)

    assert result.unique_count == 2
    assert result.duplicate_count == 0

