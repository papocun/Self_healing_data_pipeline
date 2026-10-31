"""Validation module for Yelp customer reviews with error reason tracking."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("processing.validator")

# Required fields per review record
REQUIRED_FIELDS = ("review_id", "business_id", "stars", "text", "date")


@dataclass(frozen=True)
class ValidationResult:
    """Result of batch validation separating clean data from quarantined invalid records."""

    valid_df: pd.DataFrame
    invalid_df: pd.DataFrame
    total_records: int
    valid_count: int
    invalid_count: int
    failure_breakdown: Dict[str, int] = field(default_factory=dict)

    @property
    def is_fully_valid(self) -> bool:
        return self.invalid_count == 0


class ReviewValidator:
    """Vectorized validator for Yelp reviews enforcing schema, range, and format constraints."""

    def __init__(
        self,
        min_stars: float = 1.0,
        max_stars: float = 5.0,
        min_text_length: int = 1,
    ):
        self.min_stars = min_stars
        self.max_stars = max_stars
        self.min_text_length = min_text_length

    def validate_batch(self, df: pd.DataFrame) -> ValidationResult:
        """Validates a DataFrame batch, separating valid records from invalid records.

        Args:
            df: Input DataFrame containing review records.

        Returns:
            ValidationResult: Slices for valid_df, invalid_df, and error metrics.
        """
        if df.empty:
            logger.warning("Empty DataFrame passed to ReviewValidator")
            return ValidationResult(
                valid_df=df.copy(),
                invalid_df=pd.DataFrame(columns=list(df.columns) + ["validation_errors"]),
                total_records=0,
                valid_count=0,
                invalid_count=0,
                failure_breakdown={},
            )

        total_records = len(df)
        df_work = df.copy()

        # Handle column alias for stars (e.g. 'rating' -> 'stars')
        if "stars" not in df_work.columns and "rating" in df_work.columns:
            df_work["stars"] = df_work["rating"]

        # Check missing required columns at DataFrame level
        missing_columns = [col for col in REQUIRED_FIELDS if col not in df_work.columns]
        if missing_columns:
            logger.error("Batch missing required columns: %s", missing_columns)
            error_reason = f"Missing required columns: {', '.join(missing_columns)}"
            invalid_df = df_work.copy()
            invalid_df["validation_errors"] = [[error_reason]] * total_records
            return ValidationResult(
                valid_df=df_work.iloc[0:0].copy(),
                invalid_df=invalid_df,
                total_records=total_records,
                valid_count=0,
                invalid_count=total_records,
                failure_breakdown={error_reason: total_records},
            )

        # 1. review_id validation
        review_id_str = df_work["review_id"].astype(str).str.strip()
        review_id_valid = (
            df_work["review_id"].notna()
            & (review_id_str != "")
            & (review_id_str.str.lower() != "nan")
            & (review_id_str.str.lower() != "none")
        )

        # 2. business_id validation
        biz_id_str = df_work["business_id"].astype(str).str.strip()
        biz_id_valid = (
            df_work["business_id"].notna()
            & (biz_id_str != "")
            & (biz_id_str.str.lower() != "nan")
            & (biz_id_str.str.lower() != "none")
        )

        # 3. stars validation (numeric and between min_stars and max_stars)
        numeric_stars = pd.to_numeric(df_work["stars"], errors="coerce")
        stars_valid = (
            numeric_stars.notna()
            & (numeric_stars >= self.min_stars)
            & (numeric_stars <= self.max_stars)
        )

        # 4. text validation (non-null string with min length)
        text_str = df_work["text"].astype(str).str.strip()
        text_valid = (
            df_work["text"].notna()
            & (text_str.str.len() >= self.min_text_length)
            & (text_str.str.lower() != "nan")
            & (text_str.str.lower() != "none")
        )

        # 5. date validation (parseable ISO or standard datetime)
        parsed_dates = pd.to_datetime(df_work["date"], format="mixed", errors="coerce")
        date_valid = df_work["date"].notna() & parsed_dates.notna()

        # Combine all boolean validity checks
        is_record_valid = (
            review_id_valid
            & biz_id_valid
            & stars_valid
            & text_valid
            & date_valid
        )

        valid_df = df_work[is_record_valid].copy().reset_index(drop=True)
        invalid_rows = df_work[~is_record_valid].copy()

        # Detailed failure attribution for quarantined invalid rows
        failure_breakdown: Dict[str, int] = {}
        invalid_errors: List[List[str]] = []

        if not invalid_rows.empty:
            for idx in invalid_rows.index:
                row_errors: List[str] = []

                if not review_id_valid.loc[idx]:
                    val = df_work.loc[idx, "review_id"]
                    msg = f"Invalid review_id: '{val}'"
                    row_errors.append(msg)
                    failure_breakdown["invalid_review_id"] = failure_breakdown.get("invalid_review_id", 0) + 1

                if not biz_id_valid.loc[idx]:
                    val = df_work.loc[idx, "business_id"]
                    msg = f"Invalid business_id: '{val}'"
                    row_errors.append(msg)
                    failure_breakdown["invalid_business_id"] = failure_breakdown.get("invalid_business_id", 0) + 1

                if not stars_valid.loc[idx]:
                    val = df_work.loc[idx, "stars"]
                    msg = f"Invalid stars: '{val}' (expected {self.min_stars} to {self.max_stars})"
                    row_errors.append(msg)
                    failure_breakdown["invalid_stars"] = failure_breakdown.get("invalid_stars", 0) + 1

                if not text_valid.loc[idx]:
                    val = df_work.loc[idx, "text"]
                    msg = "Missing or empty review text"
                    row_errors.append(msg)
                    failure_breakdown["empty_text"] = failure_breakdown.get("empty_text", 0) + 1

                if not date_valid.loc[idx]:
                    val = df_work.loc[idx, "date"]
                    msg = f"Unparseable date: '{val}'"
                    row_errors.append(msg)
                    failure_breakdown["unparseable_date"] = failure_breakdown.get("unparseable_date", 0) + 1

                invalid_errors.append(row_errors)

            invalid_rows["validation_errors"] = invalid_errors
            invalid_df = invalid_rows.reset_index(drop=True)
        else:
            invalid_df = pd.DataFrame(columns=list(df_work.columns) + ["validation_errors"])

        logger.info(
            "Batch validation completed: %d total, %d valid (%.1f%%), %d invalid",
            total_records,
            len(valid_df),
            (len(valid_df) / total_records * 100.0) if total_records > 0 else 0.0,
            len(invalid_df),
        )

        return ValidationResult(
            valid_df=valid_df,
            invalid_df=invalid_df,
            total_records=total_records,
            valid_count=len(valid_df),
            invalid_count=len(invalid_df),
            failure_breakdown=failure_breakdown,
        )
