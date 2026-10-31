"""Deduplication module using review_id with duplicate tracking and deterministic ordering."""

from dataclasses import dataclass
from typing import Dict, Optional, Set
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("processing.deduplicator")


@dataclass(frozen=True)
class DeduplicationResult:
    """Result of deduplication returning unique records and separated duplicate occurrences."""

    deduplicated_df: pd.DataFrame
    duplicates_df: pd.DataFrame
    total_records: int
    unique_count: int
    duplicate_count: int


class ReviewDeduplicator:
    """Memory-efficient deduplicator for review records using unique ID tracking."""

    def __init__(self, id_column: str = "review_id", stateful: bool = True):
        self.id_column = id_column
        self.stateful = stateful
        # In-memory set of unique IDs (stores only string hashes, ~50MB per 1M IDs, never loads full datasets)
        self._seen_ids: Set[str] = set()
        self._total_processed: int = 0
        self._total_duplicates: int = 0
        self._duplicate_counts_by_id: Dict[str, int] = {}

    @property
    def seen_count(self) -> int:
        return len(self._seen_ids)

    @property
    def total_duplicates_detected(self) -> int:
        return self._total_duplicates

    def reset_state(self) -> None:
        """Clears stateful tracking for a fresh pipeline execution."""
        self._seen_ids.clear()
        self._total_processed = 0
        self._total_duplicates = 0
        self._duplicate_counts_by_id.clear()

    def deduplicate_batch(self, df: pd.DataFrame, keep: str = "first") -> DeduplicationResult:
        """Deduplicates a DataFrame batch, preserving deterministic order and isolating duplicates.

        Args:
            df: Input DataFrame batch.
            keep: Which duplicate occurrence to retain ('first' or 'last'). Defaults to 'first'.

        Returns:
            DeduplicationResult: Contains deduplicated DataFrame and quarantined duplicates DataFrame.
        """
        if df.empty:
            return DeduplicationResult(
                deduplicated_df=df.copy(),
                duplicates_df=df.copy(),
                total_records=0,
                unique_count=0,
                duplicate_count=0,
            )

        if self.id_column not in df.columns:
            logger.warning("ID column '%s' not found for deduplication; returning original", self.id_column)
            return DeduplicationResult(
                deduplicated_df=df.copy(),
                duplicates_df=df.iloc[0:0].copy(),
                total_records=len(df),
                unique_count=len(df),
                duplicate_count=0,
            )

        total_batch_records = len(df)
        df_work = df.copy()

        # Step 1: In-batch deduplication (deterministic)
        in_batch_duplicate_mask = df_work.duplicated(subset=[self.id_column], keep=keep)
        in_batch_unique_df = df_work[~in_batch_duplicate_mask].copy()
        in_batch_duplicates_df = df_work[in_batch_duplicate_mask].copy()

        # Step 2: Cross-batch stateful deduplication (if stateful=True)
        if self.stateful:
            # Check against previously processed batches
            ids_str = in_batch_unique_df[self.id_column].astype(str)
            cross_batch_duplicate_mask = ids_str.isin(self._seen_ids)

            final_unique_df = in_batch_unique_df[~cross_batch_duplicate_mask].copy().reset_index(drop=True)
            cross_batch_duplicates_df = in_batch_unique_df[cross_batch_duplicate_mask].copy()

            all_duplicates_df = pd.concat(
                [in_batch_duplicates_df, cross_batch_duplicates_df],
                axis=0,
                ignore_index=True,
            )

            # Update stateful metrics and seen IDs
            new_ids = set(final_unique_df[self.id_column].astype(str))
            self._seen_ids.update(new_ids)

            # Record duplicate frequency counts
            if not all_duplicates_df.empty:
                for dup_id in all_duplicates_df[self.id_column].astype(str):
                    self._duplicate_counts_by_id[dup_id] = (
                        self._duplicate_counts_by_id.get(dup_id, 0) + 1
                    )
        else:
            final_unique_df = in_batch_unique_df.reset_index(drop=True)
            all_duplicates_df = in_batch_duplicates_df.reset_index(drop=True)

        dup_count = len(all_duplicates_df)
        self._total_processed += total_batch_records
        self._total_duplicates += dup_count

        logger.info(
            "Deduplication complete: %d records -> %d unique, %d duplicates quarantined (stateful_seen=%d)",
            total_batch_records,
            len(final_unique_df),
            dup_count,
            len(self._seen_ids),
        )

        return DeduplicationResult(
            deduplicated_df=final_unique_df,
            duplicates_df=all_duplicates_df,
            total_records=total_batch_records,
            unique_count=len(final_unique_df),
            duplicate_count=dup_count,
        )

