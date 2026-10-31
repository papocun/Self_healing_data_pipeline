"""Batch partitioning with checkpoint offset support for resumable pipeline execution."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Generator, Optional, Union
import pandas as pd

from src.ingestion.data_reader import DataReader
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger("ingestion.batch_loader")


@dataclass(frozen=True)
class BatchItem:
    batch_id: int
    total_batches: int
    data: pd.DataFrame
    source_file: Optional[str] = None
    start_row: int = 0
    end_row: int = 0

    @property
    def record_count(self) -> int:
        return len(self.data)


class BatchLoader:
    """Slices DataFrames or raw files into deterministic batches for fault-tolerant processing."""

    def __init__(
        self,
        batch_size: Optional[int] = None,
        data_reader: Optional[DataReader] = None,
    ):
        resolved_size = settings.ingestion.batch_size if batch_size is None else batch_size
        if resolved_size <= 0:
            raise ValueError(f"batch_size must be >= 1, got {resolved_size}")

        self.batch_size = resolved_size
        self.data_reader = data_reader or DataReader()

    def get_batches_from_dataframe(
        self,
        df: pd.DataFrame,
        source_name: Optional[str] = None,
        start_batch: int = 1,
    ) -> Generator[BatchItem, None, None]:
        """Yields sequential batches starting from `start_batch` (1-indexed) to support checkpoint resume."""
        if start_batch < 1:
            raise ValueError(f"start_batch must be >= 1, got {start_batch}")

        total_records = len(df)
        if total_records == 0:
            logger.warning("Empty DataFrame passed to BatchLoader; yielding 0 batches")
            return

        total_batches = math.ceil(total_records / self.batch_size)
        if start_batch > total_batches:
            logger.info(
                "start_batch (%d) exceeds total_batches (%d) for %s; nothing to yield",
                start_batch,
                total_batches,
                source_name or "dataframe",
            )
            return

        logger.info(
            "Partitioning %d records into %d batches (batch_size=%d, start_batch=%d)",
            total_records,
            total_batches,
            self.batch_size,
            start_batch,
        )

        for batch_idx in range(start_batch - 1, total_batches):
            start_row = batch_idx * self.batch_size
            end_row = min(start_row + self.batch_size, total_records)
            # Copy slice so downstream in-place mutations do not trigger SettingWithCopyWarning
            chunk = df.iloc[start_row:end_row].copy().reset_index(drop=True)

            yield BatchItem(
                batch_id=batch_idx + 1,
                total_batches=total_batches,
                data=chunk,
                source_file=source_name,
                start_row=start_row,
                end_row=end_row,
            )

    def get_batches_from_file(
        self,
        file_path: Union[str, Path],
        start_batch: int = 1,
    ) -> Generator[BatchItem, None, None]:
        path = Path(file_path)
        df = self.data_reader.read_file(path, add_source_column=True)
        yield from self.get_batches_from_dataframe(
            df, source_name=path.name, start_batch=start_batch
        )
