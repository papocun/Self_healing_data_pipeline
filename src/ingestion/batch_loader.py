"""Batch partitioning with checkpoint offset support for resumable pipeline execution."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Generator, List, Optional, Union
import pandas as pd

from src.ingestion.data_reader import DataReader, EmptyFileError
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
        """Streams batches directly from file without loading multi-GB datasets into RAM."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        if path.stat().st_size == 0:
            raise EmptyFileError(f"File is 0 bytes: {path}")

        fmt = self.data_reader.detect_format(path)
        if fmt == "jsonl":
            yield from self._stream_jsonl_batches(path, start_batch=start_batch)
        else:
            df = self.data_reader.read_file(path, add_source_column=True)
            yield from self.get_batches_from_dataframe(
                df, source_name=path.name, start_batch=start_batch
            )

    def _stream_jsonl_batches(
        self,
        path: Path,
        start_batch: int = 1,
    ) -> Generator[BatchItem, None, None]:
        """Streams line-delimited JSONL files in batches of batch_size with fast-forward resume."""
        if start_batch < 1:
            raise ValueError(f"start_batch must be >= 1, got {start_batch}")

        logger.info(
            "Streaming JSONL batches from %s (batch_size=%d, start_batch=%d)",
            path.name,
            self.batch_size,
            start_batch,
        )

        records_to_skip = (start_batch - 1) * self.batch_size
        skipped_records = 0
        yielded_batches = 0
        batch_records: List[dict] = []
        total_valid_records = 0
        malformed_lines: List[dict] = []

        encodings = list(dict.fromkeys([self.data_reader.default_encoding, *self.data_reader.fallback_encodings]))

        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    for line_num, line in enumerate(f, start=1):
                        stripped = line.strip()
                        if not stripped:
                            continue

                        try:
                            record = json.loads(stripped)
                            if not isinstance(record, dict):
                                raise ValueError(f"Expected JSON object, got {type(record).__name__}")
                        except (json.JSONDecodeError, ValueError) as err:
                            logger.warning("Malformed JSON on line %d in %s: %s", line_num, path.name, err)
                            malformed_lines.append(
                                {"line_number": line_num, "raw_line": stripped[:200], "error": str(err)}
                            )
                            continue

                        # Fast-forward past already committed batches
                        if skipped_records < records_to_skip:
                            skipped_records += 1
                            total_valid_records += 1
                            continue

                        batch_records.append(record)
                        total_valid_records += 1

                        if len(batch_records) == self.batch_size:
                            batch_idx = start_batch + yielded_batches
                            start_row = (batch_idx - 1) * self.batch_size
                            end_row = start_row + len(batch_records)

                            chunk_df = pd.DataFrame(batch_records)
                            chunk_df.columns = [str(c).strip() for c in chunk_df.columns]
                            chunk_df["_source_file"] = path.name

                            yield BatchItem(
                                batch_id=batch_idx,
                                total_batches=-1,  # Streaming: total is dynamic
                                data=chunk_df,
                                source_file=path.name,
                                start_row=start_row,
                                end_row=end_row,
                            )
                            yielded_batches += 1
                            batch_records.clear()

                    # Yield trailing records in final partial batch
                    if batch_records:
                        batch_idx = start_batch + yielded_batches
                        start_row = (batch_idx - 1) * self.batch_size
                        end_row = start_row + len(batch_records)

                        chunk_df = pd.DataFrame(batch_records)
                        chunk_df.columns = [str(c).strip() for c in chunk_df.columns]
                        chunk_df["_source_file"] = path.name

                        yield BatchItem(
                            batch_id=batch_idx,
                            total_batches=batch_idx,
                            data=chunk_df,
                            source_file=path.name,
                            start_row=start_row,
                            end_row=end_row,
                        )
                        yielded_batches += 1
                        batch_records.clear()

                if total_valid_records == 0:
                    raise EmptyFileError(f"File {path.name} contains no valid JSONL records")

                return

            except UnicodeDecodeError:
                continue
