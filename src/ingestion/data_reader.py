"""Multi-format raw data ingestion with encoding fallback, size guards, and quarantine."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union
import pandas as pd

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger("ingestion.data_reader")


class DataReaderError(Exception):
    """Base exception for ingestion read failures."""


class UnsupportedFileFormatError(DataReaderError):
    """File extension is not in the allowed ingestion formats."""


class EmptyFileError(DataReaderError):
    """File contains zero bytes or zero valid data records."""


class FileSizeLimitExceededError(DataReaderError):
    """File exceeds the configured in-memory safety threshold."""


class DataReader:
    """Ingests tabular and semi-structured files (CSV, JSON, JSONL, Parquet) into DataFrames."""

    def __init__(
        self,
        supported_extensions: Optional[Tuple[str, ...]] = None,
        default_encoding: Optional[str] = None,
        max_file_size_mb: Optional[int] = None,
    ):
        self.supported_extensions = tuple(
            ext.lower() for ext in (supported_extensions or settings.ingestion.supported_extensions)
        )
        self.default_encoding = default_encoding or settings.ingestion.default_encoding
        self.fallback_encodings = settings.ingestion.fallback_encodings
        self.max_file_size_bytes = (
            max_file_size_mb or settings.ingestion.max_file_size_mb
        ) * 1024 * 1024

    def list_files(
        self,
        directory: Optional[Union[str, Path]] = None,
        pattern: str = "*",
    ) -> List[Path]:
        target_dir = Path(directory) if directory else settings.paths.raw_data_dir
        if not target_dir.exists():
            logger.warning("Ingestion directory does not exist: %s", target_dir)
            return []

        matched = [
            f for f in target_dir.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.supported_extensions
        ]
        matched.sort()
        return matched

    def read_file(
        self,
        file_path: Union[str, Path],
        add_source_column: bool = False,
    ) -> pd.DataFrame:
        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        if not path.is_file():
            raise DataReaderError(f"Path is not a regular file: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.supported_extensions:
            raise UnsupportedFileFormatError(
                f"Unsupported format '{suffix}' for {path.name}. Allowed: {self.supported_extensions}"
            )

        file_size = path.stat().st_size
        if file_size == 0:
            raise EmptyFileError(f"File is 0 bytes: {path}")

        if file_size > self.max_file_size_bytes:
            raise FileSizeLimitExceededError(
                f"File {path.name} ({file_size / (1024 * 1024):.1f} MB) exceeds "
                f"limit of {self.max_file_size_bytes / (1024 * 1024):.1f} MB"
            )

        try:
            df = self._parse_by_format(path, suffix)
        except EmptyFileError:
            raise
        except pd.errors.EmptyDataError as exc:
            raise EmptyFileError(f"No parseable data in {path.name}") from exc
        except Exception as exc:
            logger.error("Failed to parse %s: %s", path.name, exc)
            raise DataReaderError(f"Corrupted or malformed file {path.name}: {exc}") from exc

        # Drop completely blank rows (e.g., trailing commas in CSV exports)
        df = df.dropna(how="all").reset_index(drop=True)
        if df.empty:
            raise EmptyFileError(f"File {path.name} contains no valid data rows")

        df.columns = [str(col).strip() for col in df.columns]

        if add_source_column:
            df["_source_file"] = path.name

        logger.info("Ingested %d rows from %s", len(df), path.name)
        return df

    def read_directory(
        self,
        directory: Optional[Union[str, Path]] = None,
        pattern: str = "*",
        add_source_column: bool = True,
        quarantine_invalid: bool = False,
        quarantine_dir: Optional[Union[str, Path]] = None,
    ) -> List[Tuple[Path, pd.DataFrame]]:
        """Reads all supported files in a directory; optionally moves corrupt files to failed_data_dir."""
        files = self.list_files(directory=directory, pattern=pattern)
        results: List[Tuple[Path, pd.DataFrame]] = []

        for file_path in files:
            try:
                df = self.read_file(file_path, add_source_column=add_source_column)
                results.append((file_path, df))
            except DataReaderError as err:
                logger.warning("Skipping invalid file %s: %s", file_path.name, err)
                if quarantine_invalid:
                    self._quarantine_file(file_path, quarantine_dir)

        return results

    def _quarantine_file(
        self,
        file_path: Path,
        quarantine_dir: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        target_dir = Path(quarantine_dir) if quarantine_dir else settings.paths.failed_data_dir
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            dest = target_dir / f"{file_path.stem}_{ts}{file_path.suffix}"
            shutil.move(str(file_path), str(dest))
            logger.info("Quarantined corrupt file %s -> %s", file_path.name, dest)
            return dest
        except OSError as exc:
            logger.error("Failed to quarantine %s: %s", file_path.name, exc)
            return None

    def _parse_by_format(self, path: Path, suffix: str) -> pd.DataFrame:
        if suffix == ".csv":
            return self._read_with_encoding_fallback(path, lambda p, enc: pd.read_csv(p, encoding=enc))
        if suffix == ".jsonl":
            return self._read_with_encoding_fallback(
                path, lambda p, enc: pd.read_json(p, lines=True, encoding=enc)
            )
        if suffix == ".json":
            return self._read_with_encoding_fallback(path, self._parse_json_file)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        raise UnsupportedFileFormatError(f"Unhandled extension: {suffix}")

    def _read_with_encoding_fallback(self, path: Path, reader_fn) -> pd.DataFrame:
        # Deduplicate encoding sequence while preserving order
        encodings = list(dict.fromkeys([self.default_encoding, *self.fallback_encodings]))
        last_error: Optional[Exception] = None

        for enc in encodings:
            try:
                return reader_fn(path, enc)
            except UnicodeDecodeError as err:
                last_error = err
                logger.debug("Encoding '%s' failed for %s; trying next fallback", enc, path.name)

        raise DataReaderError(f"Exhausted encodings {encodings} for {path.name}: {last_error}")

    @staticmethod
    def _parse_json_file(path: Path, encoding: str) -> pd.DataFrame:
        with open(path, "r", encoding=encoding) as fh:
            content = fh.read().strip()

        if not content:
            raise EmptyFileError(f"JSON file contains only whitespace: {path.name}")

        data = json.loads(content)

        if isinstance(data, list):
            return pd.DataFrame(data)

        if isinstance(data, dict):
            if not data:
                raise EmptyFileError(f"JSON file contains empty object: {path.name}")

            # Check common envelope keys used in REST payloads
            for envelope_key in ("reviews", "data", "records", "items", "results"):
                if envelope_key in data and isinstance(data[envelope_key], list):
                    return pd.DataFrame(data[envelope_key])

            # Distinguish columnar dict {"col": [v1, v2]} from single-record dict {"col": v1}
            if all(isinstance(v, list) for v in data.values()):
                return pd.DataFrame(data)

            return pd.DataFrame([data])

        raise DataReaderError(f"Unsupported JSON root type in {path.name}: {type(data).__name__}")
