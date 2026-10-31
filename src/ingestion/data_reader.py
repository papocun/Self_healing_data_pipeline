"""Multi-format raw data ingestion with format detection, streaming JSONL, and quarantine."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Generator, List, Optional, Tuple, Union
import pandas as pd

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger("ingestion.data_reader")


class DataReaderError(Exception):
    """Base exception for ingestion read failures."""


class UnsupportedFileFormatError(DataReaderError):
    """File extension or format is not in the allowed ingestion formats."""


class EmptyFileError(DataReaderError):
    """File contains zero bytes or zero valid data records."""


class FileSizeLimitExceededError(DataReaderError):
    """File exceeds the configured in-memory safety threshold."""


class DataReader:
    """Ingests tabular and semi-structured files (CSV, JSON, JSONL, NDJSON, Parquet)."""

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

    def detect_format(self, file_path: Union[str, Path]) -> str:
        """Determines whether a file is 'csv', 'parquet', 'jsonl', or standard 'json'.

        Inspects file extension and uses content sniffing on .json files to detect
        newline-delimited JSON (JSONL/NDJSON) vs standard JSON documents.
        """
        path = Path(file_path).resolve()
        suffix = path.suffix.lower()

        if suffix in (".jsonl", ".ndjson"):
            return "jsonl"
        if suffix == ".csv":
            return "csv"
        if suffix == ".parquet":
            return "parquet"
        if suffix != ".json":
            return suffix.lstrip(".")

        # Content sniffing for .json files (e.g. Yelp datasets)
        encodings = list(dict.fromkeys([self.default_encoding, *self.fallback_encodings]))
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    valid_json_objects = 0
                    lines_inspected = 0
                    for line in f:
                        stripped = line.strip()
                        if not stripped:
                            continue
                        lines_inspected += 1

                        # Multi-line array is standard JSON
                        if stripped.startswith("[") and not stripped.endswith("]"):
                            return "json"

                        try:
                            obj = json.loads(stripped)
                            if isinstance(obj, dict):
                                valid_json_objects += 1
                                if valid_json_objects >= 2:
                                    return "jsonl"
                        except Exception:
                            pass

                        if lines_inspected >= 10:
                            break

                if valid_json_objects >= 2:
                    return "jsonl"
                return "json"
            except (UnicodeDecodeError, OSError):
                continue

        return "json"

    def list_files(
        self,
        directory: Optional[Union[str, Path]] = None,
        pattern: str = "*",
    ) -> List[Path]:
        """Lists supported data files in a directory matching one or more comma-separated patterns."""
        target_dir = (Path(directory) if directory else settings.paths.raw_data_dir).resolve()
        if not target_dir.exists():
            logger.warning("Ingestion directory does not exist: %s", target_dir)
            return []

        discovered_files = sorted(
            [
                f.name
                for f in target_dir.iterdir()
                if f.is_file() and f.suffix.lower() in self.supported_extensions
            ]
        )

        patterns = [p.strip() for p in pattern.split(",") if p.strip()]
        matched_set = set()

        for pat in patterns:
            candidate_patterns = [pat]
            if Path(pat).name != pat:
                candidate_patterns.append(Path(pat).name)
            for candidate_pat in candidate_patterns:
                for f in target_dir.glob(candidate_pat):
                    if f.is_file() and f.suffix.lower() in self.supported_extensions:
                        matched_set.add(f.resolve())

        matched_files = sorted(list(matched_set))
        logger.info(
            "Directory discovery in %s | Discovered files (%d): %s | Selected by pattern '%s' (%d): %s",
            target_dir,
            len(discovered_files),
            discovered_files,
            pattern,
            len(matched_files),
            [f.name for f in matched_files],
        )
        return matched_files

    def read_file(
        self,
        file_path: Union[str, Path],
        add_source_column: bool = False,
    ) -> pd.DataFrame:
        """Reads a data file into memory as a DataFrame.

        For files exceeding max_file_size_mb, raises FileSizeLimitExceededError.
        Use stream_jsonl or BatchLoader.get_batches_from_file for streaming large datasets.
        """
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
                f"limit of {self.max_file_size_bytes / (1024 * 1024):.1f} MB. "
                "Use streaming ingestion via BatchLoader to avoid OOM."
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

        # Drop completely blank rows
        df = df.dropna(how="all").reset_index(drop=True)
        if df.empty:
            raise EmptyFileError(f"File {path.name} contains no valid data rows")

        df.columns = [str(col).strip() for col in df.columns]

        if add_source_column:
            df["_source_file"] = path.name

        logger.info("Ingested %d rows from %s", len(df), path.name)
        return df

    def stream_jsonl(
        self,
        file_path: Union[str, Path],
        batch_size: int = 1000,
        add_source_column: bool = False,
    ) -> Generator[pd.DataFrame, None, None]:
        """Memory-efficient generator yielding DataFrames of size batch_size from a JSONL file."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        if path.stat().st_size == 0:
            raise EmptyFileError(f"File is 0 bytes: {path}")

        records: List[dict] = []
        encodings = list(dict.fromkeys([self.default_encoding, *self.fallback_encodings]))

        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    for line_num, line in enumerate(f, start=1):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            item = json.loads(stripped)
                            if isinstance(item, dict):
                                records.append(item)
                            elif isinstance(item, list):
                                records.extend(item)
                        except (json.JSONDecodeError, ValueError) as err:
                            logger.warning("Malformed JSON on line %d in %s: %s", line_num, path.name, err)
                            continue

                        if len(records) >= batch_size:
                            chunk_df = pd.DataFrame(records)
                            chunk_df.columns = [str(c).strip() for c in chunk_df.columns]
                            if add_source_column:
                                chunk_df["_source_file"] = path.name
                            yield chunk_df
                            records.clear()

                    if records:
                        chunk_df = pd.DataFrame(records)
                        chunk_df.columns = [str(c).strip() for c in chunk_df.columns]
                        if add_source_column:
                            chunk_df["_source_file"] = path.name
                        yield chunk_df
                        records.clear()

                return
            except UnicodeDecodeError:
                continue

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
        fmt = self.detect_format(path)
        if fmt == "csv":
            return self._read_with_encoding_fallback(path, lambda p, enc: pd.read_csv(p, encoding=enc))
        if fmt == "jsonl":
            return self._read_with_encoding_fallback(path, self._parse_jsonl_file)
        if fmt == "json":
            return self._read_with_encoding_fallback(path, self._parse_json_file)
        if fmt == "parquet":
            return pd.read_parquet(path)
        raise UnsupportedFileFormatError(f"Unhandled format: {fmt} ({suffix})")

    def _parse_jsonl_file(self, path: Path, encoding: str) -> pd.DataFrame:
        """Parses a JSONL / NDJSON file into a DataFrame, gracefully skipping malformed individual lines."""
        records = []
        with open(path, "r", encoding=encoding) as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    if isinstance(data, dict):
                        records.append(data)
                    elif isinstance(data, list):
                        records.extend(data)
                except (json.JSONDecodeError, ValueError) as err:
                    logger.warning("Skipping malformed line %d in %s: %s", line_num, path.name, err)
                    continue

        if not records:
            raise EmptyFileError(f"File {path.name} contains no valid JSONL records")

        return pd.DataFrame(records)

    def _read_with_encoding_fallback(self, path: Path, reader_fn) -> pd.DataFrame:
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
