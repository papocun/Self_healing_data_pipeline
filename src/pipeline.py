"""End-to-end self-healing pipeline orchestrator integrating ingestion, processing, inference, and recovery."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.inference.ollama_client import OllamaClient
from src.inference.sentiment_analyzer import (
    MalformedModelResponseError,
    SentimentAnalyzer,
    SentimentResult,
)
from src.ingestion.batch_loader import BatchItem, BatchLoader
from src.ingestion.data_reader import DataReader
from src.processing.cleaner import TextCleaner
from src.processing.deduplicator import ReviewDeduplicator
from src.processing.validator import ReviewValidator
from src.recovery.checkpoint import CheckpointManager
from src.recovery.retry_handler import RetryExhaustedError, RetryHandler
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger("pipeline.orchestrator")


@dataclass
class PipelineRunSummary:
    """Consolidated metrics from a pipeline execution."""

    total_batches: int = 0
    batches_processed: int = 0
    batches_skipped: int = 0
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    duplicate_records: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    prediction_files: List[Path] = field(default_factory=list)
    quarantine_files: List[Path] = field(default_factory=list)


class PipelineOrchestrator:
    """Orchestrates end-to-end sentiment batch processing with self-healing failure recovery."""

    def __init__(
        self,
        batch_size: Optional[int] = None,
        data_reader: Optional[DataReader] = None,
        validator: Optional[ReviewValidator] = None,
        cleaner: Optional[TextCleaner] = None,
        deduplicator: Optional[ReviewDeduplicator] = None,
        ollama_client: Optional[OllamaClient] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        retry_handler: Optional[RetryHandler] = None,
    ):
        settings.ensure_directories()
        self.batch_size = batch_size or settings.ingestion.batch_size

        # Ingestion & Processing components
        self.data_reader = data_reader or DataReader()
        self.batch_loader = BatchLoader(batch_size=self.batch_size, data_reader=self.data_reader)
        self.validator = validator or ReviewValidator()
        self.cleaner = cleaner or TextCleaner()
        self.deduplicator = deduplicator or ReviewDeduplicator(stateful=True)

        # Inference components
        self.ollama_client = ollama_client or OllamaClient()
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer(client=self.ollama_client)

        # Recovery components
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.retry_handler = retry_handler or RetryHandler()

    def run_all(self, pattern: Optional[str] = None) -> PipelineRunSummary:
        """Processes all raw data files discovered in settings.paths.raw_data_dir matching pattern."""
        search_pattern = pattern or settings.ingestion.raw_file_pattern
        resolved_raw_dir = Path(settings.paths.raw_data_dir).resolve()

        if not resolved_raw_dir.exists():
            raise FileNotFoundError(f"Resolved raw data directory does not exist: {resolved_raw_dir}")

        files = self.data_reader.list_files(directory=resolved_raw_dir, pattern=search_pattern)
        if not files:
            raise FileNotFoundError(
                f"No input files matched pattern '{search_pattern}' in resolved raw data directory '{resolved_raw_dir}'"
            )

        logger.info(
            "Starting pipeline execution | resolved_raw_data_dir=%s | pattern=%s | selected_files=%s",
            resolved_raw_dir,
            search_pattern,
            [f.name for f in files],
        )

        overall_summary = PipelineRunSummary()
        for file_path in files:
            file_summary = self.run_file(file_path)
            self._merge_summary(overall_summary, file_summary)

        logger.info(
            "Pipeline execution summary | resolved_raw_data_dir=%s | files_selected=%s | "
            "batches_processed=%d | batches_skipped=%d | total_records=%d | prediction_output_paths=%s",
            resolved_raw_dir,
            [f.name for f in files],
            overall_summary.batches_processed,
            overall_summary.batches_skipped,
            overall_summary.total_records,
            overall_summary.prediction_files,
        )

        if overall_summary.total_records == 0 or overall_summary.batches_processed == 0:
            raise RuntimeError(
                f"Pipeline completed with 0 records or 0 batches processed "
                f"(total_records={overall_summary.total_records}, "
                f"batches_processed={overall_summary.batches_processed}) "
                f"for pattern '{search_pattern}' in '{resolved_raw_dir}'"
            )

        return overall_summary

    def run_file(self, file_path: Union[str, Path]) -> PipelineRunSummary:
        """Executes the pipeline for a single raw data file with restart recovery support."""
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {path}")

        source_name = path.name
        start_batch = self.checkpoint_manager.get_next_batch_id(source_file=source_name)

        logger.info(
            "Starting execution for %s (full_path=%s, resuming from batch_id=%d)",
            source_name,
            path,
            start_batch,
        )

        batch_generator = self.batch_loader.get_batches_from_file(path, start_batch=start_batch)
        first_batch = next(batch_generator, None)

        if first_batch is None and start_batch > 1:
            logger.info(
                "Source '%s' already completed prior run up to batch_id=%d (0 remaining batches at current "
                "batch_size=%d); resetting checkpoint for '%s' to start a fresh processing cycle from batch_id=1",
                source_name,
                start_batch - 1,
                self.batch_size,
                source_name,
            )
            self.checkpoint_manager.clear_source_checkpoint(source_file=source_name)
            self.deduplicator.reset_state()
            start_batch = 1
            batch_generator = self.batch_loader.get_batches_from_file(path, start_batch=1)
            first_batch = next(batch_generator, None)

        if first_batch is None:
            raise RuntimeError(
                f"Input file '{path}' yielded 0 batches and 0 records. Empty runs are not permitted."
            )

        chained_batches = itertools.chain([first_batch], batch_generator)
        summary = self._process_batch_stream(chained_batches, source_name=source_name)

        if start_batch > 1:
            skipped_count = start_batch - 1
            summary.batches_skipped += skipped_count
            summary.total_batches += skipped_count

        logger.info(
            "Completed file %s | batches_processed=%d | batches_skipped=%d | "
            "total_records=%d | prediction_output_paths=%s",
            source_name,
            summary.batches_processed,
            summary.batches_skipped,
            summary.total_records,
            summary.prediction_files,
        )

        if summary.total_records == 0 or summary.batches_processed == 0:
            raise RuntimeError(
                f"Execution for file '{path}' processed 0 records or 0 batches "
                f"(total_records={summary.total_records}, batches_processed={summary.batches_processed})"
            )

        return summary

    def run_dataframe(self, df: pd.DataFrame, source_name: str = "in_memory") -> PipelineRunSummary:
        """Executes the pipeline on an in-memory DataFrame chunk."""
        start_batch = self.checkpoint_manager.get_next_batch_id(source_file=source_name)
        batch_generator = self.batch_loader.get_batches_from_dataframe(
            df, source_name=source_name, start_batch=start_batch
        )
        return self._process_batch_stream(batch_generator, source_name=source_name)

    def _process_batch_stream(
        self,
        batch_generator,
        source_name: str,
    ) -> PipelineRunSummary:
        """Core batch processing loop with validation, deduplication, inference, and recovery."""
        summary = PipelineRunSummary()

        for batch in batch_generator:
            summary.total_batches += 1
            summary.total_records += batch.record_count

            # 1. Skip previously completed batches (idempotency guard)
            if self.checkpoint_manager.is_batch_completed(batch.batch_id, source_file=source_name):
                logger.info(
                    "Skipping already completed batch %d for source %s",
                    batch.batch_id,
                    source_name,
                )
                summary.batches_skipped += 1
                continue

            if batch.total_batches > 0:
                logger.info(
                    "Processing batch %d/%d (%d records) for %s",
                    batch.batch_id,
                    batch.total_batches,
                    batch.record_count,
                    source_name,
                )
            else:
                logger.info(
                    "Processing batch %d (%d records) for %s",
                    batch.batch_id,
                    batch.record_count,
                    source_name,
                )

            # 2. Validation
            val_result = self.validator.validate_batch(batch.data)
            summary.valid_records += val_result.valid_count
            summary.invalid_records += val_result.invalid_count

            # Quarantine invalid records separately if any exist
            if val_result.invalid_count > 0:
                q_file = self._persist_invalid_records(val_result.invalid_df, batch.batch_id, source_name)
                summary.quarantine_files.append(q_file)

            if val_result.valid_count == 0:
                logger.warning("Batch %d has 0 valid records; committing checkpoint", batch.batch_id)
                self._commit_batch_checkpoint(batch, records_processed=0, source_name=source_name)
                summary.batches_processed += 1
                continue

            # 3. Cleaning
            cleaned_df = self.cleaner.clean_dataframe(val_result.valid_df)

            # 4. Deduplication
            dedup_result = self.deduplicator.deduplicate_batch(cleaned_df)
            summary.duplicate_records += dedup_result.duplicate_count

            if dedup_result.duplicate_count > 0:
                dup_file = self._persist_duplicate_records(dedup_result.duplicates_df, batch.batch_id, source_name)
                summary.quarantine_files.append(dup_file)

            if dedup_result.unique_count == 0:
                logger.warning("Batch %d has 0 unique records after deduplication", batch.batch_id)
                self._commit_batch_checkpoint(batch, records_processed=0, source_name=source_name)
                summary.batches_processed += 1
                continue

            # 5. Sentiment Inference with Retry Handling
            inference_records, failed_records = self._run_inference_on_batch(
                dedup_result.deduplicated_df, batch.batch_id, source_name
            )
            summary.successful_predictions += len(inference_records)
            summary.failed_predictions += len(failed_records)

            # Quarantine unrecoverable/malformed inference records
            if failed_records:
                fail_file = self._persist_failed_inference_records(failed_records, batch.batch_id, source_name)
                summary.quarantine_files.append(fail_file)

            # 6. Persist Predictions Deterministically
            if inference_records:
                pred_file = self._persist_predictions(inference_records, batch.batch_id, source_name)
                summary.prediction_files.append(pred_file)

            # 7. Commit Checkpoint ONLY after successful output persistence
            self._commit_batch_checkpoint(
                batch,
                records_processed=len(inference_records),
                source_name=source_name,
                metadata={
                    "successful_predictions": len(inference_records),
                    "failed_predictions": len(failed_records),
                    "duplicates": dedup_result.duplicate_count,
                    "invalids": val_result.invalid_count,
                },
            )
            summary.batches_processed += 1

        return summary

    def _run_inference_on_batch(
        self,
        df: pd.DataFrame,
        batch_id: int,
        source_name: str,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Runs sentiment analysis row-by-row protected by exponential retry handler."""
        successful_records: List[Dict[str, Any]] = []
        failed_records: List[Dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for _, row in df.iterrows():
            rev_id = str(row["review_id"])
            text = str(row["text"])

            try:
                # Wrap inference call in exponential backoff retry handler
                sentiment_res: SentimentResult = self.retry_handler.execute(
                    self.sentiment_analyzer.analyze_review,
                    review_id=rev_id,
                    text=text,
                )

                # Merge original row data with structured sentiment prediction and metadata
                record = row.to_dict()
                record.update(
                    {
                        "sentiment": sentiment_res.sentiment,
                        "confidence": sentiment_res.confidence,
                        "positive_aspects": sentiment_res.positive_aspects,
                        "negative_aspects": sentiment_res.negative_aspects,
                        "reasoning": sentiment_res.reasoning,
                        "model_name": settings.ollama.model,
                        "inference_timestamp": now_iso,
                        "batch_id": batch_id,
                        "source_file": source_name,
                    }
                )
                successful_records.append(record)

            except MalformedModelResponseError as malformed_err:
                logger.error("Non-retryable malformed response on review %s: %s", rev_id, malformed_err)
                failed_records.append(
                    {
                        "review_id": rev_id,
                        "text": text,
                        "batch_id": batch_id,
                        "source_file": source_name,
                        "error_type": "MalformedModelResponseError",
                        "error_message": str(malformed_err),
                        "timestamp": now_iso,
                    }
                )
            except RetryExhaustedError as retry_err:
                logger.error("Retry exhausted on review %s; aborting batch: %s", rev_id, retry_err)
                # Re-raise so batch remains uncommitted in checkpoint for future retry
                raise
            except Exception as unhandled_err:
                logger.error("Unhandled inference error on review %s: %s", rev_id, unhandled_err)
                failed_records.append(
                    {
                        "review_id": rev_id,
                        "text": text,
                        "batch_id": batch_id,
                        "source_file": source_name,
                        "error_type": type(unhandled_err).__name__,
                        "error_message": str(unhandled_err),
                        "timestamp": now_iso,
                    }
                )

        return successful_records, failed_records

    def _persist_predictions(
        self, records: List[Dict[str, Any]], batch_id: int, source_name: str
    ) -> Path:
        """Deterministically persists prediction batch to output/predictions directory."""
        stem = Path(source_name).stem
        target_path = settings.paths.predictions_dir / f"predictions_{stem}_batch_{batch_id:05d}.parquet"

        df_out = pd.DataFrame(records)
        try:
            df_out.to_parquet(target_path, index=False)
        except Exception:
            # Fallback to json if parquet engine is unavailable
            json_target = target_path.with_suffix(".json")
            df_out.to_json(json_target, orient="records", indent=2)
            target_path = json_target

        logger.info("Persisted %d predictions -> %s", len(records), target_path.name)
        return target_path

    def _persist_invalid_records(
        self, df_invalid: pd.DataFrame, batch_id: int, source_name: str
    ) -> Path:
        stem = Path(source_name).stem
        target_path = settings.paths.failed_data_dir / f"validation_failed_{stem}_batch_{batch_id:05d}.json"
        df_invalid.to_json(target_path, orient="records", indent=2)
        logger.warning("Quarantined %d invalid records -> %s", len(df_invalid), target_path.name)
        return target_path

    def _persist_duplicate_records(
        self, df_duplicates: pd.DataFrame, batch_id: int, source_name: str
    ) -> Path:
        stem = Path(source_name).stem
        target_path = settings.paths.failed_data_dir / f"duplicates_{stem}_batch_{batch_id:05d}.json"
        df_duplicates.to_json(target_path, orient="records", indent=2)
        logger.info("Quarantined %d duplicate records -> %s", len(df_duplicates), target_path.name)
        return target_path

    def _persist_failed_inference_records(
        self, failed_records: List[Dict[str, Any]], batch_id: int, source_name: str
    ) -> Path:
        stem = Path(source_name).stem
        target_path = settings.paths.failed_data_dir / f"inference_failed_{stem}_batch_{batch_id:05d}.json"
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(failed_records, f, indent=2)
        logger.warning("Quarantined %d failed inference records -> %s", len(failed_records), target_path.name)
        return target_path

    def _commit_batch_checkpoint(
        self,
        batch: BatchItem,
        records_processed: int,
        source_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.checkpoint_manager.record_batch_completion(
            batch_id=batch.batch_id,
            start_offset=batch.start_row,
            end_offset=batch.end_row,
            records_count=records_processed,
            source_file=source_name,
            metadata=metadata or {},
        )

    @staticmethod
    def _merge_summary(target: PipelineRunSummary, source: PipelineRunSummary) -> None:
        target.total_batches += source.total_batches
        target.batches_processed += source.batches_processed
        target.batches_skipped += source.batches_skipped
        target.total_records += source.total_records
        target.valid_records += source.valid_records
        target.invalid_records += source.invalid_records
        target.duplicate_records += source.duplicate_records
        target.successful_predictions += source.successful_predictions
        target.failed_predictions += source.failed_predictions
        target.prediction_files.extend(source.prediction_files)
        target.quarantine_files.extend(source.quarantine_files)

