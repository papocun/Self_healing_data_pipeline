"""Checkpoint management for fault-tolerant batch resumption using atomic persistence."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Set

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger("recovery.checkpoint")


@dataclass
class BatchCheckpoint:
    """Metadata representing a single processed batch checkpoint."""

    batch_id: int
    start_offset: int
    end_offset: int
    timestamp: str
    processing_status: str  # "COMPLETED", "IN_PROGRESS", "FAILED"
    records_count: int = 0
    source_file: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Manages atomic persistence and state restoration for pipeline checkpoints."""

    def __init__(
        self,
        checkpoint_dir: Optional[Path | str] = None,
        filename: str = "pipeline_checkpoint.json",
    ):
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else settings.paths.checkpoint_dir
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / filename

        # In-memory tracking
        self._last_completed_batch: Optional[BatchCheckpoint] = None
        self._completed_batches: Dict[str, BatchCheckpoint] = {}  # key: "source_file:batch_id"

        # Attempt restoration on initialization
        self.load_checkpoint()

    @property
    def last_completed_batch(self) -> Optional[BatchCheckpoint]:
        return self._last_completed_batch

    @property
    def completed_count(self) -> int:
        return len(self._completed_batches)

    def _make_key(self, batch_id: int, source_file: Optional[str] = None) -> str:
        source_key = source_file or "default"
        return f"{source_key}:{batch_id}"

    def is_batch_completed(self, batch_id: int, source_file: Optional[str] = None) -> bool:
        """Checks if a specific batch has already been successfully committed."""
        key = self._make_key(batch_id, source_file)
        return key in self._completed_batches

    def get_next_batch_id(self, source_file: Optional[str] = None) -> int:
        """Determines the next sequential batch ID to process for a given source."""
        source_key = source_file or "default"
        matching_ids = [
            cp.batch_id
            for cp in self._completed_batches.values()
            if (cp.source_file or "default") == source_key
        ]
        if not matching_ids:
            return 1
        return max(matching_ids) + 1

    def clear_source_checkpoint(self, source_file: Optional[str] = None) -> int:
        """Removes all completed batch entries for a specific source file and persists atomically."""
        source_key = source_file or "default"
        keys_to_remove = [
            key
            for key, cp in self._completed_batches.items()
            if (cp.source_file or "default") == source_key
        ]
        for key in keys_to_remove:
            del self._completed_batches[key]

        if self._last_completed_batch and (self._last_completed_batch.source_file or "default") == source_key:
            remaining = list(self._completed_batches.values())
            self._last_completed_batch = remaining[-1] if remaining else None

        if keys_to_remove:
            self._save_checkpoint_atomic()
            logger.info(
                "Cleared %d completed checkpoint entries for source %s",
                len(keys_to_remove),
                source_key,
            )
        return len(keys_to_remove)

    def record_batch_completion(
        self,
        batch_id: int,
        start_offset: int,
        end_offset: int,
        records_count: int,
        source_file: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchCheckpoint:
        """Atomically commits a successful batch completion to disk.

        Args:
            batch_id: Sequential batch identifier.
            start_offset: Starting row offset in dataset.
            end_offset: Ending row offset in dataset.
            records_count: Count of records processed.
            source_file: Name or path of source file.
            metadata: Additional operational metrics.

        Returns:
            BatchCheckpoint: The persisted checkpoint object.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        checkpoint = BatchCheckpoint(
            batch_id=batch_id,
            start_offset=start_offset,
            end_offset=end_offset,
            timestamp=now_utc,
            processing_status="COMPLETED",
            records_count=records_count,
            source_file=source_file,
            metadata=metadata or {},
        )

        key = self._make_key(batch_id, source_file)
        self._completed_batches[key] = checkpoint
        self._last_completed_batch = checkpoint

        self._save_checkpoint_atomic()
        logger.info(
            "Checkpoint committed: batch_id=%d [%d:%d] records=%d for %s",
            batch_id,
            start_offset,
            end_offset,
            records_count,
            source_file or "default",
        )
        return checkpoint

    def load_checkpoint(self) -> bool:
        """Loads and deserializes existing checkpoint file into memory.

        Returns:
            bool: True if a valid checkpoint was loaded, False if new or corrupt.
        """
        if not self.checkpoint_file.exists():
            logger.debug("No existing checkpoint file found at %s", self.checkpoint_file)
            return False

        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                logger.warning("Checkpoint file is empty at %s", self.checkpoint_file)
                return False

            data = json.loads(content)
            completed_dict = {}

            for key, item in data.get("completed_batches", {}).items():
                completed_dict[key] = BatchCheckpoint(
                    batch_id=item["batch_id"],
                    start_offset=item["start_offset"],
                    end_offset=item["end_offset"],
                    timestamp=item["timestamp"],
                    processing_status=item.get("processing_status", "COMPLETED"),
                    records_count=item.get("records_count", 0),
                    source_file=item.get("source_file"),
                    metadata=item.get("metadata", {}),
                )

            self._completed_batches = completed_dict

            last_data = data.get("last_completed_batch")
            if last_data:
                self._last_completed_batch = BatchCheckpoint(
                    batch_id=last_data["batch_id"],
                    start_offset=last_data["start_offset"],
                    end_offset=last_data["end_offset"],
                    timestamp=last_data["timestamp"],
                    processing_status=last_data.get("processing_status", "COMPLETED"),
                    records_count=last_data.get("records_count", 0),
                    source_file=last_data.get("source_file"),
                    metadata=last_data.get("metadata", {}),
                )

            logger.info(
                "Restored %d completed batches from checkpoint (%s)",
                len(self._completed_batches),
                self.checkpoint_file.name,
            )
            return True

        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error("Corrupted checkpoint file detected at %s: %s", self.checkpoint_file, exc)
            self._quarantine_corrupted_checkpoint()
            return False

    def reset_checkpoint(self) -> None:
        """Removes the checkpoint file and clears memory tracking."""
        self._completed_batches.clear()
        self._last_completed_batch = None
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                logger.info("Checkpoint file cleared: %s", self.checkpoint_file)
            except OSError as exc:
                logger.warning("Failed to unlink checkpoint file %s: %s", self.checkpoint_file, exc)

    def _save_checkpoint_atomic(self) -> None:
        """Atomically writes checkpoint data via a sibling temporary file and atomic rename."""
        state = {
            "version": "1.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "last_completed_batch": (
                asdict(self._last_completed_batch) if self._last_completed_batch else None
            ),
            "completed_batches": {
                key: asdict(cp) for key, cp in self._completed_batches.items()
            },
        }

        serialized = json.dumps(state, indent=2)

        # Write to temporary file in the same directory to ensure same filesystem for os.replace
        temp_fd, temp_path = tempfile.mkstemp(
            prefix="checkpoint_tmp_",
            suffix=".json",
            dir=str(self.checkpoint_dir),
        )

        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(serialized)
                temp_file.flush()
                os.fsync(temp_file.fileno())

            # Atomic swap replacing destination file
            os.replace(temp_path, str(self.checkpoint_file))

        except Exception as exc:
            logger.error("Atomic checkpoint write failed: %s", exc)
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            raise

    def _quarantine_corrupted_checkpoint(self) -> None:
        """Preserves a corrupted checkpoint file for offline inspection rather than crashing."""
        if not self.checkpoint_file.exists():
            return
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        corrupt_backup = self.checkpoint_dir / f"corrupted_{ts}_{self.checkpoint_file.name}"
        try:
            os.replace(str(self.checkpoint_file), str(corrupt_backup))
            logger.warning("Corrupted checkpoint moved to %s", corrupt_backup)
        except OSError as exc:
            logger.error("Failed to backup corrupt checkpoint: %s", exc)

