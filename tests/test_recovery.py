"""Unit tests for recovery layer: CheckpointManager and RetryHandler."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx
import pytest

from src.inference.ollama_client import OllamaConnectionError, OllamaTimeoutError
from src.inference.sentiment_analyzer import MalformedModelResponseError
from src.recovery.checkpoint import BatchCheckpoint, CheckpointManager
from src.recovery.retry_handler import (
    RetryExhaustedError,
    RetryHandler,
    retry_with_backoff,
)


# ============================================================================
# CheckpointManager Tests
# ============================================================================


def test_checkpoint_record_and_atomic_write(tmp_path: Path):
    manager = CheckpointManager(checkpoint_dir=tmp_path)
    assert manager.completed_count == 0
    assert manager.last_completed_batch is None

    cp = manager.record_batch_completion(
        batch_id=1,
        start_offset=0,
        end_offset=32,
        records_count=32,
        source_file="yelp_batch_1.csv",
        metadata={"processing_time_s": 1.25},
    )

    assert isinstance(cp, BatchCheckpoint)
    assert cp.batch_id == 1
    assert cp.processing_status == "COMPLETED"
    assert manager.is_batch_completed(1, source_file="yelp_batch_1.csv") is True
    assert manager.is_batch_completed(2, source_file="yelp_batch_1.csv") is False
    assert manager.get_next_batch_id(source_file="yelp_batch_1.csv") == 2

    # Verify physical file written atomically
    assert manager.checkpoint_file.exists()
    content = json.loads(manager.checkpoint_file.read_text(encoding="utf-8"))
    assert "completed_batches" in content
    assert "yelp_batch_1.csv:1" in content["completed_batches"]


def test_checkpoint_restart_recovery(tmp_path: Path):
    # Phase 1: Simulate initial pipeline execution running batches 1 and 2
    mgr1 = CheckpointManager(checkpoint_dir=tmp_path)
    mgr1.record_batch_completion(1, 0, 10, 10, source_file="data.json")
    mgr1.record_batch_completion(2, 10, 20, 10, source_file="data.json")
    assert mgr1.completed_count == 2
    assert mgr1.last_completed_batch.batch_id == 2

    # Phase 2: Simulate complete application restart by instantiating a fresh CheckpointManager
    mgr2 = CheckpointManager(checkpoint_dir=tmp_path)
    assert mgr2.completed_count == 2
    assert mgr2.is_batch_completed(1, source_file="data.json") is True
    assert mgr2.is_batch_completed(2, source_file="data.json") is True
    assert mgr2.is_batch_completed(3, source_file="data.json") is False
    assert mgr2.get_next_batch_id(source_file="data.json") == 3
    assert mgr2.last_completed_batch.batch_id == 2


def test_checkpoint_corrupted_file_recovery(tmp_path: Path):
    checkpoint_file = tmp_path / "pipeline_checkpoint.json"
    # Write truncated/malformed JSON
    checkpoint_file.write_text('{"version": "1.0", "completed_batches": { "bad_json": ', encoding="utf-8")

    manager = CheckpointManager(checkpoint_dir=tmp_path)
    # Must not crash; corrupt file quarantined and empty state initialized
    assert manager.completed_count == 0
    assert manager.last_completed_batch is None

    # Check that the corrupted file was safely backed up with timestamp
    backups = list(tmp_path.glob("corrupted_*pipeline_checkpoint.json"))
    assert len(backups) == 1


def test_checkpoint_empty_file_handling(tmp_path: Path):
    checkpoint_file = tmp_path / "pipeline_checkpoint.json"
    checkpoint_file.write_text("", encoding="utf-8")

    manager = CheckpointManager(checkpoint_dir=tmp_path)
    assert manager.completed_count == 0


def test_checkpoint_reset(tmp_path: Path):
    manager = CheckpointManager(checkpoint_dir=tmp_path)
    manager.record_batch_completion(1, 0, 10, 10)
    assert manager.checkpoint_file.exists()

    manager.reset_checkpoint()
    assert manager.completed_count == 0
    assert manager.last_completed_batch is None
    assert not manager.checkpoint_file.exists()


# ============================================================================
# RetryHandler Tests
# ============================================================================


def test_retry_successful_first_attempt():
    handler = RetryHandler(max_retries=3, initial_delay=0.01)
    mock_func = MagicMock(return_value="success")

    result = handler.execute(mock_func, "arg1", key="val")
    assert result == "success"
    assert mock_func.call_count == 1


def test_retry_successful_after_transient_failures():
    handler = RetryHandler(max_retries=3, initial_delay=0.01, backoff_factor=1.0)
    mock_func = MagicMock(
        side_effect=[
            OllamaConnectionError("Network glitch"),
            OllamaTimeoutError("Read timed out"),
            "recovered_value",
        ]
    )

    with patch("time.sleep") as mock_sleep:
        result = handler.execute(mock_func)

    assert result == "recovered_value"
    assert mock_func.call_count == 3
    assert mock_sleep.call_count == 2


def test_retry_exhaustion_raises_retry_exhausted_error():
    handler = RetryHandler(max_retries=2, initial_delay=0.01)
    mock_func = MagicMock(side_effect=httpx.ConnectError("Ollama down"))

    with patch("time.sleep"):
        with pytest.raises(RetryExhaustedError) as exc_info:
            handler.execute(mock_func)

    assert exc_info.value.attempts == 2
    assert isinstance(exc_info.value.last_exception, httpx.ConnectError)
    assert mock_func.call_count == 3  # Initial try + 2 retries


def test_non_retryable_error_fails_immediately():
    handler = RetryHandler(max_retries=3, initial_delay=0.01)
    mock_func = MagicMock(side_effect=MalformedModelResponseError("Bad JSON returned"))

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(MalformedModelResponseError):
            handler.execute(mock_func)

    # Must fail on attempt 1 without sleeping or retrying
    assert mock_func.call_count == 1
    assert mock_sleep.call_count == 0


def test_retry_decorator_usage():
    mock_target = MagicMock(
        side_effect=[ConnectionError("Temporary drop"), "decorator_success"]
    )

    @retry_with_backoff(max_retries=2, initial_delay=0.01)
    def decorated_call():
        return mock_target()

    with patch("time.sleep"):
        res = decorated_call()

    assert res == "decorator_success"
    assert mock_target.call_count == 2

