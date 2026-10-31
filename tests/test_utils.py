"""Unit tests for configuration validation and logger handler sharing."""

import logging
from pathlib import Path
from pydantic import ValidationError
import pytest

from src.utils.config import IngestionConfig, RecoveryConfig, Settings
from src.utils.logger import get_logger, setup_logger


def test_settings_dynamic_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    custom_raw = tmp_path / "custom_raw"
    monkeypatch.setenv("DATA_RAW_DIR", str(custom_raw))
    monkeypatch.setenv("BATCH_SIZE", "64")

    cfg = Settings()
    assert cfg.paths.raw_data_dir == custom_raw.resolve()
    assert cfg.ingestion.batch_size == 64


def test_invalid_config_bounds_rejected():
    with pytest.raises(ValidationError):
        IngestionConfig(batch_size=0)

    with pytest.raises(ValidationError):
        RecoveryConfig(max_retries=-1)

    with pytest.raises(ValidationError):
        RecoveryConfig(backoff_factor=0.5)


def test_ensure_directories(tmp_path: Path):
    cfg = Settings()
    cfg.paths.raw_data_dir = tmp_path / "raw"
    cfg.paths.logs_dir = tmp_path / "logs"

    cfg.ensure_directories()

    assert cfg.paths.raw_data_dir.exists()
    assert cfg.paths.logs_dir.exists()


def test_logger_writes_and_shares_file_handler(tmp_path: Path):
    log_path = tmp_path / "shared.log"
    logger_a = setup_logger(name="test.mod_a", log_file=log_path, level="DEBUG")
    logger_b = setup_logger(name="test.mod_b", log_file=log_path, level="DEBUG")

    assert logger_a.level == logging.DEBUG
    # Both loggers targeting the same file must share the exact RotatingFileHandler instance
    file_handlers_a = [h for h in logger_a.handlers if hasattr(h, "baseFilename")]
    file_handlers_b = [h for h in logger_b.handlers if hasattr(h, "baseFilename")]
    assert len(file_handlers_a) == 1
    assert file_handlers_a[0] is file_handlers_b[0]

    logger_a.info("Event from module A")
    logger_b.warning("Event from module B")

    for handler in file_handlers_a:
        handler.flush()

    content = log_path.read_text(encoding="utf-8")
    assert "Event from module A" in content
    assert "Event from module B" in content


def test_get_logger_idempotent():
    logger = get_logger("test.idempotent")
    count_before = len(logger.handlers)
    logger_again = get_logger("test.idempotent")
    assert len(logger_again.handlers) == count_before
