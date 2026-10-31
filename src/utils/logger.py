"""Centralized logging with shared file descriptors to prevent rotation locks on Windows."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional

from src.utils.config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Registry of active file handlers keyed by resolved path.
# Sharing a single RotatingFileHandler per file path avoids WinError 32
# (file in use) when multiple module loggers trigger rollover simultaneously.
_FILE_HANDLERS: Dict[Path, RotatingFileHandler] = {}


def _get_or_create_file_handler(
    log_path: Path,
    formatter: logging.Formatter,
    max_bytes: int,
    backup_count: int,
) -> RotatingFileHandler:
    resolved = log_path.resolve()
    if resolved not in _FILE_HANDLERS:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            filename=str(resolved),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,  # Defer file descriptor opening until first emit
        )
        handler.setFormatter(formatter)
        _FILE_HANDLERS[resolved] = handler
    return _FILE_HANDLERS[resolved]


def setup_logger(
    name: str = "pipeline",
    log_file: Optional[Path | str] = None,
    level: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    logger = logging.getLogger(name)

    resolved_level_str = (level or settings.logging.level).upper()
    log_level = getattr(logging, resolved_level_str, logging.INFO)
    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        target_file = (
            Path(log_file)
            if log_file is not None
            else settings.paths.logs_dir / settings.logging.log_file_name
        )
        file_handler = _get_or_create_file_handler(
            target_file, formatter, max_bytes, backup_count
        )
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("File logging disabled due to filesystem error: %s", exc)

    logger.propagate = False
    return logger


def get_logger(name: str = "pipeline") -> logging.Logger:
    return setup_logger(name=name)
