"""Runtime configuration and environment bindings."""

import os
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(env_key: str, default_relative: str) -> Path:
    """Resolve path from env var; anchors relative paths to PROJECT_ROOT."""
    raw = os.getenv(env_key, default_relative).strip()
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class PathConfig(BaseModel):
    root_dir: Path = PROJECT_ROOT
    raw_data_dir: Path = Field(default_factory=lambda: _resolve_path("DATA_RAW_DIR", "data/raw"))
    processed_data_dir: Path = Field(default_factory=lambda: _resolve_path("DATA_PROCESSED_DIR", "data/processed"))
    failed_data_dir: Path = Field(default_factory=lambda: _resolve_path("DATA_FAILED_DIR", "data/failed"))
    checkpoint_dir: Path = Field(default_factory=lambda: _resolve_path("CHECKPOINT_DIR", "checkpoints"))
    predictions_dir: Path = Field(default_factory=lambda: _resolve_path("OUTPUT_PREDICTIONS_DIR", "output/predictions"))
    analytics_dir: Path = Field(default_factory=lambda: _resolve_path("OUTPUT_ANALYTICS_DIR", "output/analytics"))
    logs_dir: Path = Field(default_factory=lambda: _resolve_path("LOGS_DIR", "logs"))


class IngestionConfig(BaseModel):
    batch_size: int = Field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", "32")),
        ge=1,
        le=100_000,
    )
    max_file_size_mb: int = Field(
        default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "500")),
        ge=1,
    )
    supported_extensions: Tuple[str, ...] = (".csv", ".json", ".jsonl", ".parquet")
    # utf-8-sig transparently strips BOM markers common in Excel/Windows exports
    default_encoding: str = "utf-8-sig"
    fallback_encodings: Tuple[str, ...] = ("utf-8", "latin1", "cp1252", "iso-8859-1")


class OllamaConfig(BaseModel):
    base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"))
    model: str = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3").strip())
    timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT", "60.0")),
        gt=0.0,
    )


class RecoveryConfig(BaseModel):
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")),
        ge=0,
        le=20,
    )
    retry_delay_seconds: float = Field(
        default_factory=lambda: float(os.getenv("RETRY_DELAY_SECONDS", "2.0")),
        gt=0.0,
    )
    backoff_factor: float = Field(
        default_factory=lambda: float(os.getenv("BACKOFF_FACTOR", "2.0")),
        ge=1.0,
    )


class LoggingConfig(BaseModel):
    level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").strip().upper())
    log_file_name: str = Field(default_factory=lambda: os.getenv("LOG_FILE_NAME", "pipeline.log").strip())


class Settings(BaseModel):
    paths: PathConfig = Field(default_factory=PathConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def ensure_directories(self) -> None:
        for directory in (
            self.paths.raw_data_dir,
            self.paths.processed_data_dir,
            self.paths.failed_data_dir,
            self.paths.checkpoint_dir,
            self.paths.predictions_dir,
            self.paths.analytics_dir,
            self.paths.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
