"""Data loading and caching utilities for Self-Healing Pipeline Streamlit Dashboard."""

from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger("dashboard.data_loader")


def _normalize_aspects(aspect_val: Any) -> List[str]:
    """Normalizes aspect values (numpy array, list, json string, or None) to a clean List[str]."""
    if aspect_val is None:
        return []
    if isinstance(aspect_val, (list, tuple, np.ndarray)):
        return [str(x).strip() for x in aspect_val if str(x).strip()]
    if isinstance(aspect_val, str):
        val_str = aspect_val.strip()
        if not val_str or val_str in ("[]", "None", "nan"):
            return []
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        # Fallback to comma separation
        return [x.strip(" '\"[]") for x in val_str.split(",") if x.strip(" '\"[]")]
    return []


@st.cache_data(ttl=30, show_spinner=False)
def load_all_predictions() -> pd.DataFrame:
    """Loads and aggregates all prediction files (Parquet and JSON fallback) from output/predictions."""
    pred_dir = Path(settings.paths.predictions_dir)
    if not pred_dir.exists():
        return _empty_predictions_df()

    parquet_files = list(pred_dir.glob("*.parquet"))
    json_files = list(pred_dir.glob("*.json"))

    frames: List[pd.DataFrame] = []

    # Read Parquet files
    for p_file in parquet_files:
        try:
            df = pd.read_parquet(p_file)
            if not df.empty:
                frames.append(df)
        except Exception as err:
            logger.warning("Failed to read parquet file %s: %s", p_file.name, err)

    # Read JSON files (if any exist)
    for j_file in json_files:
        try:
            df = pd.read_json(j_file)
            if not df.empty:
                frames.append(df)
        except Exception as err:
            logger.warning("Failed to read json prediction file %s: %s", j_file.name, err)

    if not frames:
        return _empty_predictions_df()

    combined_df = pd.concat(frames, ignore_index=True)

    # Normalize columns
    if "stars" in combined_df.columns:
        combined_df["stars"] = pd.to_numeric(combined_df["stars"], errors="coerce")
    else:
        combined_df["stars"] = np.nan

    if "confidence" in combined_df.columns:
        combined_df["confidence"] = pd.to_numeric(combined_df["confidence"], errors="coerce").fillna(0.0)
    else:
        combined_df["confidence"] = 0.0

    if "sentiment" in combined_df.columns:
        combined_df["sentiment"] = combined_df["sentiment"].astype(str).str.lower().str.strip()
    else:
        combined_df["sentiment"] = "neutral"

    # Normalize aspects
    if "positive_aspects" in combined_df.columns:
        combined_df["positive_aspects"] = combined_df["positive_aspects"].apply(_normalize_aspects)
    else:
        combined_df["positive_aspects"] = [[] for _ in range(len(combined_df))]

    if "negative_aspects" in combined_df.columns:
        combined_df["negative_aspects"] = combined_df["negative_aspects"].apply(_normalize_aspects)
    else:
        combined_df["negative_aspects"] = [[] for _ in range(len(combined_df))]

    # Text fallbacks
    if "text" not in combined_df.columns and "cleaned_text" in combined_df.columns:
        combined_df["text"] = combined_df["cleaned_text"]
    elif "text" not in combined_df.columns:
        combined_df["text"] = ""

    if "reasoning" not in combined_df.columns:
        combined_df["reasoning"] = "No reasoning provided"

    # Deduplicate in case overlapping files exist
    if "review_id" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=["review_id"]).reset_index(drop=True)

    return combined_df


def _empty_predictions_df() -> pd.DataFrame:
    """Returns an empty DataFrame with expected prediction columns."""
    return pd.DataFrame(
        columns=[
            "review_id",
            "business_id",
            "stars",
            "text",
            "cleaned_text",
            "sentiment",
            "confidence",
            "positive_aspects",
            "negative_aspects",
            "reasoning",
            "model_name",
            "inference_timestamp",
            "batch_id",
            "source_file",
        ]
    )


@st.cache_data(ttl=15, show_spinner=False)
def load_checkpoint_info() -> Dict[str, Any]:
    """Loads checkpoint status from checkpoints/pipeline_checkpoint.json with resilience."""
    checkpoint_file = Path(settings.paths.checkpoint_dir) / "pipeline_checkpoint.json"
    backup_file = Path(settings.paths.checkpoint_dir) / "pipeline_checkpoint.backup.json"

    target_file = None
    if checkpoint_file.exists() and checkpoint_file.stat().st_size > 0:
        target_file = checkpoint_file
    elif backup_file.exists() and backup_file.stat().st_size > 0:
        target_file = backup_file

    if target_file is None:
        return {
            "exists": False,
            "version": "None",
            "updated_at": None,
            "last_completed_batch": 0,
            "total_completed_batches": 0,
            "batches": {},
            "total_successful": 0,
            "total_failed": 0,
            "total_duplicates": 0,
            "total_invalids": 0,
            "total_records_processed": 0,
        }

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        batches = data.get("completed_batches", {})
        total_batches = len(batches)

        # Aggregate metrics across completed batches
        total_succ = 0
        total_failed = 0
        total_dups = 0
        total_invs = 0
        total_recs = 0

        for b_entry in batches.values():
            total_recs += b_entry.get("records_count", 0)
            meta = b_entry.get("metadata", {})
            total_succ += meta.get("successful_predictions", 0)
            total_failed += meta.get("failed_predictions", 0)
            total_dups += meta.get("duplicates", 0)
            total_invs += meta.get("invalids", 0)

        return {
            "exists": True,
            "file_path": str(target_file),
            "version": data.get("version", "1.0"),
            "updated_at": data.get("updated_at"),
            "last_completed_batch": data.get("last_completed_batch", 0),
            "total_completed_batches": total_batches,
            "batches": batches,
            "total_successful": total_succ,
            "total_failed": total_failed,
            "total_duplicates": total_dups,
            "total_invalids": total_invs,
            "total_records_processed": total_recs,
        }
    except Exception as err:
        logger.error("Failed to read checkpoint file %s: %s", target_file, err)
        return {
            "exists": False,
            "error": str(err),
            "last_completed_batch": 0,
            "total_completed_batches": 0,
            "batches": {},
            "total_successful": 0,
            "total_failed": 0,
            "total_duplicates": 0,
            "total_invalids": 0,
            "total_records_processed": 0,
        }


@st.cache_data(ttl=15, show_spinner=False)
def load_quarantine_stats() -> Dict[str, Any]:
    """Scans data/failed directory for validation failures, duplicates, and failed inferences."""
    failed_dir = Path(settings.paths.failed_data_dir)
    stats = {
        "validation_failed_count": 0,
        "duplicates_count": 0,
        "inference_failed_count": 0,
        "sample_validation_failures": [],
        "sample_duplicates": [],
        "sample_inference_failures": [],
    }

    if not failed_dir.exists():
        return stats

    val_files: List[Path] = []
    dup_files: List[Path] = []
    inf_files: List[Path] = []

    try:
        for entry in os.scandir(failed_dir):
            if not entry.name.endswith(".json"):
                continue
            if entry.name.startswith("validation_failed"):
                val_files.append(Path(entry.path))
            elif entry.name.startswith("duplicates"):
                dup_files.append(Path(entry.path))
            elif entry.name.startswith("inference_failed"):
                inf_files.append(Path(entry.path))

        stats["validation_failed_count"] = len(val_files)
        stats["duplicates_count"] = len(dup_files)
        stats["inference_failed_count"] = len(inf_files)

        # Load samples from the latest files for UI inspection
        def _read_sample(paths: List[Path], max_items: int = 5) -> List[Dict[str, Any]]:
            items: List[Dict[str, Any]] = []
            for p in sorted(paths, key=lambda x: x.stat().st_mtime, reverse=True)[:max_items]:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            items.extend(data[:max_items])
                        elif isinstance(data, dict):
                            items.append(data)
                except Exception:
                    pass
                if len(items) >= max_items:
                    break
            return items[:max_items]

        stats["sample_validation_failures"] = _read_sample(val_files, 5)
        stats["sample_duplicates"] = _read_sample(dup_files, 5)
        stats["sample_inference_failures"] = _read_sample(inf_files, 5)

    except Exception as err:
        logger.warning("Error reading quarantine stats from %s: %s", failed_dir, err)

    return stats


@st.cache_data(ttl=10, show_spinner=False)
def load_recent_logs(max_lines: int = 250, level_filter: str = "ALL") -> List[Dict[str, str]]:
    """Reads and parses recent log entries from logs/pipeline.log."""
    log_file = Path(settings.paths.logs_dir) / "pipeline.log"
    if not log_file.exists():
        return []

    entries: List[Dict[str, str]] = []
    pattern = re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s\|\s(?P<level>[A-Z]+)\s+\|\s(?P<logger>[\w\.:_]+)\s-\s(?P<message>.*)$"
    )

    try:
        file_size = log_file.stat().st_size
        read_size = min(file_size, max(128 * 1024, max_lines * 512))
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            if file_size > read_size:
                f.seek(file_size - read_size)
                # Discard partial first line
                f.readline()
            lines = f.readlines()

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if match:
                entry = match.groupdict()
                if level_filter == "ALL" or entry["level"] == level_filter:
                    entries.append(entry)
                    if len(entries) >= max_lines:
                        break
            else:
                # Continuation line
                if entries and (level_filter == "ALL" or entries[-1]["level"] == level_filter):
                    entries[-1]["message"] += "\n" + line
    except Exception as err:
        logger.error("Failed to read log file %s: %s", log_file, err)

    return entries


def check_ollama_status() -> Dict[str, Any]:
    """Performs a lightweight health check to determine Ollama connection status."""
    base_url = settings.ollama.base_url.rstrip("/")
    status = {
        "is_online": False,
        "base_url": base_url,
        "model": settings.ollama.model,
        "model_available": False,
        "error": None,
    }
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=0.5)
        if resp.status_code == 200:
            status["is_online"] = True
            tags = resp.json().get("models", [])
            model_names = [m.get("name") for m in tags if isinstance(m, dict)]
            status["model_available"] = any(
                settings.ollama.model in name or name.startswith(settings.ollama.model.split(":")[0])
                for name in model_names
            )
            status["available_models"] = model_names
        else:
            status["error"] = f"HTTP {resp.status_code}"
    except Exception as exc:
        status["error"] = str(exc)

    return status
