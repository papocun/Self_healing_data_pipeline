"""Unit tests for the Streamlit dashboard components, data loaders, charts, and filters."""

from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from dashboard.components.charts import (
    create_confidence_distribution_chart,
    create_pipeline_records_breakdown_chart,
    create_sentiment_by_stars_chart,
    create_sentiment_donut_chart,
    create_top_aspects_bar_chart,
)
from dashboard.components.data_loader import (
    _empty_predictions_df,
    _normalize_aspects,
    load_all_predictions,
    load_checkpoint_info,
    load_quarantine_stats,
    load_recent_logs,
)
from src.utils.config import settings


def test_normalize_aspects_varieties():
    assert _normalize_aspects(None) == []
    assert _normalize_aspects("") == []
    assert _normalize_aspects("[]") == []
    assert _normalize_aspects(["fresh bread", "great coffee"]) == ["fresh bread", "great coffee"]
    assert _normalize_aspects(np.array(["service", "pricing"])) == ["service", "pricing"]
    assert _normalize_aspects('["almond croissants", "sourdough"]') == ["almond croissants", "sourdough"]
    assert _normalize_aspects("fast delivery, polite staff") == ["fast delivery", "polite staff"]


def test_empty_predictions_dataframe():
    df = _empty_predictions_df()
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    expected_cols = [
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
    for col in expected_cols:
        assert col in df.columns


def test_load_all_predictions_real_or_fallback(monkeypatch, tmp_path):
    # Test reading actual or simulated parquet predictions
    monkeypatch.setattr(settings.paths, "predictions_dir", tmp_path / "predictions")
    (tmp_path / "predictions").mkdir(parents=True)

    # When directory is empty
    empty_df = load_all_predictions()
    assert empty_df.empty

    # Create dummy parquet
    dummy_data = pd.DataFrame(
        [
            {
                "review_id": "rev_01",
                "business_id": "biz_01",
                "stars": 5.0,
                "text": "Amazing coffee and bakery!",
                "cleaned_text": "amazing coffee and bakery",
                "sentiment": "positive",
                "confidence": 0.95,
                "positive_aspects": ["coffee", "bakery"],
                "negative_aspects": [],
                "reasoning": "High praise for products",
                "model_name": "qwen2.5:3b",
                "inference_timestamp": datetime.now(timezone.utc).isoformat(),
                "batch_id": 1,
                "source_file": "test.json",
            },
            {
                "review_id": "rev_02",
                "business_id": "biz_02",
                "stars": 1.0,
                "text": "Terrible service and cold food.",
                "cleaned_text": "terrible service and cold food",
                "sentiment": "negative",
                "confidence": 0.90,
                "positive_aspects": [],
                "negative_aspects": ["service", "cold food"],
                "reasoning": "Unacceptable experience",
                "model_name": "qwen2.5:3b",
                "inference_timestamp": datetime.now(timezone.utc).isoformat(),
                "batch_id": 1,
                "source_file": "test.json",
            },
        ]
    )
    dummy_data.to_parquet(tmp_path / "predictions" / "predictions_batch_00001.parquet")

    # Clear streamlit cache and reload
    load_all_predictions.clear()
    loaded_df = load_all_predictions()

    assert len(loaded_df) == 2
    assert set(loaded_df["sentiment"]) == {"positive", "negative"}
    assert loaded_df.iloc[0]["positive_aspects"] == ["coffee", "bakery"]
    assert loaded_df.iloc[1]["negative_aspects"] == ["service", "cold food"]


def test_load_checkpoint_info(monkeypatch, tmp_path):
    monkeypatch.setattr(settings.paths, "checkpoint_dir", tmp_path / "checkpoints")
    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir(parents=True)

    # Empty / non-existent checkpoint
    load_checkpoint_info.clear()
    info = load_checkpoint_info()
    assert info["exists"] is False
    assert info["total_completed_batches"] == 0

    # Populated checkpoint
    cp_file = cp_dir / "pipeline_checkpoint.json"
    checkpoint_data = {
        "version": "1.0",
        "updated_at": "2026-11-01T00:00:00Z",
        "last_completed_batch": 5,
        "completed_batches": {
            "file.json:1": {
                "batch_id": 1,
                "records_count": 10,
                "processing_status": "COMPLETED",
                "metadata": {"successful_predictions": 8, "failed_predictions": 2, "duplicates": 0, "invalids": 0},
            },
            "file.json:2": {
                "batch_id": 2,
                "records_count": 10,
                "processing_status": "COMPLETED",
                "metadata": {"successful_predictions": 10, "failed_predictions": 0, "duplicates": 1, "invalids": 1},
            },
        },
    }
    with open(cp_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f)

    load_checkpoint_info.clear()
    info = load_checkpoint_info()
    assert info["exists"] is True
    assert info["total_completed_batches"] == 2
    assert info["last_completed_batch"] == 5
    assert info["total_successful"] == 18
    assert info["total_failed"] == 2
    assert info["total_duplicates"] == 1
    assert info["total_invalids"] == 1


def test_load_quarantine_stats(monkeypatch, tmp_path):
    failed_dir = tmp_path / "failed"
    failed_dir.mkdir(parents=True)
    monkeypatch.setattr(settings.paths, "failed_data_dir", failed_dir)

    # Write dummy quarantine files
    with open(failed_dir / "validation_failed_batch_001.json", "w", encoding="utf-8") as f:
        json.dump([{"review_id": "bad_01", "error": "invalid rating"}], f)
    with open(failed_dir / "duplicates_batch_001.json", "w", encoding="utf-8") as f:
        json.dump([{"review_id": "dup_01"}], f)
    with open(failed_dir / "inference_failed_batch_001.json", "w", encoding="utf-8") as f:
        json.dump([{"review_id": "inf_bad_01", "error": "timeout"}], f)

    load_quarantine_stats.clear()
    stats = load_quarantine_stats()

    assert stats["validation_failed_count"] == 1
    assert stats["duplicates_count"] == 1
    assert stats["inference_failed_count"] == 1
    assert len(stats["sample_validation_failures"]) == 1
    assert stats["sample_validation_failures"][0]["review_id"] == "bad_01"


def test_load_recent_logs(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    monkeypatch.setattr(settings.paths, "logs_dir", log_dir)

    log_file = log_dir / "pipeline.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("2026-11-01 00:01:00 | INFO     | pipeline.orchestrator:run:10 - Pipeline started\n")
        f.write("2026-11-01 00:01:05 | WARNING  | pipeline.validator:check:20 - Schema warning\n")
        f.write("2026-11-01 00:01:10 | ERROR    | pipeline.inference:run:30 - Connection timeout\n")

    load_recent_logs.clear()
    all_logs = load_recent_logs(max_lines=10, level_filter="ALL")
    assert len(all_logs) == 3

    load_recent_logs.clear()
    err_logs = load_recent_logs(max_lines=10, level_filter="ERROR")
    assert len(err_logs) == 1
    assert err_logs[0]["message"] == "Connection timeout"


def test_charts_empty_data_handling():
    empty_df = _empty_predictions_df()

    donut = create_sentiment_donut_chart(empty_df)
    assert donut is not None

    stars_bar = create_sentiment_by_stars_chart(empty_df)
    assert stars_bar is not None

    conf_hist = create_confidence_distribution_chart(empty_df)
    assert conf_hist is not None

    top_pos = create_top_aspects_bar_chart(empty_df, aspect_type="positive")
    assert top_pos is not None

    top_neg = create_top_aspects_bar_chart(empty_df, aspect_type="negative")
    assert top_neg is not None

    breakdown = create_pipeline_records_breakdown_chart({}, {})
    assert breakdown is not None


def test_charts_with_sample_data():
    df = pd.DataFrame(
        [
            {
                "stars": 5.0,
                "sentiment": "positive",
                "confidence": 0.95,
                "positive_aspects": ["great service", "tasty food"],
                "negative_aspects": [],
            },
            {
                "stars": 1.0,
                "sentiment": "negative",
                "confidence": 0.85,
                "positive_aspects": [],
                "negative_aspects": ["slow service"],
            },
            {
                "stars": 3.0,
                "sentiment": "neutral",
                "confidence": 0.60,
                "positive_aspects": ["okay price"],
                "negative_aspects": ["loud noise"],
            },
        ]
    )

    donut = create_sentiment_donut_chart(df)
    assert len(donut.data) > 0

    stars_bar = create_sentiment_by_stars_chart(df)
    assert len(stars_bar.data) > 0

    conf_hist = create_confidence_distribution_chart(df)
    assert len(conf_hist.data) > 0

    top_pos = create_top_aspects_bar_chart(df, aspect_type="positive")
    assert len(top_pos.data) > 0

    top_neg = create_top_aspects_bar_chart(df, aspect_type="negative")
    assert len(top_neg.data) > 0


def test_metrics_rendering_mocked(monkeypatch):
    from unittest.mock import MagicMock
    import streamlit as st
    from dashboard.components.metrics import render_pipeline_kpis, render_sentiment_kpis

    mock_col = MagicMock()
    monkeypatch.setattr(st, "columns", lambda n: [mock_col] * n)

    # Empty df
    render_sentiment_kpis(_empty_predictions_df())
    assert mock_col.metric.called

    # Populated df
    sample_df = pd.DataFrame(
        [
            {"sentiment": "positive", "stars": 5.0, "confidence": 0.95},
            {"sentiment": "negative", "stars": 1.0, "confidence": 0.80},
        ]
    )
    render_sentiment_kpis(sample_df)

    # Pipeline KPIs
    render_pipeline_kpis(
        checkpoint_info={"total_completed_batches": 10, "last_completed_batch": 10, "total_successful": 100},
        quarantine_info={"validation_failed_count": 2, "duplicates_count": 1, "inference_failed_count": 0},
    )
    assert mock_col.metric.call_count >= 10

