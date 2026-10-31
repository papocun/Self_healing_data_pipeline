"""Shared pytest fixtures for pipeline unit and integration tests."""

import os
from pathlib import Path
import pandas as pd
import pytest

os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"review_id": "rev_001", "text": "This product works exceptionally well!", "rating": 5},
            {"review_id": "rev_002", "text": "Terrible customer experience and late delivery.", "rating": 1},
            {"review_id": "rev_003", "text": "Average product, neither good nor bad.", "rating": 3},
            {"review_id": "rev_004", "text": "", "rating": 2},
            {"review_id": "rev_001", "text": "Duplicate entry", "rating": 5},
        ]
    )


@pytest.fixture
def mock_ollama_sentiment_response() -> dict:
    return {
        "sentiment": "POSITIVE",
        "confidence": 0.96,
        "aspects": ["performance", "reliability"],
    }


@pytest.fixture
def temp_pipeline_workspace(tmp_path: Path) -> dict[str, Path]:
    dirs = {
        "raw": tmp_path / "data" / "raw",
        "processed": tmp_path / "data" / "processed",
        "failed": tmp_path / "data" / "failed",
        "checkpoints": tmp_path / "checkpoints",
        "output_predictions": tmp_path / "output" / "predictions",
        "output_analytics": tmp_path / "output" / "analytics",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs
