"""Reusable KPI metrics and card rendering components for Streamlit Dashboard."""

from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st


def render_sentiment_kpis(df: pd.DataFrame) -> None:
    """Renders top-level KPIs for sentiment analysis: Total, Positive, Negative, Neutral, and Avg Stars."""
    total_reviews = len(df)
    if total_reviews == 0:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Reviews", "0")
        c2.metric("Positive", "0 (0.0%)")
        c3.metric("Negative", "0 (0.0%)")
        c4.metric("Neutral", "0 (0.0%)")
        c5.metric("Avg Stars", "N/A")
        return

    pos_count = int((df["sentiment"] == "positive").sum())
    neg_count = int((df["sentiment"] == "negative").sum())
    neu_count = int((df["sentiment"] == "neutral").sum())

    pos_pct = (pos_count / total_reviews) * 100
    neg_pct = (neg_count / total_reviews) * 100
    neu_pct = (neu_count / total_reviews) * 100

    avg_stars = df["stars"].mean() if "stars" in df.columns and not df["stars"].isna().all() else None
    avg_conf = df["confidence"].mean() if "confidence" in df.columns and not df["confidence"].isna().all() else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        label="Total Reviews Processed",
        value=f"{total_reviews:,}",
        help="Total deduplicated reviews with LLM sentiment inferences",
    )
    c2.metric(
        label="Positive Reviews",
        value=f"{pos_count:,}",
        delta=f"{pos_pct:.1f}%",
        delta_color="normal",
        help="Reviews classified with positive sentiment",
    )
    c3.metric(
        label="Negative Reviews",
        value=f"{neg_count:,}",
        delta=f"{neg_pct:.1f}%",
        delta_color="inverse",
        help="Reviews classified with negative sentiment",
    )
    c4.metric(
        label="Neutral Reviews",
        value=f"{neu_count:,}",
        delta=f"{neu_pct:.1f}%",
        delta_color="off",
        help="Reviews classified with neutral sentiment",
    )
    c5.metric(
        label="Average Star Rating",
        value=f"{avg_stars:.2f} ★" if avg_stars is not None else "N/A",
        delta=f"Avg Conf: {avg_conf:.1%}" if avg_conf is not None else None,
        help="Mean user star rating across processed reviews",
    )


def render_pipeline_kpis(
    checkpoint_info: Dict[str, Any],
    quarantine_info: Dict[str, Any],
    total_predictions: int = 0,
) -> None:
    """Renders pipeline operational KPIs: Batches, Valid/Success, Failed, Quarantined, Checkpoint status."""
    total_batches = checkpoint_info.get("total_completed_batches", 0)
    last_batch = checkpoint_info.get("last_completed_batch", 0)

    # Use checkpoint numbers if available, else fallback to files
    succ_records = checkpoint_info.get("total_successful", total_predictions)
    failed_inference = checkpoint_info.get("total_failed", quarantine_info.get("inference_failed_count", 0))
    invalids = checkpoint_info.get("total_invalids", quarantine_info.get("validation_failed_count", 0))
    duplicates = checkpoint_info.get("total_duplicates", quarantine_info.get("duplicates_count", 0))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        label="Batches Committed",
        value=f"{total_batches:,}",
        delta=f"Last Batch #{last_batch}",
        help="Total batches committed atomically to checkpoint",
    )
    c2.metric(
        label="Successful Inferences",
        value=f"{succ_records:,}",
        delta="Persisted to Parquet",
        delta_color="normal",
        help="Reviews successfully analyzed and saved to Parquet predictions",
    )
    c3.metric(
        label="Failed Inferences",
        value=f"{failed_inference:,}",
        delta="Quarantined",
        delta_color="inverse",
        help="LLM inference failures or timeouts quarantined safely",
    )
    c4.metric(
        label="Invalid Records",
        value=f"{invalids:,}",
        delta="Schema Rejection",
        delta_color="inverse",
        help="Source records failing schema validation (quarantined)",
    )
    c5.metric(
        label="Duplicate Records",
        value=f"{duplicates:,}",
        delta="Deduplicated",
        delta_color="off",
        help="Duplicate review IDs identified and removed",
    )

