"""Dashboard components package for Self-Healing Pipeline."""

from dashboard.components.charts import (
    create_confidence_distribution_chart,
    create_pipeline_records_breakdown_chart,
    create_sentiment_by_stars_chart,
    create_sentiment_donut_chart,
    create_top_aspects_bar_chart,
)
from dashboard.components.data_loader import (
    check_ollama_status,
    load_all_predictions,
    load_checkpoint_info,
    load_quarantine_stats,
    load_recent_logs,
)
from dashboard.components.filters import render_review_filters
from dashboard.components.metrics import render_pipeline_kpis, render_sentiment_kpis

__all__ = [
    "create_confidence_distribution_chart",
    "create_pipeline_records_breakdown_chart",
    "create_sentiment_by_stars_chart",
    "create_sentiment_donut_chart",
    "create_top_aspects_bar_chart",
    "check_ollama_status",
    "load_all_predictions",
    "load_checkpoint_info",
    "load_quarantine_stats",
    "load_recent_logs",
    "render_review_filters",
    "render_pipeline_kpis",
    "render_sentiment_kpis",
]

