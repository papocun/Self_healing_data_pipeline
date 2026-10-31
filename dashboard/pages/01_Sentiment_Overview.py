"""Page 1: Sentiment Overview - Aggregated Analytics and Distribution."""

import streamlit as st

from dashboard.components.charts import (
    create_confidence_distribution_chart,
    create_sentiment_by_stars_chart,
    create_sentiment_donut_chart,
    create_top_aspects_bar_chart,
)
from dashboard.components.data_loader import load_all_predictions, load_checkpoint_info
from dashboard.components.metrics import render_sentiment_kpis

# Page configuration
st.set_page_config(
    page_title="Sentiment Overview | Self-Healing Pipeline",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Sentiment Analytics & Overview")
st.caption("Aggregated LLM sentiment classifications, star rating correlation, and aspect extraction.")

# Sidebar Refresh
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Load real dataset
df = load_all_predictions()
checkpoint = load_checkpoint_info()

if df.empty:
    st.warning(
        "⚠️ No prediction data found in `output/predictions/`. "
        "Run the pipeline or trigger the Airflow DAG to process reviews."
    )
    if checkpoint.get("exists"):
        st.info(
            f"ℹ️ Checkpoint exists with {checkpoint.get('total_completed_batches', 0):,} batches committed, "
            f"but no parquet prediction files are currently populated."
        )
    st.stop()

# 1. Top KPI Metrics
render_sentiment_kpis(df)
st.divider()

# 2. Main Distribution Charts (2 Columns)
col1, col2 = st.columns([1, 1])

with col1:
    donut_fig = create_sentiment_donut_chart(df)
    st.plotly_chart(donut_fig, use_container_width=True)

with col2:
    stars_fig = create_sentiment_by_stars_chart(df)
    st.plotly_chart(stars_fig, use_container_width=True)

st.divider()

# 3. Model Confidence & Key Aspects
col3, col4, col5 = st.columns([1, 1, 1])

with col3:
    conf_fig = create_confidence_distribution_chart(df)
    st.plotly_chart(conf_fig, use_container_width=True)

with col4:
    pos_aspects_fig = create_top_aspects_bar_chart(df, aspect_type="positive")
    st.plotly_chart(pos_aspects_fig, use_container_width=True)

with col5:
    neg_aspects_fig = create_top_aspects_bar_chart(df, aspect_type="negative")
    st.plotly_chart(neg_aspects_fig, use_container_width=True)

# 4. Summary Statistics Table
with st.expander("📋 View Summary Table by Star Rating", expanded=False):
    if "stars" in df.columns:
        summary_table = (
            df.groupby(["stars", "sentiment"])
            .agg(
                review_count=("review_id", "count"),
                avg_confidence=("confidence", "mean"),
            )
            .reset_index()
        )
        summary_table["avg_confidence"] = summary_table["avg_confidence"].apply(lambda x: f"{x:.1%}")
        st.dataframe(summary_table, use_container_width=True)
    else:
        st.write("Star rating column not available.")

