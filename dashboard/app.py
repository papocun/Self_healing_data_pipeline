"""Self-Healing Pipeline Dashboard - Executive Landing Page and Navigation Hub."""

import streamlit as st

from dashboard.components.data_loader import (
    check_ollama_status,
    load_all_predictions,
    load_checkpoint_info,
    load_quarantine_stats,
)
from dashboard.components.metrics import render_pipeline_kpis, render_sentiment_kpis

# Streamlit Multi-page App Configuration
st.set_page_config(
    page_title="Self-Healing Pipeline Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Self-Healing Sentiment Analysis Pipeline")
st.markdown(
    """
    **Autonomous Yelp Review Sentiment Analysis & Monitoring Platform**  
    This dashboard provides real-time visibility into the end-to-end data pipeline, featuring
    local LLM sentiment classification (*qwen2.5:3b*), self-healing recovery checkpoints,
    schema quarantine vaults, and interactive review exploration.
    """
)

# Sidebar Actions
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg",
        width=50,
    )
    st.markdown("### **Self-Healing Pipeline**")
    st.caption("Version 1.0.0 | Production Architecture")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("**Navigation Pages:**")
    st.markdown("- **01 Sentiment Overview**: Aggregate metrics & charts")
    st.markdown("- **02 Review Explorer**: Search & aspect analysis")
    st.markdown("- **03 Pipeline Monitor**: Health, retries & checkpoints")

# Load pipeline data
predictions_df = load_all_predictions()
checkpoint = load_checkpoint_info()
quarantine = load_quarantine_stats()
ollama_status = check_ollama_status()

# 1. System Health Status Banner
c1, c2, c3 = st.columns(3)
with c1:
    if ollama_status["is_online"]:
        st.success(f"🟢 **Ollama LLM:** Online (`{ollama_status['model']}`)")
    else:
        st.error(f"🔴 **Ollama LLM:** Offline ({ollama_status['error'] or 'Unreachable'})")

with c2:
    if checkpoint["exists"]:
        st.success(f"💾 **Checkpoint:** Active ({checkpoint['total_completed_batches']:,} batches)")
    else:
        st.warning("⚠️ **Checkpoint:** Not Found")

with c3:
    total_quarantined = (
        quarantine["validation_failed_count"]
        + quarantine["duplicates_count"]
        + quarantine["inference_failed_count"]
    )
    st.info(f"📦 **Quarantine Vault:** {total_quarantined:,} isolated items")

st.divider()

# 2. Executive Metrics
st.subheader("📈 Executive Metrics Summary")
render_sentiment_kpis(predictions_df)

st.markdown("<br>", unsafe_allow_html=True)
render_pipeline_kpis(
    checkpoint_info=checkpoint,
    quarantine_info=quarantine,
    total_predictions=len(predictions_df),
)

st.divider()

# 3. Interactive Navigation Guide
st.subheader("🧭 Dashboard Navigation")
nav1, nav2, nav3 = st.columns(3)

with nav1:
    st.markdown("### 📊 01. Sentiment Overview")
    st.write(
        "Explore sentiment distributions, star rating correlations, confidence histograms, "
        "and top positive/negative customer aspects extracted by the LLM."
    )
    st.page_link(
        "pages/01_Sentiment_Overview.py",
        label="Go to Sentiment Overview →",
        icon="📊",
    )

with nav2:
    st.markdown("### 🔎 02. Review Explorer")
    st.write(
        "Search through individual reviews, filter by sentiment or rating, inspect model reasoning, "
        "and export filtered subsets to CSV or JSON."
    )
    st.page_link(
        "pages/02_Review_Explorer.py",
        label="Go to Review Explorer →",
        icon="🔎",
    )

with nav3:
    st.markdown("### 🛡️ 03. Pipeline Monitor")
    st.write(
        "Inspect atomic checkpoints, batch execution status, retry resilience mechanisms, "
        "quarantined error payloads, and live system log streams."
    )
    st.page_link(
        "pages/03_Pipeline_Monitor.py",
        label="Go to Pipeline Monitor →",
        icon="🛡️",
    )

st.divider()

# 4. Pipeline Architecture Reference
with st.expander("🏗️ Self-Healing Pipeline Architecture", expanded=False):
    st.markdown(
        """
        ```
        Raw Review Stream (JSONL / Parquet)
                       │
                       ▼
        ┌─────────────────────────────┐
        │     Schema Validation       │ ──► [Quarantine: validation_failed]
        └──────────────┬──────────────┘
                       │ Valid
                       ▼
        ┌─────────────────────────────┐
        │       Text Cleaning         │ (Whitespace, unicode normalization)
        └──────────────┬──────────────┘
                       │
                       ▼
        ┌─────────────────────────────┐
        │     Deduplication Guard     │ ──► [Quarantine: duplicates]
        └──────────────┬──────────────┘
                       │ Unique
                       ▼
        ┌─────────────────────────────┐
        │   Ollama Sentiment Engine   │ ◄──► [Exponential Backoff Retries]
        └──────────────┬──────────────┘         │ Exhausted / Malformed
                       │ Success                ▼
                       ▼                [Quarantine: inference_failed]
        ┌─────────────────────────────┐
        │ Deterministic Parquet Store │ (output/predictions/)
        └──────────────┬──────────────┘
                       │ Output Committed
                       ▼
        ┌─────────────────────────────┐
        │  Atomic Checkpoint Commit   │ (checkpoints/pipeline_checkpoint.json)
        └─────────────────────────────┘
        ```
        """
    )

