"""Page 3: Pipeline Monitor - System Health, Recovery Checkpoints, and Execution Logs."""

from datetime import datetime
import streamlit as st

from dashboard.components.charts import create_pipeline_records_breakdown_chart
from dashboard.components.data_loader import (
    check_ollama_status,
    load_all_predictions,
    load_checkpoint_info,
    load_quarantine_stats,
    load_recent_logs,
)
from dashboard.components.metrics import render_pipeline_kpis

# Page configuration
st.set_page_config(
    page_title="Pipeline Monitor | Self-Healing Pipeline",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Pipeline Execution & Self-Healing Monitor")
st.caption("Live monitoring of batch checkpoints, retry resilience, quarantine stores, and system execution logs.")

# Sidebar Refresh
col_side1, col_side2 = st.sidebar.columns(2)
with col_side1:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with col_side2:
    auto_refresh = st.checkbox("Auto", value=False, help="Enable automatic polling")

if auto_refresh:
    # Minimal Streamlit auto-rerun interval simulation
    pass

# Load operational artifacts
predictions_df = load_all_predictions()
checkpoint = load_checkpoint_info()
quarantine = load_quarantine_stats()
ollama_info = check_ollama_status()

# 1. System Health Status Banner
c_stat1, c_stat2, c_stat3 = st.columns(3)

with c_stat1:
    if ollama_info["is_online"]:
        model_badge = "Available" if ollama_info["model_available"] else "Model Not Found"
        st.success(f"🟢 **Ollama Service:** Online (`{ollama_info['model']}`: {model_badge})")
    else:
        st.error(f"🔴 **Ollama Service:** Offline ({ollama_info['error'] or 'Connection Refused'})")

with c_stat2:
    if checkpoint["exists"]:
        updated_str = checkpoint.get("updated_at")
        if updated_str:
            try:
                dt = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                formatted_dt = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                formatted_dt = updated_str
        else:
            formatted_dt = "Unknown"
        st.success(f"💾 **Checkpoint Store:** Synchronized (Updated: {formatted_dt})")
    else:
        st.warning("⚠️ **Checkpoint Store:** Uninitialized or Missing")

with c_stat3:
    total_q = quarantine["validation_failed_count"] + quarantine["duplicates_count"] + quarantine["inference_failed_count"]
    if total_q > 0:
        st.warning(f"📦 **Quarantine Vault:** {total_q:,} batches isolated")
    else:
        st.info("📦 **Quarantine Vault:** Clean (0 isolated batches)")

st.divider()

# 2. Key Operational Metrics
render_pipeline_kpis(
    checkpoint_info=checkpoint,
    quarantine_info=quarantine,
    total_predictions=len(predictions_df),
)

st.divider()

# 3. Checkpoint Details & Records Breakdown Chart
ch_col1, ch_col2 = st.columns([1, 1])

with ch_col1:
    records_chart = create_pipeline_records_breakdown_chart(checkpoint, quarantine)
    st.plotly_chart(records_chart, use_container_width=True)

with ch_col2:
    st.subheader("📌 Checkpoint State & Recovery Meta")
    if checkpoint["exists"]:
        st.markdown(
            f"""
            - **Storage File:** `{checkpoint.get('file_path', 'pipeline_checkpoint.json')}`
            - **Schema Version:** `{checkpoint.get('version', '1.0')}`
            - **Last Completed Batch ID:** `#{checkpoint.get('last_completed_batch', 0)}`
            - **Total Committed Batches:** `{checkpoint.get('total_completed_batches', 0):,}`
            - **Atomic Persistence:** Enabled (`.tmp` write + atomic OS rename)
            - **Resumption Behavior:** Fast-forward skips committed batches deterministically.
            """
        )
        # Recent batches table
        batches_dict = checkpoint.get("batches", {})
        if batches_dict:
            latest_batch_items = list(batches_dict.items())[-10:]
            recent_table = []
            for k, b in reversed(latest_batch_items):
                meta = b.get("metadata", {})
                recent_table.append(
                    {
                        "Source / Batch": k,
                        "Batch ID": b.get("batch_id"),
                        "Range [Start:End]": f"[{b.get('start_offset')}:{b.get('end_offset')}]",
                        "Status": b.get("processing_status"),
                        "Success": meta.get("successful_predictions", 0),
                        "Failed": meta.get("failed_predictions", 0),
                        "Duplicates": meta.get("duplicates", 0),
                        "Timestamp": b.get("timestamp", "")[:19],
                    }
                )
            st.caption("Latest 10 Committed Batches:")
            st.dataframe(recent_table, use_container_width=True, height=220)
    else:
        st.info("No checkpoint records loaded yet.")

st.divider()

# 4. Quarantine Vault Inspector (Tabs)
st.subheader("📦 Quarantined Records Inspector")
st.caption("Inspect and audit records safely isolated by schema validators, deduplication guards, and inference handlers.")

tab_val, tab_dup, tab_inf = st.tabs(
    [
        f"Validation Failures ({quarantine['validation_failed_count']})",
        f"Duplicate Records ({quarantine['duplicates_count']})",
        f"Inference Failures ({quarantine['inference_failed_count']})",
    ]
)

with tab_val:
    if quarantine["sample_validation_failures"]:
        st.write("Recent sample records that failed schema or domain constraints:")
        st.json(quarantine["sample_validation_failures"])
    else:
        st.info("No validation failure records found.")

with tab_dup:
    if quarantine["sample_duplicates"]:
        st.write("Recent sample records quarantined during deduplication:")
        st.json(quarantine["sample_duplicates"])
    else:
        st.info("No duplicate records found.")

with tab_inf:
    if quarantine["sample_inference_failures"]:
        st.write("Recent sample records quarantined due to LLM errors or timeouts:")
        st.json(quarantine["sample_inference_failures"][:5])
    else:
        st.info("No inference failure records found.")

st.divider()

# 5. Live Pipeline Logs Viewer
st.subheader("📜 Recent Pipeline Execution Logs")
log_cols = st.columns([1, 1, 2])
with log_cols[0]:
    level_filter = st.selectbox("Log Level", options=["ALL", "INFO", "WARNING", "ERROR"], index=0)
with log_cols[1]:
    max_lines = st.selectbox("Max Entries", options=[50, 100, 250, 500], index=1)
with log_cols[2]:
    log_search = st.text_input("Search Logs", placeholder="Filter by keyword or review ID...").strip()

logs = load_recent_logs(max_lines=max_lines, level_filter=level_filter)

if log_search:
    logs = [entry for entry in logs if log_search.lower() in str(entry).lower()]

if logs:
    log_display_text = "\n".join(
        [
            f"[{entry['timestamp']}] {entry['level']:<7} | {entry['logger']} - {entry['message']}"
            for entry in logs
        ]
    )
    st.text_area("Pipeline Log Stream", value=log_display_text, height=350, disabled=True)
else:
    st.info(f"No log entries found matching criteria (Level={level_filter}).")

