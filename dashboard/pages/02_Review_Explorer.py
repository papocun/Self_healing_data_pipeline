"""Page 2: Review Explorer - Interactive Review Inspector and Aspect Analysis."""

import streamlit as st

from dashboard.components.data_loader import load_all_predictions
from dashboard.components.filters import render_review_filters

# Page configuration
st.set_page_config(
    page_title="Review Explorer | Self-Healing Pipeline",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Review Explorer & Aspect Inspector")
st.caption("Inspect individual reviews, verify Ollama sentiment predictions, and explore extracted positive/negative aspects.")

# Refresh
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Load real dataset
df = load_all_predictions()

if df.empty:
    st.warning("⚠️ No prediction data found in `output/predictions/`. Please run the pipeline first.")
    st.stop()

# Apply filters
filtered_df, search_term = render_review_filters(df)

# Results header & metrics
total_matches = len(filtered_df)
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    search_notice = f" matching **'{search_term}'**" if search_term else ""
    st.subheader(f"Showing {total_matches:,} of {len(df):,} reviews{search_notice}")
with c2:
    view_mode = st.radio("Display View", options=["Detailed Cards", "Compact Table"], horizontal=True)
with c3:
    max_display = st.selectbox("Page Size", options=[10, 25, 50, 100], index=0)

if total_matches == 0:
    st.info("No reviews match the selected filter criteria. Try adjusting the sliders or clearing your search term.")
    st.stop()

# Export options
with st.sidebar:
    st.divider()
    st.subheader("📥 Export Filtered Data")
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download as CSV",
        data=csv_data,
        file_name="filtered_sentiment_reviews.csv",
        mime="text/csv",
        use_container_width=True,
    )
    json_data = filtered_df.to_json(orient="records", indent=2)
    st.download_button(
        label="Download as JSON",
        data=json_data,
        file_name="filtered_sentiment_reviews.json",
        mime="application/json",
        use_container_width=True,
    )

# Sentiment styling helper
def _sentiment_badge(sentiment: str) -> str:
    colors = {
        "positive": ("#10B981", "#ECFDF5"),
        "negative": ("#EF4444", "#FEF2F2"),
        "neutral": ("#6B7280", "#F3F4F6"),
    }
    border, bg = colors.get(sentiment.lower(), ("#6B7280", "#F3F4F6"))
    return (
        f'<span style="background-color: {bg}; color: {border}; padding: 3px 10px; '
        f'border-radius: 12px; font-weight: 600; border: 1px solid {border}; font-size: 0.85rem;">'
        f'{sentiment.upper()}</span>'
    )


# Display Mode: Compact Table
if view_mode == "Compact Table":
    display_cols = [
        col for col in [
            "review_id", "sentiment", "confidence", "stars", "text", "reasoning", "model_name", "batch_id"
        ] if col in filtered_df.columns
    ]
    st.dataframe(
        filtered_df[display_cols].head(max_display),
        use_container_width=True,
        height=500,
    )

# Display Mode: Detailed Cards
else:
    for idx, row in filtered_df.head(max_display).iterrows():
        sentiment = str(row.get("sentiment", "neutral")).lower()
        stars = row.get("stars")
        stars_str = f"{stars:.1f} ★" if pd.notna(stars) else "N/A"
        confidence = row.get("confidence", 0.0)
        review_id = row.get("review_id", f"Row-{idx}")
        biz_id = row.get("business_id", "N/A")
        text = row.get("text", "")
        reasoning = row.get("reasoning", "No explanation provided.")
        pos_aspects = row.get("positive_aspects", [])
        neg_aspects = row.get("negative_aspects", [])
        model_name = row.get("model_name", "qwen2.5:3b")
        source = row.get("source_file", "unknown")
        batch_id = row.get("batch_id", "N/A")

        with st.container():
            col_header, col_meta = st.columns([3, 1])
            with col_header:
                badge_html = _sentiment_badge(sentiment)
                st.markdown(
                    f"#### Review `{review_id}` &nbsp; {badge_html} &nbsp; `Rating: {stars_str}`",
                    unsafe_allow_html=True,
                )
            with col_meta:
                st.markdown(
                    f"<div style='text-align: right; color: #64748B; font-size: 0.85rem;'>"
                    f"Confidence: <b>{confidence:.1%}</b> | Batch #{batch_id}</div>",
                    unsafe_allow_html=True,
                )

            # Review Text Block
            st.markdown(
                f"<div style='background-color: #F8FAFC; padding: 12px 16px; border-left: 4px solid #3B82F6; "
                f"border-radius: 4px; font-style: italic; margin-bottom: 8px;'>\"{text}\"</div>",
                unsafe_allow_html=True,
            )

            # Model Reasoning & Aspects
            r_col1, r_col2 = st.columns([1, 1])
            with r_col1:
                st.markdown(f"**🧠 Model Reasoning ({model_name}):**")
                st.write(reasoning)

            with r_col2:
                # Positive Aspects tags
                if pos_aspects:
                    pos_tags = " ".join(
                        [
                            f"<span style='background-color: #DCFCE7; color: #166534; padding: 2px 8px; "
                            f"border-radius: 4px; font-size: 0.8rem; margin: 2px;'>+{a}</span>"
                            for a in pos_aspects
                        ]
                    )
                    st.markdown(f"**✨ Positive Aspects:**<br>{pos_tags}", unsafe_allow_html=True)
                else:
                    st.markdown("**✨ Positive Aspects:** _None detected_")

                # Negative Aspects tags
                if neg_aspects:
                    neg_tags = " ".join(
                        [
                            f"<span style='background-color: #FEE2E2; color: #991B1B; padding: 2px 8px; "
                            f"border-radius: 4px; font-size: 0.8rem; margin: 2px;'>-{a}</span>"
                            for a in neg_aspects
                        ]
                    )
                    st.markdown(f"<div style='margin-top: 6px;'><b>⚠️ Negative Aspects:</b><br>{neg_tags}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='margin-top: 6px;'><b>⚠️ Negative Aspects:</b> <i>None detected</i></div>", unsafe_allow_html=True)

            with st.expander(f"Metadata Details ({review_id})", expanded=False):
                meta_cols = st.columns(4)
                meta_cols[0].write(f"**Business ID:** `{biz_id}`")
                meta_cols[1].write(f"**Source File:** `{source}`")
                meta_cols[2].write(f"**Timestamp:** `{row.get('inference_timestamp', 'N/A')}`")
                meta_cols[3].write(f"**Date:** `{row.get('date', 'N/A')}`")

            st.divider()

