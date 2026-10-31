"""Filtering and search controls for the Streamlit Review Explorer."""

from typing import Tuple
import pandas as pd
import streamlit as st


def render_review_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Renders sidebar filters for sentiment, star rating, confidence, source file, and text search.

    Returns the filtered DataFrame and the search query string.
    """
    if df.empty:
        return df, ""

    st.sidebar.header("🔍 Filter Reviews")

    # 1. Text Search Box
    search_query = st.sidebar.text_input(
        "Search in Review Text",
        value="",
        placeholder="e.g. croissants, rude, noisy, wait...",
        help="Filters reviews containing the specified keyword (case-insensitive)",
    ).strip()

    # 2. Sentiment Filter
    available_sentiments = sorted(list(df["sentiment"].unique()))
    selected_sentiments = st.sidebar.multiselect(
        "Sentiment Class",
        options=available_sentiments,
        default=available_sentiments,
        help="Select one or more sentiments to display",
    )

    # 3. Star Rating Filter
    has_stars = "stars" in df.columns and not df["stars"].isna().all()
    if has_stars:
        min_star = float(df["stars"].min())
        max_star = float(df["stars"].max())
        if min_star < max_star:
            star_range = st.sidebar.slider(
                "Star Rating Range",
                min_value=min_star,
                max_value=max_star,
                value=(min_star, max_star),
                step=0.5,
            )
        else:
            star_range = (min_star, max_star)
    else:
        star_range = (0.0, 5.0)

    # 4. Confidence Score Filter
    has_conf = "confidence" in df.columns and not df["confidence"].isna().all()
    if has_conf:
        min_c = float(df["confidence"].min())
        max_c = float(df["confidence"].max())
        if min_c < max_c:
            conf_range = st.sidebar.slider(
                "Model Confidence Range",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.05,
            )
        else:
            conf_range = (0.0, 1.0)
    else:
        conf_range = (0.0, 1.0)

    # 5. Source File Filter
    if "source_file" in df.columns and df["source_file"].nunique() > 1:
        sources = ["All Sources"] + sorted(list(df["source_file"].dropna().unique()))
        selected_source = st.sidebar.selectbox("Source File", options=sources)
    else:
        selected_source = "All Sources"

    # Apply Filters
    filtered = df.copy()

    # Sentiment
    if selected_sentiments:
        filtered = filtered[filtered["sentiment"].isin(selected_sentiments)]
    else:
        filtered = filtered.iloc[0:0]

    # Star rating
    if has_stars and not filtered.empty:
        filtered = filtered[
            (filtered["stars"] >= star_range[0]) & (filtered["stars"] <= star_range[1])
        ]

    # Confidence
    if has_conf and not filtered.empty:
        filtered = filtered[
            (filtered["confidence"] >= conf_range[0]) & (filtered["confidence"] <= conf_range[1])
        ]

    # Source File
    if selected_source != "All Sources" and not filtered.empty:
        filtered = filtered[filtered["source_file"] == selected_source]

    # Text Search
    if search_query and not filtered.empty:
        q_lower = search_query.lower()
        text_match = filtered["text"].astype(str).str.lower().str.contains(q_lower, regex=False)
        reasoning_match = filtered["reasoning"].astype(str).str.lower().str.contains(q_lower, regex=False)
        filtered = filtered[text_match | reasoning_match]

    return filtered, search_query

