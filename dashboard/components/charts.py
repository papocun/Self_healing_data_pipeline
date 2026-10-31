"""Plotly visualization charts for the Self-Healing Pipeline Streamlit Dashboard."""

from collections import Counter
from typing import Any, Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Consistent color palette
SENTIMENT_COLOR_MAP = {
    "positive": "#10B981",  # Emerald Green
    "negative": "#EF4444",  # Crimson Red
    "neutral": "#64748B",   # Slate Gray
}


def create_sentiment_donut_chart(df: pd.DataFrame) -> go.Figure:
    """Creates a donut chart displaying sentiment distribution and percentages."""
    if df.empty or "sentiment" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No sentiment data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#6B7280"),
        )
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        return fig

    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]

    fig = px.pie(
        counts,
        names="sentiment",
        values="count",
        hole=0.55,
        color="sentiment",
        color_discrete_map=SENTIMENT_COLOR_MAP,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>",
        marker=dict(line=dict(color="#FFFFFF", width=2)),
    )
    fig.update_layout(
        title=dict(text="<b>Sentiment Distribution</b>", font=dict(size=16)),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    return fig


def create_sentiment_by_stars_chart(df: pd.DataFrame) -> go.Figure:
    """Creates a stacked or grouped bar chart displaying sentiment breakdown across star ratings (1 to 5)."""
    if df.empty or "stars" not in df.columns or "sentiment" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No star rating data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#6B7280"),
        )
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        return fig

    valid_stars_df = df.dropna(subset=["stars"]).copy()
    if valid_stars_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Star ratings are unpopulated or missing",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=350)
        return fig

    # Round or format stars to standard ratings 1, 2, 3, 4, 5
    valid_stars_df["star_label"] = valid_stars_df["stars"].apply(lambda s: f"{s:.1f} ★" if s % 1 != 0 else f"{int(s)} ★")
    grouped = (
        valid_stars_df.groupby(["star_label", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    # Sort star labels properly
    fig = px.bar(
        grouped,
        x="star_label",
        y="count",
        color="sentiment",
        barmode="stack",
        color_discrete_map=SENTIMENT_COLOR_MAP,
        category_orders={"sentiment": ["positive", "neutral", "negative"]},
    )
    fig.update_layout(
        title=dict(text="<b>Sentiment Breakdown by Star Rating</b>", font=dict(size=16)),
        xaxis=dict(title="Star Rating", categoryorder="category ascending"),
        yaxis=dict(title="Number of Reviews"),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Sentiment: %{fullData.name}<br>Count: %{y:,}<extra></extra>"
    )
    return fig


def create_confidence_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Creates a histogram displaying model confidence score distribution."""
    if df.empty or "confidence" not in df.columns:
        fig = go.Figure()
        fig.update_layout(height=300)
        return fig

    fig = px.histogram(
        df,
        x="confidence",
        color="sentiment",
        nbins=20,
        barmode="overlay",
        color_discrete_map=SENTIMENT_COLOR_MAP,
        opacity=0.75,
    )
    mean_conf = df["confidence"].mean()
    fig.add_vline(
        x=mean_conf,
        line_dash="dash",
        line_color="#1E293B",
        annotation_text=f"Mean: {mean_conf:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=dict(text="<b>Model Confidence Score Distribution</b>", font=dict(size=16)),
        xaxis=dict(title="Confidence Score (0.0 to 1.0)", range=[0.0, 1.05]),
        yaxis=dict(title="Review Count"),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def create_top_aspects_bar_chart(df: pd.DataFrame, aspect_type: str = "positive", top_n: int = 8) -> go.Figure:
    """Creates a horizontal bar chart of the most frequent positive or negative aspects."""
    col_name = "positive_aspects" if aspect_type == "positive" else "negative_aspects"
    if df.empty or col_name not in df.columns:
        fig = go.Figure()
        fig.update_layout(height=280)
        return fig

    all_aspects: List[str] = []
    for aspects_list in df[col_name]:
        if isinstance(aspects_list, (list, tuple)):
            for a in aspects_list:
                cleaned = str(a).strip().lower()
                if cleaned and len(cleaned) > 2:
                    all_aspects.append(cleaned)

    if not all_aspects:
        fig = go.Figure()
        fig.add_annotation(
            text=f"No {aspect_type} aspects extracted yet",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#6B7280"),
        )
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
        return fig

    counter = Counter(all_aspects).most_common(top_n)
    aspect_df = pd.DataFrame(counter, columns=["aspect", "frequency"])
    aspect_df = aspect_df.sort_values(by="frequency", ascending=True)

    bar_color = SENTIMENT_COLOR_MAP["positive"] if aspect_type == "positive" else SENTIMENT_COLOR_MAP["negative"]

    fig = px.bar(
        aspect_df,
        x="frequency",
        y="aspect",
        orientation="h",
        color_discrete_sequence=[bar_color],
    )
    title_label = "Top Positive Aspects Mentioned" if aspect_type == "positive" else "Top Negative Aspects Mentioned"
    fig.update_layout(
        title=dict(text=f"<b>{title_label}</b>", font=dict(size=14)),
        xaxis=dict(title="Mentions"),
        yaxis=dict(title=""),
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Mentions: %{x:,}<extra></extra>"
    )
    return fig


def create_pipeline_records_breakdown_chart(checkpoint_info: Dict[str, Any], quarantine_info: Dict[str, Any]) -> go.Figure:
    """Creates a funnel or bar comparison of record health across pipeline processing stages."""
    succ = checkpoint_info.get("total_successful", 0)
    failed = checkpoint_info.get("total_failed", quarantine_info.get("inference_failed_count", 0))
    invalids = checkpoint_info.get("total_invalids", quarantine_info.get("validation_failed_count", 0))
    duplicates = checkpoint_info.get("total_duplicates", quarantine_info.get("duplicates_count", 0))

    categories = ["Successful Inferences", "Failed Inferences", "Schema Invalids", "Duplicates Quarantined"]
    values = [succ, failed, invalids, duplicates]
    colors = ["#10B981", "#EF4444", "#F59E0B", "#6366F1"]

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:,}" for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=dict(text="<b>Pipeline Records Health Breakdown</b>", font=dict(size=16)),
        yaxis=dict(title="Record Count", type="log" if max(values or [0]) > 1000 else "linear"),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>"
    )
    return fig

