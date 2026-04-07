"""
chart_generator.py – Intelligent Chart Recommendation & Generation
===================================================================
Detects column types, scores chart relevance, and renders Plotly/Seaborn visuals.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import streamlit as st

# ── Plotly theme ─────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_white"
COLOR_PALETTE = [
    "#6c5ce7", "#00cec9", "#fd79a8", "#0984e3",
    "#00b894", "#fdcb6e", "#e17055", "#a29bfe",
    "#55efc4", "#fab1a0", "#74b9ff", "#ffeaa7",
]


# ── Chart recommendation engine ─────────────────────────────────
def recommend_charts(df: pd.DataFrame, profile: dict) -> list[dict]:
    """
    Analyse column types and return a ranked list of chart recommendations.
    Each item: {type, title, description, score, config}
    """
    recs = []
    num = profile["numeric_cols"]
    cat = profile["categorical_cols"]
    dt = profile["datetime_cols"]

    # ── Histogram (numeric distributions) ────────────────────────
    if len(num) >= 1:
        col = _best_numeric(df, num)
        recs.append({
            "type": "histogram",
            "title": f"Distribution of {col}",
            "description": f"Shows the frequency distribution of '{col}'.",
            "score": 92,
            "config": {"col": col},
        })

    # ── Scatter (correlation between top 2 numerics) ─────────────
    if len(num) >= 2:
        pair = _best_corr_pair(df, num)
        hue = cat[0] if cat else None
        recs.append({
            "type": "scatter",
            "title": f"{pair[0]} vs {pair[1]}",
            "description": f"Explores correlation between '{pair[0]}' and '{pair[1]}'.",
            "score": 88,
            "config": {"x": pair[0], "y": pair[1], "color": hue},
        })

    # ── Bar chart (category vs numeric) ──────────────────────────
    if cat and num:
        c, n = cat[0], num[0]
        recs.append({
            "type": "bar",
            "title": f"Average {n} by {c}",
            "description": f"Compares mean '{n}' across '{c}' categories.",
            "score": 85,
            "config": {"x": c, "y": n, "agg": "mean"},
        })

    # ── Line chart (time series) ─────────────────────────────────
    if dt and num:
        t, n = dt[0], num[0]
        recs.append({
            "type": "line",
            "title": f"{n} over Time",
            "description": f"Tracks '{n}' trend along '{t}'.",
            "score": 90,
            "config": {"x": t, "y": n},
        })

    # ── Box plot (outlier detection) ─────────────────────────────
    if num:
        col = _best_numeric(df, num)
        group = cat[0] if cat else None
        recs.append({
            "type": "box",
            "title": f"Outlier Analysis – {col}",
            "description": f"Identifies spread & outliers in '{col}'.",
            "score": 78,
            "config": {"col": col, "group": group},
        })

    # ── Pie chart (category proportions) ─────────────────────────
    if cat:
        c = cat[0]
        n_unique = df[c].nunique()
        if n_unique <= 10:
            recs.append({
                "type": "pie",
                "title": f"Proportion by {c}",
                "description": f"Shows share of each '{c}' category.",
                "score": 75,
                "config": {"col": c},
            })

    # ── Correlation heatmap ──────────────────────────────────────
    if len(num) >= 3:
        recs.append({
            "type": "heatmap",
            "title": "Correlation Heatmap",
            "description": "Shows pairwise correlations among numeric columns.",
            "score": 82,
            "config": {"cols": num},
        })

    # Sort by relevance score descending
    recs.sort(key=lambda r: r["score"], reverse=True)
    return recs


# ── Render individual charts ────────────────────────────────────

def render_chart(df: pd.DataFrame, rec: dict) -> go.Figure | None:
    """Render a Plotly Figure for the given recommendation dict."""
    ctype = rec["type"]
    cfg = rec["config"]

    try:
        if ctype == "histogram":
            fig = px.histogram(
                df, x=cfg["col"], nbins=40,
                color_discrete_sequence=[COLOR_PALETTE[0]],
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(bargap=0.05)

        elif ctype == "scatter":
            fig = px.scatter(
                df, x=cfg["x"], y=cfg["y"], color=cfg.get("color"),
                color_discrete_sequence=COLOR_PALETTE,
                template=PLOTLY_TEMPLATE, opacity=0.7,
            )
            # Add trendline
            try:
                fig_trend = px.scatter(
                    df, x=cfg["x"], y=cfg["y"], trendline="ols",
                    color_discrete_sequence=["rgba(108,92,231,0.3)"],
                    template=PLOTLY_TEMPLATE,
                )
                if fig_trend.data and len(fig_trend.data) > 1:
                    fig.add_trace(fig_trend.data[1])
            except Exception:
                pass

        elif ctype == "bar":
            agg_df = df.groupby(cfg["x"])[cfg["y"]].mean().reset_index()
            agg_df = agg_df.sort_values(cfg["y"], ascending=False)
            fig = px.bar(
                agg_df, x=cfg["x"], y=cfg["y"],
                color=cfg["x"],
                color_discrete_sequence=COLOR_PALETTE,
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(showlegend=False)

        elif ctype == "line":
            temp = df.copy()
            temp[cfg["x"]] = pd.to_datetime(temp[cfg["x"]], errors="coerce")
            temp = temp.dropna(subset=[cfg["x"]]).sort_values(cfg["x"])
            fig = px.line(
                temp, x=cfg["x"], y=cfg["y"],
                color_discrete_sequence=[COLOR_PALETTE[0]],
                template=PLOTLY_TEMPLATE,
            )
            fig.update_traces(line_width=2.5)

        elif ctype == "box":
            fig = px.box(
                df, y=cfg["col"], x=cfg.get("group"),
                color=cfg.get("group"),
                color_discrete_sequence=COLOR_PALETTE,
                template=PLOTLY_TEMPLATE,
            )

        elif ctype == "pie":
            counts = df[cfg["col"]].value_counts().reset_index()
            counts.columns = [cfg["col"], "Count"]
            fig = px.pie(
                counts, names=cfg["col"], values="Count",
                color_discrete_sequence=COLOR_PALETTE,
                template=PLOTLY_TEMPLATE, hole=0.4,
            )

        elif ctype == "heatmap":
            corr = df[cfg["cols"]].corr()
            fig = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale="RdBu_r",
                template=PLOTLY_TEMPLATE,
                aspect="auto",
            )

        else:
            return None

        # Common layout tweaks
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="Inter, sans-serif", size=12),
            title=dict(
                text=rec["title"],
                font=dict(size=15, color="#1a1a2e"),
                x=0, xanchor="left",
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hoverlabel=dict(
                bgcolor="#1a1a2e",
                font_color="white",
                font_size=12,
            ),
        )
        return fig

    except Exception:
        return None


def render_custom_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str | None = None,
    y_col: str | None = None,
    color_col: str | None = None,
) -> go.Figure | None:
    """Render a user‑selected chart with the chosen columns."""
    try:
        kw = dict(
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=COLOR_PALETTE,
        )

        if chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, **kw)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, opacity=0.7, **kw)
        elif chart_type == "Bar Chart":
            agg = df.groupby(x_col)[y_col].mean().reset_index().sort_values(y_col, ascending=False)
            fig = px.bar(agg, x=x_col, y=y_col, color=x_col, **kw)
            fig.update_layout(showlegend=False)
        elif chart_type == "Line Chart":
            temp = df.copy()
            temp[x_col] = pd.to_datetime(temp[x_col], errors="coerce")
            temp = temp.dropna(subset=[x_col]).sort_values(x_col)
            fig = px.line(temp, x=x_col, y=y_col, **kw)
        elif chart_type == "Box Plot":
            fig = px.box(df, y=y_col, x=x_col, color=x_col, **kw)
        elif chart_type == "Pie Chart":
            counts = df[x_col].value_counts().reset_index()
            counts.columns = [x_col, "Count"]
            fig = px.pie(counts, names=x_col, values="Count", hole=0.4, **kw)
        elif chart_type == "Correlation Heatmap":
            num = df.select_dtypes("number").columns.tolist()
            corr = df[num].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            template=PLOTLY_TEMPLATE, aspect="auto")
        else:
            return None

        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="Inter, sans-serif", size=12),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    except Exception:
        return None


# ── Seaborn / Matplotlib fallback charts ─────────────────────────
def render_seaborn_pairplot(df: pd.DataFrame, num_cols: list[str]) -> BytesIO:
    """Return a Seaborn pairplot as PNG bytes."""
    cols = num_cols[:4]  # Limit to 4 for readability
    fig = sns.pairplot(df[cols].dropna(), plot_kws={"alpha": 0.5, "s": 20},
                       diag_kind="kde",
                       palette=[COLOR_PALETTE[0], COLOR_PALETTE[1]])
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    buf.seek(0)
    return buf


def chart_to_png_bytes(fig: go.Figure) -> bytes | None:
    """Export a Plotly figure to PNG bytes for download / PDF."""
    try:
        return fig.to_image(format="png", width=1000, height=600, scale=2)
    except Exception:
        # Fallback if kaleido is not available
        return None


# ── Internal helpers ─────────────────────────────────────────────
def _best_numeric(df, cols):
    """Pick the numeric column with highest variance."""
    variances = {c: df[c].var() for c in cols if df[c].var() is not None}
    return max(variances, key=variances.get) if variances else cols[0]


def _best_corr_pair(df, cols):
    """Return the pair of numeric columns with strongest absolute correlation."""
    if len(cols) < 2:
        return cols[0], cols[0]
    corr = df[cols].corr().abs()
    # Zero out diagonal
    np.fill_diagonal(corr.values, 0)
    idx = np.unravel_index(corr.values.argmax(), corr.shape)
    return corr.columns[idx[0]], corr.columns[idx[1]]
