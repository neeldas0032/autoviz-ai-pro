"""
story_generator.py – Data Story Builder
========================================
Generates a step-by-step "presentation" from the dataset,
rendering slides inside Streamlit with premium styling.
"""

import pandas as pd
import numpy as np
import streamlit as st
from utils.insight_engine import generate_insights, generate_ai_summary, detect_anomalies
from utils.chart_generator import recommend_charts, render_chart


def generate_story(df: pd.DataFrame, profile: dict):
    """Render a full data story as styled slides inside Streamlit."""

    insights = generate_insights(df, profile)
    summary = generate_ai_summary(df, profile)
    recs = recommend_charts(df, profile)
    anomalies = detect_anomalies(df, profile["numeric_cols"])

    num = profile["numeric_cols"]
    cat = profile["categorical_cols"]
    dt = profile["datetime_cols"]

    # ── Slide 1: Dataset Overview ────────────────────────────────
    st.markdown(f"""
    <div class="story-slide">
        <div class="slide-number">1</div>
        <div class="slide-title">Dataset Overview</div>
        <div class="slide-content">
            {summary}
            <br><br>
            <table style="width:100%; border-collapse: collapse; font-size: 0.9rem;">
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 0.5rem 1rem; font-weight: 600;">Rows x Columns</td>
                    <td style="padding: 0.5rem 1rem;">{profile['rows']:,} x {profile['columns']}</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 0.5rem 1rem; font-weight: 600;">Numeric Features</td>
                    <td style="padding: 0.5rem 1rem;">{len(num)}</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 0.5rem 1rem; font-weight: 600;">Categorical Features</td>
                    <td style="padding: 0.5rem 1rem;">{len(cat)}</td>
                </tr>
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 0.5rem 1rem; font-weight: 600;">Datetime Features</td>
                    <td style="padding: 0.5rem 1rem;">{len(dt)}</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 1rem; font-weight: 600;">Missing Values</td>
                    <td style="padding: 0.5rem 1rem;">{profile['total_missing']:,}</td>
                </tr>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Slide 2: Key Trends ──────────────────────────────────────
    st.markdown("""
    <div class="story-slide">
        <div class="slide-number">2</div>
        <div class="slide-title">Key Trends & Patterns</div>
    </div>
    """, unsafe_allow_html=True)

    # Render the top 2 recommended charts
    for rec in recs[:2]:
        fig = render_chart(df, rec)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    trend_insights = [i for i in insights if i["category"] in ("trend", "distribution")]
    if trend_insights:
        for t in trend_insights[:4]:
            _render_insight_card(t)
    else:
        st.info("No significant trends detected in this dataset.")

    # ── Slide 3: Relationships & Correlations ────────────────────
    st.markdown("""
    <div class="story-slide">
        <div class="slide-number">3</div>
        <div class="slide-title">Relationships & Correlations</div>
    </div>
    """, unsafe_allow_html=True)

    # Render correlation-related chart
    corr_recs = [r for r in recs if r["type"] in ("scatter", "heatmap")]
    for rec in corr_recs[:2]:
        fig = render_chart(df, rec)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    corr_insights = [i for i in insights if i["category"] == "correlation"]
    if corr_insights:
        for c in corr_insights:
            _render_insight_card(c)
    else:
        st.info("No strong correlations found among numeric variables.")

    # ── Slide 4: Insights & Conclusion ───────────────────────────
    st.markdown("""
    <div class="story-slide">
        <div class="slide-number">4</div>
        <div class="slide-title">Final Insights & Conclusion</div>
    </div>
    """, unsafe_allow_html=True)

    # Anomaly alerts
    if anomalies:
        st.markdown("#### Anomaly Alerts")
        for a in anomalies[:5]:
            st.markdown(f"""
            <div class="insight-card danger">
                <span class="insight-badge danger-badge">ANOMALY</span>
                <span class="insight-text">
                    <strong>{a['column']}</strong>: {a['count']} outlier(s) detected
                    (range: {a['min_outlier']:.2f} to {a['max_outlier']:.2f},
                    expected: {a['lower_bound']:.2f} to {a['upper_bound']:.2f})
                </span>
            </div>
            """, unsafe_allow_html=True)

    # Category insights
    cat_insights = [i for i in insights if i["category"] in ("category", "overview", "quality")]
    for ci in cat_insights:
        _render_insight_card(ci)

    # Conclusion
    st.markdown(f"""
    <div class="story-slide" style="text-align: center; background: linear-gradient(135deg, #1a1a2e, #16213e);">
        <div class="slide-number" style="background: linear-gradient(135deg, #00b894, #00cec9);">&#10003;</div>
        <div class="slide-title" style="color: #a29bfe;">Story Complete</div>
        <div class="slide-content" style="color: #a0a0d0;">
            This automated data story was generated by <strong style="color: #00cec9;">AutoViz AI Pro</strong>.
            Explore the interactive dashboard for deeper analysis, or export a PDF report
            with all charts and insights.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Helper ──────────────────────────────────────────────────────
def _render_insight_card(insight: dict):
    severity = insight.get("severity", "info")
    icon_label = insight.get("icon", "INFO")
    badge_class = f"{severity}-badge"
    st.markdown(f"""
    <div class="insight-card {severity}">
        <span class="insight-badge {badge_class}">{icon_label}</span>
        <span class="insight-text">{insight['text']}</span>
    </div>
    """, unsafe_allow_html=True)
