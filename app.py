"""
AutoViz AI Pro – Smart Data Visualization Platform
Turn Data into Decisions Instantly
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from pathlib import Path

# ── Local modules ────────────────────────────────────────────────
from utils.data_loader import (
    load_data, profile_data, auto_clean,
    get_sample_dataset, df_to_csv_bytes, df_to_excel_bytes,
)
from utils.chart_generator import (
    recommend_charts, render_chart, render_custom_chart,
    chart_to_png_bytes, COLOR_PALETTE,
)
from utils.insight_engine import (
    generate_insights, generate_ai_summary, detect_anomalies,
)
from utils.story_generator import generate_story
from utils.pdf_report import generate_pdf_report

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoViz AI Pro - Smart Data Visualization",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load custom CSS ──────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Google Fonts
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "df": None,
        "df_clean": None,
        "profile": None,
        "data_loaded": False,
        "cleaned": False,
        "active_tab": "dashboard",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo / branding
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 1rem;">
        <div style="font-family: 'Inter', sans-serif; font-size: 1.5rem; font-weight: 900;
                    background: linear-gradient(135deg, #a29bfe, #00cec9);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    letter-spacing: -0.5px;">
            AutoViz AI Pro
        </div>
        <div style="font-size: 0.7rem; color: #7070a0; margin-top: 0.35rem; letter-spacing: 1.5px;
                    text-transform: uppercase; font-weight: 600;">
            Smart Data Visualization
        </div>
    </div>
    <hr style="border-color: rgba(108,92,231,0.2); margin: 0.5rem 0 1.5rem;">
    """, unsafe_allow_html=True)

    # ── File Upload ──────────────────────────────────────────────
    st.markdown("#### Data Source")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xls", "xlsx"],
        help="Drag and drop a CSV or Excel file here",
    )

    # Sample data buttons
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("Sales Data", use_container_width=True):
            st.session_state.df = get_sample_dataset("sales")
            st.session_state.data_loaded = True
            st.session_state.cleaned = False
            st.session_state.df_clean = None
    with col_s2:
        if st.button("HR Data", use_container_width=True):
            st.session_state.df = get_sample_dataset("hr")
            st.session_state.data_loaded = True
            st.session_state.cleaned = False
            st.session_state.df_clean = None

    # Process uploaded file
    if uploaded_file is not None:
        df_loaded, err = load_data(uploaded_file)
        if err:
            st.error(err)
        elif df_loaded is not None:
            st.session_state.df = df_loaded
            st.session_state.data_loaded = True
            st.session_state.cleaned = False
            st.session_state.df_clean = None

    st.markdown("<hr style='border-color: rgba(108,92,231,0.2);'>", unsafe_allow_html=True)

    # ── Controls (only when data loaded) ─────────────────────────
    if st.session_state.data_loaded and st.session_state.df is not None:
        working_df = st.session_state.df_clean if st.session_state.cleaned else st.session_state.df
        profile = profile_data(working_df)
        st.session_state.profile = profile

        st.markdown("#### Controls")

        # Column selector
        all_cols = working_df.columns.tolist()
        selected_cols = st.multiselect(
            "Select Columns", all_cols, default=all_cols[:8],
            help="Choose columns to include in analysis",
        )

        # Filters
        if profile["categorical_cols"]:
            st.markdown("##### Category Filters")
            filter_col = st.selectbox("Filter Column", ["None"] + profile["categorical_cols"])
            if filter_col != "None":
                unique_vals = working_df[filter_col].dropna().unique().tolist()
                selected_vals = st.multiselect(
                    f"Filter {filter_col}", unique_vals, default=unique_vals,
                )
            else:
                selected_vals = None
                filter_col = None
        else:
            filter_col = None
            selected_vals = None

        if profile["numeric_cols"]:
            st.markdown("##### Range Filter")
            range_col = st.selectbox("Numeric Column", ["None"] + profile["numeric_cols"])
            if range_col != "None":
                col_min = float(working_df[range_col].min())
                col_max = float(working_df[range_col].max())
                range_vals = st.slider(
                    f"Range for {range_col}",
                    col_min, col_max, (col_min, col_max),
                )
            else:
                range_col = None
                range_vals = None
        else:
            range_col = None
            range_vals = None

        # Chart override
        st.markdown("##### Chart Override")
        chart_override = st.selectbox(
            "Chart Type",
            ["Auto (Recommended)", "Histogram", "Scatter Plot", "Bar Chart",
             "Line Chart", "Box Plot", "Pie Chart", "Correlation Heatmap"],
        )

        # Custom chart axes
        custom_x = custom_y = custom_color = None
        if chart_override != "Auto (Recommended)":
            custom_x = st.selectbox("X Axis", all_cols, key="cx")
            if chart_override in ("Scatter Plot", "Bar Chart", "Line Chart", "Box Plot"):
                num_options = profile["numeric_cols"]
                custom_y = st.selectbox("Y Axis", num_options, key="cy") if num_options else None
            if chart_override == "Scatter Plot" and profile["categorical_cols"]:
                custom_color = st.selectbox("Color By", ["None"] + profile["categorical_cols"], key="cc")
                custom_color = None if custom_color == "None" else custom_color

    st.markdown("<hr style='border-color: rgba(108,92,231,0.2);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem; font-size: 0.7rem; color: #5050a0;">
        Built with Streamlit & Plotly<br>
        AutoViz AI Pro &copy; 2025
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════

# ── Landing page (no data loaded) ────────────────────────────────
if not st.session_state.data_loaded or st.session_state.df is None:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">AutoViz AI Pro</div>
        <div class="hero-tagline">Turn Data into Decisions Instantly</div>
        <div style="margin-top: 1.5rem; position: relative; z-index: 1;">
            <span class="pill-tag purple">Smart Charts</span>
            <span class="pill-tag teal">AI Insights</span>
            <span class="pill-tag pink">Data Stories</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon upload">CSV</div>
            <div class="feature-title">Upload Dataset</div>
            <div class="feature-desc">
                Drag and drop CSV or Excel files. Instant preview, profiling, and auto-cleaning.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon ai">AI</div>
            <div class="feature-title">AI Analysis</div>
            <div class="feature-desc">
                Smart chart recommendations, anomaly detection, and human-like insights.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon story">DOC</div>
            <div class="feature-title">Data Stories</div>
            <div class="feature-desc">
                One-click story generation with step-by-step visual narratives.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.info("Upload a dataset or click a sample data button in the sidebar to get started.")
    st.stop()


# ══════════════════════════════════════════════════════════════════
#  DATA LOADED - BUILD DASHBOARD
# ══════════════════════════════════════════════════════════════════
working_df = st.session_state.df_clean if st.session_state.cleaned else st.session_state.df
profile = st.session_state.profile or profile_data(working_df)

# Apply filters
filtered_df = working_df.copy()
if selected_cols:
    available = [c for c in selected_cols if c in filtered_df.columns]
    filtered_df = filtered_df[available]

if filter_col and selected_vals is not None and filter_col in filtered_df.columns:
    filtered_df = filtered_df[filtered_df[filter_col].isin(selected_vals)]

if range_col and range_vals is not None and range_col in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df[range_col] >= range_vals[0]) &
        (filtered_df[range_col] <= range_vals[1])
    ]

# Refresh profile for filtered data
filt_profile = profile_data(filtered_df)

# ── Navigation Tabs ──────────────────────────────────────────────
tab_dash, tab_data, tab_story, tab_export = st.tabs([
    "Dashboard", "Data Explorer", "Data Story", "Export"
])


# ══════════════════════════════════════════════════════════════════
#  TAB 1 - DASHBOARD
# ══════════════════════════════════════════════════════════════════
with tab_dash:
    # ── KPI Cards ────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi-card purple">
            <div class="kpi-icon">RECORDS</div>
            <div class="kpi-value">{len(filtered_df):,}</div>
            <div class="kpi-label">Total Rows</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        if filt_profile["numeric_cols"]:
            avg_col = filt_profile["numeric_cols"][0]
            avg_val = filtered_df[avg_col].mean()
            st.markdown(f"""
            <div class="kpi-card teal">
                <div class="kpi-icon">MEAN</div>
                <div class="kpi-value">{avg_val:,.1f}</div>
                <div class="kpi-label">Avg {avg_col[:15]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="kpi-card teal">
                <div class="kpi-icon">MEAN</div>
                <div class="kpi-value">&mdash;</div>
                <div class="kpi-label">Average</div>
            </div>
            """, unsafe_allow_html=True)

    with k3:
        if filt_profile["categorical_cols"]:
            top_cat_col = filt_profile["categorical_cols"][0]
            top_cat_val = filtered_df[top_cat_col].mode()[0] if not filtered_df[top_cat_col].mode().empty else "N/A"
            st.markdown(f"""
            <div class="kpi-card pink">
                <div class="kpi-icon">MODE</div>
                <div class="kpi-value" style="font-size:1.3rem;">{str(top_cat_val)[:12]}</div>
                <div class="kpi-label">Top {top_cat_col[:12]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="kpi-card pink">
                <div class="kpi-icon">MODE</div>
                <div class="kpi-value">&mdash;</div>
                <div class="kpi-label">Top Category</div>
            </div>
            """, unsafe_allow_html=True)

    with k4:
        if filt_profile["numeric_cols"]:
            trend_col = filt_profile["numeric_cols"][0]
            vals = filtered_df[trend_col].dropna()
            if len(vals) > 10:
                mid = len(vals) // 2
                first_half = vals.iloc[:mid].mean()
                second_half = vals.iloc[mid:].mean()
                if first_half != 0:
                    change = ((second_half - first_half) / abs(first_half)) * 100
                    color = "#00b894" if change > 0 else "#e17055"
                else:
                    change = 0
                    color = "#6c5ce7"
            else:
                change = 0
                color = "#6c5ce7"
            st.markdown(f"""
            <div class="kpi-card blue">
                <div class="kpi-icon">TREND</div>
                <div class="kpi-value" style="color: {color};">{change:+.1f}%</div>
                <div class="kpi-label">Trend</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="kpi-card blue">
                <div class="kpi-icon">TREND</div>
                <div class="kpi-value">&mdash;</div>
                <div class="kpi-label">Trend</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── Charts Section ───────────────────────────────────────────
    if chart_override == "Auto (Recommended)":
        # AI Recommended Charts
        st.markdown("""
        <div class="section-header">
            <h2>Recommended for You</h2>
            <span class="section-badge">AI Powered</span>
        </div>
        """, unsafe_allow_html=True)

        recs = recommend_charts(filtered_df, filt_profile)

        if recs:
            # Large primary chart
            primary = recs[0]
            st.markdown(f"""
            <div class="chart-title">
                {primary['title']}
                <span class="relevance-score">Score: {primary['score']}%</span>
            </div>
            """, unsafe_allow_html=True)
            fig = render_chart(filtered_df, primary)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="primary_chart")

            # Two smaller supporting charts
            if len(recs) >= 3:
                sc1, sc2 = st.columns(2)
                with sc1:
                    r = recs[1]
                    st.markdown(f"""
                    <div class="chart-title">
                        {r['title']} <span class="relevance-score">Score: {r['score']}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    fig2 = render_chart(filtered_df, r)
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True, key="support_chart_1")
                with sc2:
                    r = recs[2]
                    st.markdown(f"""
                    <div class="chart-title">
                        {r['title']} <span class="relevance-score">Score: {r['score']}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    fig3 = render_chart(filtered_df, r)
                    if fig3:
                        st.plotly_chart(fig3, use_container_width=True, key="support_chart_2")

            # Additional charts in expander
            if len(recs) > 3:
                with st.expander(f"View {len(recs) - 3} more charts", expanded=False):
                    for i, r in enumerate(recs[3:], start=3):
                        st.markdown(f"""
                        <div class="chart-title">
                            {r['title']} <span class="relevance-score">Score: {r['score']}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                        fig_extra = render_chart(filtered_df, r)
                        if fig_extra:
                            st.plotly_chart(fig_extra, use_container_width=True, key=f"extra_chart_{i}")
        else:
            st.info("Not enough data to generate chart recommendations. Try loading a richer dataset.")

    else:
        # Custom chart
        st.markdown(f"""
        <div class="section-header">
            <h2>{chart_override}</h2>
            <span class="section-badge">Manual</span>
        </div>
        """, unsafe_allow_html=True)

        fig_custom = render_custom_chart(
            filtered_df, chart_override,
            x_col=custom_x, y_col=custom_y, color_col=custom_color,
        )
        if fig_custom:
            st.plotly_chart(fig_custom, use_container_width=True, key="custom_chart")
        else:
            st.warning("Could not render chart with the selected columns. Please adjust your selections.")

    # ── AI Insights Panel ────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <h2>AI Insights</h2>
        <span class="section-badge">Auto-Generated</span>
    </div>
    """, unsafe_allow_html=True)

    # AI Summary
    with st.spinner("Generating AI summary..."):
        summary_text = generate_ai_summary(filtered_df, filt_profile)
    st.markdown(f"""
    <div class="chart-container" style="border-left: 4px solid #6c5ce7;">
        <div class="chart-title">AI Summary</div>
        <div style="color: #4a4a6a; font-size: 0.95rem; line-height: 1.7;">{summary_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # Individual insights
    insights = generate_insights(filtered_df, filt_profile)

    ins_col1, ins_col2 = st.columns([1, 1])
    half = len(insights) // 2

    with ins_col1:
        st.markdown("##### Key Observations")
        for ins in insights[:half]:
            sev = ins.get("severity", "info")
            icon_label = ins.get("icon", "INFO")
            badge_class = f"{sev}-badge"
            st.markdown(f"""
            <div class="insight-card {sev}">
                <span class="insight-badge {badge_class}">{icon_label}</span>
                <span class="insight-text">{ins['text']}</span>
            </div>
            """, unsafe_allow_html=True)

    with ins_col2:
        st.markdown("##### Anomaly Alerts")
        anomaly_insights = [i for i in insights[half:] if i["category"] in ("anomaly", "distribution")]
        other_insights = [i for i in insights[half:] if i["category"] not in ("anomaly", "distribution")]
        for ins in anomaly_insights + other_insights:
            sev = ins.get("severity", "info")
            icon_label = ins.get("icon", "INFO")
            badge_class = f"{sev}-badge"
            st.markdown(f"""
            <div class="insight-card {sev}">
                <span class="insight-badge {badge_class}">{icon_label}</span>
                <span class="insight-text">{ins['text']}</span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 2 - DATA EXPLORER
# ══════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("""
    <div class="section-header">
        <h2>Data Explorer</h2>
    </div>
    """, unsafe_allow_html=True)

    # Auto Clean button
    col_clean1, col_clean2, col_clean3 = st.columns([1, 1, 2])
    with col_clean1:
        if st.button("Auto Clean Data", use_container_width=True, type="primary"):
            with st.spinner("Cleaning dataset..."):
                cleaned, log = auto_clean(st.session_state.df)
                st.session_state.df_clean = cleaned
                st.session_state.cleaned = True
            for msg in log:
                st.success(msg)
            st.rerun()
    with col_clean2:
        if st.session_state.cleaned:
            st.markdown("""
            <div style="display: flex; align-items: center; height: 100%; padding-top: 0.5rem;">
                <span style="color: #00b894; font-weight: 600;">Data cleaned successfully</span>
            </div>
            """, unsafe_allow_html=True)

    # Data preview tabs
    dtab1, dtab2, dtab3 = st.tabs(["Preview", "Profile", "Statistics"])

    with dtab1:
        st.markdown("##### Top 10 Rows")
        st.dataframe(
            filtered_df.head(10),
            use_container_width=True,
            height=400,
        )
        st.caption(f"Showing 10 of {len(filtered_df):,} rows  |  {len(filtered_df.columns)} columns")

    with dtab2:
        prof_col1, prof_col2 = st.columns(2)
        with prof_col1:
            st.markdown("##### Dataset Size")
            st.markdown(f"""
            <div class="chart-container">
                <table style="width:100%; font-size: 0.9rem;">
                    <tr><td style="padding: 0.4rem 0; font-weight: 600;">Rows</td><td>{filt_profile['rows']:,}</td></tr>
                    <tr><td style="padding: 0.4rem 0; font-weight: 600;">Columns</td><td>{filt_profile['columns']}</td></tr>
                    <tr><td style="padding: 0.4rem 0; font-weight: 600;">Memory</td><td>{filt_profile['memory_mb']:.2f} MB</td></tr>
                    <tr><td style="padding: 0.4rem 0; font-weight: 600;">Duplicates</td><td>{filt_profile['duplicates']}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### Column Types")
            type_df = pd.DataFrame({
                "Column": filtered_df.columns,
                "Type": filtered_df.dtypes.astype(str).values,
                "Non-Null": filtered_df.notnull().sum().values,
                "Unique": [filtered_df[c].nunique() for c in filtered_df.columns],
            })
            st.dataframe(type_df, use_container_width=True, height=300)

        with prof_col2:
            st.markdown("##### Missing Values")
            if not filt_profile["missing_info"].empty:
                st.dataframe(filt_profile["missing_info"], use_container_width=True)
            else:
                st.success("No missing values detected.")

            st.markdown("##### Column Categories")
            cat_data = {
                "Category": ["Numeric", "Categorical", "Datetime"],
                "Count": [
                    len(filt_profile["numeric_cols"]),
                    len(filt_profile["categorical_cols"]),
                    len(filt_profile["datetime_cols"]),
                ],
            }
            import plotly.express as px
            fig_types = px.pie(
                pd.DataFrame(cat_data), names="Category", values="Count",
                color_discrete_sequence=COLOR_PALETTE[:3],
                hole=0.5, template="plotly_white",
            )
            fig_types.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=250,
                font=dict(family="Inter", size=12),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_types, use_container_width=True)

    with dtab3:
        if filt_profile["numeric_cols"]:
            st.markdown("##### Descriptive Statistics")
            st.dataframe(
                filtered_df[filt_profile["numeric_cols"]].describe().T.round(2),
                use_container_width=True,
            )
        else:
            st.info("No numeric columns available for statistics.")


# ══════════════════════════════════════════════════════════════════
#  TAB 3 - DATA STORY
# ══════════════════════════════════════════════════════════════════
with tab_story:
    st.markdown("""
    <div class="section-header">
        <h2>Data Story Mode</h2>
        <span class="section-badge">Premium</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="chart-container" style="text-align: center; padding: 2rem;">
        <div class="feature-icon ai" style="margin: 0 auto 0.75rem;">GEN</div>
        <div style="font-weight: 700; font-size: 1.1rem; color: #1a1a2e; margin-bottom: 0.5rem;">
            Generate Data Story
        </div>
        <div style="color: #8888a8; font-size: 0.9rem; margin-bottom: 1rem;">
            Automatically create a step-by-step visual narrative from your dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Generate Data Story", use_container_width=True, type="primary"):
        with st.spinner("Crafting your data story..."):
            time.sleep(0.5)  # Brief pause for effect
            generate_story(filtered_df, filt_profile)


# ══════════════════════════════════════════════════════════════════
#  TAB 4 - EXPORT
# ══════════════════════════════════════════════════════════════════
with tab_export:
    st.markdown("""
    <div class="section-header">
        <h2>Export Center</h2>
    </div>
    """, unsafe_allow_html=True)

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        st.markdown("""
        <div class="chart-container" style="text-align: center; padding: 2rem;">
            <div class="feature-icon upload" style="margin: 0 auto 0.75rem;">CSV</div>
            <div style="font-weight: 700; color: #1a1a2e; margin-bottom: 0.5rem;">Download Dataset</div>
            <div style="color: #8888a8; font-size: 0.85rem; margin-bottom: 1rem;">
                Export cleaned data as CSV
            </div>
        </div>
        """, unsafe_allow_html=True)
        csv_bytes = df_to_csv_bytes(filtered_df)
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="autoviz_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp_col2:
        st.markdown("""
        <div class="chart-container" style="text-align: center; padding: 2rem;">
            <div class="feature-icon ai" style="margin: 0 auto 0.75rem;">PNG</div>
            <div style="font-weight: 700; color: #1a1a2e; margin-bottom: 0.5rem;">Download Charts</div>
            <div style="color: #8888a8; font-size: 0.85rem; margin-bottom: 1rem;">
                Export all recommended charts as PNG
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Generate Chart PNGs", use_container_width=True):
            recs = recommend_charts(filtered_df, filt_profile)
            for i, rec in enumerate(recs[:4]):
                fig = render_chart(filtered_df, rec)
                if fig:
                    png_bytes = chart_to_png_bytes(fig)
                    if png_bytes:
                        st.download_button(
                            f"{rec['title'][:30]}.png",
                            data=png_bytes,
                            file_name=f"chart_{i+1}_{rec['type']}.png",
                            mime="image/png",
                            key=f"dl_chart_{i}",
                        )

    with exp_col3:
        st.markdown("""
        <div class="chart-container" style="text-align: center; padding: 2rem;">
            <div class="feature-icon story" style="margin: 0 auto 0.75rem;">PDF</div>
            <div style="font-weight: 700; color: #1a1a2e; margin-bottom: 0.5rem;">PDF Report</div>
            <div style="color: #8888a8; font-size: 0.85rem; margin-bottom: 1rem;">
                Full report with insights and charts
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Generate PDF Report", use_container_width=True):
            with st.spinner("Building PDF report..."):
                insights_list = generate_insights(filtered_df, filt_profile)

                # Try to include chart images
                chart_imgs = []
                recs = recommend_charts(filtered_df, filt_profile)
                for rec in recs[:4]:
                    fig = render_chart(filtered_df, rec)
                    if fig:
                        png = chart_to_png_bytes(fig)
                        if png:
                            chart_imgs.append(png)

                pdf_bytes = generate_pdf_report(
                    filtered_df, filt_profile, insights_list,
                    chart_images=chart_imgs if chart_imgs else None,
                )

            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name="autoviz_ai_pro_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_pdf",
            )
            st.success("PDF report generated successfully.")
