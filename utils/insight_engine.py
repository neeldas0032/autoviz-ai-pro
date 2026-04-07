"""
insight_engine.py – AI-Powered Insight & Anomaly Detection
===========================================================
Generates human-like textual insights using rule-based statistics,
correlation analysis, trend detection, and outlier flagging.
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats


# ── Master insight generator ────────────────────────────────────
def generate_insights(df: pd.DataFrame, profile: dict) -> list[dict]:
    """
    Return a list of insight dicts:
        {icon, text, category, severity}
    severity: 'info' | 'success' | 'warning' | 'danger'
    """
    insights = []
    num = profile["numeric_cols"]
    cat = profile["categorical_cols"]
    dt = profile["datetime_cols"]

    # ── Dataset overview ──────────────────────────────────────────
    insights.append({
        "icon": "OVERVIEW",
        "text": (
            f"Dataset contains **{profile['rows']:,}** rows and "
            f"**{profile['columns']}** columns "
            f"({len(num)} numeric, {len(cat)} categorical"
            f"{', ' + str(len(dt)) + ' datetime' if dt else ''})."
        ),
        "category": "overview",
        "severity": "info",
    })

    if profile["total_missing"] > 0:
        pct = round(profile["total_missing"] / (profile["rows"] * profile["columns"]) * 100, 1)
        insights.append({
            "icon": "QUALITY",
            "text": f"**{profile['total_missing']:,}** missing values detected ({pct}% of all cells).",
            "category": "quality",
            "severity": "warning",
        })

    if profile["duplicates"] > 0:
        insights.append({
            "icon": "DUPLICATES",
            "text": f"**{profile['duplicates']:,}** duplicate rows found.",
            "category": "quality",
            "severity": "warning",
        })

    # ── Correlations ──────────────────────────────────────────────
    insights.extend(_correlation_insights(df, num))

    # ── Distributions & outliers ──────────────────────────────────
    insights.extend(_distribution_insights(df, num))

    # ── Category dominance ────────────────────────────────────────
    insights.extend(_category_insights(df, cat))

    # ── Trend detection ───────────────────────────────────────────
    insights.extend(_trend_insights(df, dt, num))

    return insights


# ── Correlation insights ────────────────────────────────────────
def _correlation_insights(df, num_cols):
    results = []
    if len(num_cols) < 2:
        return results
    corr = df[num_cols].corr()
    seen = set()
    for i, c1 in enumerate(num_cols):
        for j, c2 in enumerate(num_cols):
            if i >= j:
                continue
            key = tuple(sorted([c1, c2]))
            if key in seen:
                continue
            seen.add(key)
            r = corr.loc[c1, c2]
            if abs(r) >= 0.7:
                direction = "positive" if r > 0 else "negative"
                strength = "strong" if abs(r) >= 0.85 else "moderate-to-strong"
                results.append({
                    "icon": "CORRELATION",
                    "text": (
                        f"**{strength.capitalize()} {direction} correlation** "
                        f"detected between **{c1}** and **{c2}** (r = {r:.2f})."
                    ),
                    "category": "correlation",
                    "severity": "success",
                })
    return results


# ── Distribution & outlier insights ─────────────────────────────
def _distribution_insights(df, num_cols):
    results = []
    for col in num_cols[:6]:  # Limit to avoid too many insights
        series = df[col].dropna()
        if len(series) < 10:
            continue

        mean_val = series.mean()
        std_val = series.std()
        skew = series.skew()

        # Skewness insight
        if abs(skew) > 1.5:
            direction = "right (positively)" if skew > 0 else "left (negatively)"
            results.append({
                "icon": "DISTRIBUTION",
                "text": (
                    f"**{col}** is heavily skewed {direction} "
                    f"(skewness = {skew:.2f}), suggesting the presence of extreme values."
                ),
                "category": "distribution",
                "severity": "info",
            })

        # Outlier detection (IQR method)
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        if len(outliers) > 0:
            pct = round(len(outliers) / len(series) * 100, 1)
            results.append({
                "icon": "ANOMALY",
                "text": (
                    f"**{len(outliers)} potential outlier(s)** detected in **{col}** "
                    f"({pct}% of values outside the IQR range [{lower:.1f}, {upper:.1f}])."
                ),
                "category": "anomaly",
                "severity": "danger",
            })

    return results


# ── Category dominance insights ─────────────────────────────────
def _category_insights(df, cat_cols):
    results = []
    for col in cat_cols[:4]:
        counts = df[col].value_counts(normalize=True)
        if len(counts) == 0:
            continue
        top = counts.index[0]
        top_pct = round(counts.iloc[0] * 100, 1)
        if top_pct >= 35:
            results.append({
                "icon": "DOMINANT",
                "text": (
                    f"Category **'{top}'** dominates **{col}** with "
                    f"**{top_pct}%** share."
                ),
                "category": "category",
                "severity": "success",
            })
        unique = df[col].nunique()
        if unique <= 2:
            results.append({
                "icon": "BINARY",
                "text": f"**{col}** is a binary column with {unique} unique values.",
                "category": "category",
                "severity": "info",
            })
    return results


# ── Trend detection ─────────────────────────────────────────────
def _trend_insights(df, dt_cols, num_cols):
    results = []
    if not dt_cols or not num_cols:
        return results

    dt_col = dt_cols[0]
    for col in num_cols[:3]:
        try:
            temp = df[[dt_col, col]].copy()
            temp[dt_col] = pd.to_datetime(temp[dt_col], errors="coerce")
            temp = temp.dropna().sort_values(dt_col)
            if len(temp) < 10:
                continue

            # Simple linear trend via Spearman rank
            x = np.arange(len(temp))
            y = temp[col].values
            slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x, y)

            if p_val < 0.05 and abs(r_val) > 0.3:
                direction = "upward" if slope > 0 else "downward"
                strength = "significant" if abs(r_val) > 0.6 else "moderate"
                results.append({
                    "icon": "TREND",
                    "text": (
                        f"**{strength.capitalize()} {direction} trend** observed in "
                        f"**{col}** over time (R-squared = {r_val**2:.2f}, p < {max(p_val, 0.001):.3f})."
                    ),
                    "category": "trend",
                    "severity": "success",
                })
        except Exception:
            continue

    return results


# ── Summary text block ──────────────────────────────────────────
def generate_ai_summary(df: pd.DataFrame, profile: dict) -> str:
    """Generate a paragraph-style AI summary of the dataset."""
    num = profile["numeric_cols"]
    cat = profile["categorical_cols"]
    dt = profile["datetime_cols"]

    parts = []
    parts.append(
        f"This dataset comprises <strong>{profile['rows']:,} records</strong> across "
        f"<strong>{profile['columns']} features</strong>. "
    )

    if num:
        top = num[0]
        parts.append(
            f"The primary numeric variable <strong>{top}</strong> has a mean of "
            f"<strong>{df[top].mean():,.2f}</strong> and a standard deviation of "
            f"<strong>{df[top].std():,.2f}</strong>. "
        )

    if cat:
        top_cat = cat[0]
        top_val = df[top_cat].mode()[0] if not df[top_cat].mode().empty else "N/A"
        parts.append(
            f"Among categorical features, <strong>{top_cat}</strong> most frequently takes the value "
            f"<strong>'{top_val}'</strong>. "
        )

    if profile["total_missing"] > 0:
        parts.append(
            f"Data quality analysis reveals <strong>{profile['total_missing']:,} missing values</strong> "
            f"that may need attention. "
        )

    if dt:
        try:
            temp = pd.to_datetime(df[dt[0]], errors="coerce").dropna()
            date_range = f"{temp.min().strftime('%b %Y')} to {temp.max().strftime('%b %Y')}"
            parts.append(f"The data spans from <strong>{date_range}</strong>. ")
        except Exception:
            pass

    if len(num) >= 2:
        corr = df[num].corr().abs()
        np.fill_diagonal(corr.values, 0)
        max_corr = corr.max().max()
        if max_corr > 0.5:
            idx = np.unravel_index(corr.values.argmax(), corr.shape)
            parts.append(
                f"Notable correlation exists between <strong>{corr.columns[idx[0]]}</strong> and "
                f"<strong>{corr.columns[idx[1]]}</strong> (|r| = {max_corr:.2f}). "
            )

    parts.append(
        "Further exploration through the interactive dashboard is recommended "
        "to uncover deeper patterns and actionable insights."
    )

    return "".join(parts)


# ── Anomaly detection summary ───────────────────────────────────
def detect_anomalies(df: pd.DataFrame, num_cols: list[str]) -> list[dict]:
    """Return a list of anomaly dicts with column, count, and values."""
    anomalies = []
    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (series < lower) | (series > upper)
        if outlier_mask.sum() > 0:
            anomalies.append({
                "column": col,
                "count": int(outlier_mask.sum()),
                "min_outlier": float(series[outlier_mask].min()),
                "max_outlier": float(series[outlier_mask].max()),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            })
    return anomalies
