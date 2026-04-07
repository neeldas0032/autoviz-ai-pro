"""
pdf_report.py – PDF Report Generator
======================================
Creates a downloadable PDF report with dataset summary,
charts (as embedded PNGs), and AI insights.
"""

import pandas as pd
import numpy as np
from fpdf import FPDF
from io import BytesIO
from datetime import datetime


class ReportPDF(FPDF):
    """Custom PDF with header/footer branding."""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(108, 92, 231)
        self.cell(0, 8, "AutoViz AI Pro  |  Smart Data Visualization Report", align="L")
        self.set_draw_color(108, 92, 231)
        self.line(10, 14, 200, 14)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 170)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}  |  Generated {datetime.now().strftime('%b %d, %Y %H:%M')}", align="C")


def generate_pdf_report(
    df: pd.DataFrame,
    profile: dict,
    insights: list[dict],
    chart_images: list[bytes] | None = None,
) -> bytes:
    """
    Build a PDF report and return it as bytes.

    Parameters
    ----------
    df : cleaned DataFrame
    profile : data profile dict from data_loader.profile_data
    insights : list of insight dicts from insight_engine
    chart_images : optional list of PNG bytes for charts
    """
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Page 1 - Title ───────────────────────────────────────────
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(26, 26, 46)
    pdf.cell(0, 14, "AutoViz AI Pro", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(100, 100, 140)
    pdf.cell(0, 10, "Smart Data Visualization Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_draw_color(108, 92, 231)
    pdf.set_line_width(0.8)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 100)
    pdf.cell(0, 8, f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Dataset: {profile['rows']:,} rows x {profile['columns']} columns", align="C", new_x="LMARGIN", new_y="NEXT")

    # ── Page 2 - Dataset Summary ─────────────────────────────────
    pdf.add_page()
    _section_title(pdf, "1. Dataset Summary")

    summary_data = [
        ("Total Rows", f"{profile['rows']:,}"),
        ("Total Columns", str(profile['columns'])),
        ("Numeric Columns", str(len(profile['numeric_cols']))),
        ("Categorical Columns", str(len(profile['categorical_cols']))),
        ("Datetime Columns", str(len(profile['datetime_cols']))),
        ("Missing Values", f"{profile['total_missing']:,}"),
        ("Duplicate Rows", f"{profile['duplicates']:,}"),
        ("Memory Usage", f"{profile['memory_mb']:.2f} MB"),
    ]

    pdf.set_font("Helvetica", "", 10)
    for label, value in summary_data:
        pdf.set_text_color(80, 80, 100)
        pdf.cell(80, 8, label, border="B")
        pdf.set_text_color(26, 26, 46)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 8, value, border="B", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

    # Column info
    pdf.ln(6)
    _section_title(pdf, "2. Column Details")
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(240, 240, 250)
    pdf.cell(70, 7, "Column Name", border=1, fill=True)
    pdf.cell(50, 7, "Data Type", border=1, fill=True)
    pdf.cell(35, 7, "Missing", border=1, fill=True)
    pdf.cell(35, 7, "Unique", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    for col in df.columns[:30]:  # limit
        pdf.set_text_color(40, 40, 60)
        missing = int(df[col].isnull().sum())
        unique = int(df[col].nunique())
        pdf.cell(70, 6, str(col)[:35], border=1)
        pdf.cell(50, 6, str(df[col].dtype), border=1)
        pdf.cell(35, 6, str(missing), border=1)
        pdf.cell(35, 6, str(unique), border=1, new_x="LMARGIN", new_y="NEXT")

    # ── Page 3 - Descriptive Statistics ──────────────────────────
    if profile['numeric_cols']:
        pdf.add_page()
        _section_title(pdf, "3. Descriptive Statistics")
        desc = df[profile['numeric_cols']].describe().T
        cols_to_show = ["mean", "std", "min", "25%", "50%", "75%", "max"]

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(240, 240, 250)
        pdf.cell(30, 7, "Column", border=1, fill=True)
        for c in cols_to_show:
            pdf.cell(23, 7, c, border=1, fill=True, align="C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for idx, row in desc.iterrows():
            pdf.cell(30, 6, str(idx)[:18], border=1)
            for c in cols_to_show:
                val = row[c]
                pdf.cell(23, 6, f"{val:,.2f}" if abs(val) < 1e8 else f"{val:.2e}", border=1, align="R")
            pdf.ln()

    # ── Charts ───────────────────────────────────────────────────
    if chart_images:
        pdf.add_page()
        _section_title(pdf, "4. Visualizations")
        for i, img_bytes in enumerate(chart_images):
            if img_bytes is None:
                continue
            try:
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(img_bytes)
                    tmp_path = tmp.name
                if pdf.get_y() > 180:
                    pdf.add_page()
                pdf.image(tmp_path, x=15, w=180)
                pdf.ln(5)
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── Insights ─────────────────────────────────────────────────
    pdf.add_page()
    _section_title(pdf, "5. AI-Generated Insights")

    for ins in insights:
        if pdf.get_y() > 265:
            pdf.add_page()
        icon = ins.get("icon", "INFO")
        text = ins.get("text", "").replace("**", "").replace("<strong>", "").replace("</strong>", "")
        severity = ins.get("severity", "info")

        if severity == "danger":
            pdf.set_text_color(225, 112, 85)
        elif severity == "warning":
            pdf.set_text_color(200, 160, 50)
        elif severity == "success":
            pdf.set_text_color(0, 148, 120)
        else:
            pdf.set_text_color(80, 80, 100)

        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(22, 6, f"[{icon}]")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # ── Output ───────────────────────────────────────────────────
    return bytes(pdf.output())


def _section_title(pdf: ReportPDF, title: str):
    """Render a styled section title."""
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(108, 92, 231)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(108, 92, 231)
    pdf.set_line_width(0.4)
    pdf.line(10, pdf.get_y(), 100, pdf.get_y())
    pdf.ln(4)
