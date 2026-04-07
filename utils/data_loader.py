"""
data_loader.py – Smart Data Ingestion & Cleaning
=================================================
Handles CSV / Excel uploads, profiling, and auto-cleaning.
"""

import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO


# ── Load dataset from uploaded file ──────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    """Load CSV or Excel file into a DataFrame."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            return None, "Unsupported file format. Please upload CSV or Excel."

        if df.empty:
            return None, "The uploaded file is empty. Please upload a file with data."

        return df, None
    except Exception as e:
        return None, f"Error reading file: {str(e)}"


# ── Generate dataset profile ────────────────────────────────────
def profile_data(df: pd.DataFrame) -> dict:
    """Return a dictionary summarising the dataset."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    # Try to detect hidden datetime columns stored as strings
    for col in categorical_cols[:]:
        try:
            pd.to_datetime(df[col], infer_datetime_format=True)
            datetime_cols.append(col)
            categorical_cols.remove(col)
        except (ValueError, TypeError):
            pass

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_info = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    }).loc[missing > 0].sort_values("Missing Count", ascending=False)

    profile = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_info": missing_info,
        "total_missing": int(missing.sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "duplicates": int(df.duplicated().sum()),
    }
    return profile


# ── Auto-clean dataset ──────────────────────────────────────────
def auto_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Intelligently clean the DataFrame:
    - Fill numeric NaNs with median
    - Fill categorical NaNs with mode
    - Drop columns with >60% missing
    - Remove duplicate rows
    - Attempt datetime parsing
    Returns (cleaned_df, log_messages).
    """
    log = []
    df = df.copy()

    # 1. Drop columns with excessive missing data
    thresh = 0.6
    drop_cols = [c for c in df.columns if df[c].isnull().mean() > thresh]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        log.append(f"[DROP] Removed {len(drop_cols)} column(s) with >{int(thresh*100)}% missing: {', '.join(drop_cols)}")

    # 2. Fill numeric columns
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            log.append(f"[FILL] Filled {n_miss} missing values in '{col}' with median ({median_val:.2f})")

    # 3. Fill categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col].fillna(mode_val, inplace=True)
            log.append(f"[FILL] Filled {n_miss} missing values in '{col}' with mode ('{mode_val}')")

    # 4. Remove duplicates
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        df.drop_duplicates(inplace=True)
        log.append(f"[DEDUP] Removed {n_dup} duplicate row(s)")

    # 5. Auto-parse datetime columns
    for col in df.select_dtypes(include="object").columns:
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True)
            df[col] = parsed
            log.append(f"[PARSE] Converted '{col}' to datetime")
        except (ValueError, TypeError):
            pass

    if not log:
        log.append("[OK] Dataset is already clean - no changes needed.")

    return df, log


# ── Generate sample datasets ────────────────────────────────────
def get_sample_dataset(name: str = "sales") -> pd.DataFrame:
    """Return a built-in sample DataFrame for demo purposes."""
    np.random.seed(42)

    if name == "sales":
        n = 500
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        categories = np.random.choice(["Electronics", "Clothing", "Food", "Books", "Sports"], n)
        regions = np.random.choice(["North", "South", "East", "West"], n)
        revenue = np.random.exponential(scale=500, size=n).round(2)
        quantity = np.random.poisson(lam=20, size=n)
        profit = (revenue * np.random.uniform(0.05, 0.40, n)).round(2)
        satisfaction = np.clip(np.random.normal(4.0, 0.8, n), 1, 5).round(1)
        df = pd.DataFrame({
            "Date": dates,
            "Category": categories,
            "Region": regions,
            "Revenue": revenue,
            "Quantity": quantity,
            "Profit": profit,
            "Customer_Satisfaction": satisfaction,
        })
        # Inject a few missing values for demo
        for col in ["Revenue", "Profit", "Customer_Satisfaction"]:
            idx = np.random.choice(n, size=10, replace=False)
            df.loc[idx, col] = np.nan
        return df

    elif name == "hr":
        n = 400
        departments = np.random.choice(["Engineering", "Sales", "Marketing", "HR", "Finance"], n)
        experience = np.random.randint(0, 30, n)
        salary = (30000 + experience * 3500 + np.random.normal(0, 8000, n)).round(0)
        performance = np.clip(np.random.normal(3.5, 0.9, n), 1, 5).round(1)
        attrition = np.random.choice(["Yes", "No"], n, p=[0.18, 0.82])
        df = pd.DataFrame({
            "Department": departments,
            "Experience_Years": experience,
            "Salary": salary,
            "Performance_Score": performance,
            "Attrition": attrition,
            "Age": (22 + experience + np.random.randint(-2, 5, n)),
        })
        return df

    # Default fallback
    return get_sample_dataset("sales")


# ── Export helpers ───────────────────────────────────────────────
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to Excel bytes for download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return output.getvalue()
