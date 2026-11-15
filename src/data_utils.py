"""Loading, cleaning and profiling of user-uploaded datasets."""
from __future__ import annotations

import io
import re
import warnings
from typing import Iterable

import numpy as np
import pandas as pd


def _is_texty(s: pd.Series) -> bool:
    """True for text columns under both old (object) and new (str) pandas dtypes."""
    return bool(pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and not (
        pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
    )


_ID_NAME = re.compile(r"(^|_)(id|code|key|index|idx)$", re.IGNORECASE)


MAX_MISSING_DEFAULT = 0.5  # columns emptier than this are not auto-selected for analysis


def load_dataframe(source, filename: str) -> pd.DataFrame:
    """Read csv / xls / xlsx from a path, bytes or file-like object."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(source, low_memory=False)
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(source, engine="openpyxl")
    raise ValueError("Unsupported file type; use .csv, .xls or .xlsx")


def coerce_numeric(df: pd.DataFrame, min_parse_rate: float = 0.9) -> pd.DataFrame:
    """Turn text columns that are really numbers (e.g. '0', '12.5', NULL) into floats."""
    out = df.copy()
    for col in out.columns:
        if _is_texty(out[col]):
            parsed = pd.to_numeric(out[col], errors="coerce")
            non_null = out[col].notna().sum()
            if non_null and parsed.notna().sum() / non_null >= min_parse_rate:
                out[col] = parsed
    return out


def _is_identifier(name: str, s: pd.Series) -> bool:
    if _ID_NAME.search(str(name)):
        return True
    non_null = s.dropna()
    if len(non_null) > 20 and pd.api.types.is_integer_dtype(non_null):
        return non_null.nunique() == len(non_null)  # strictly unique integers
    return False


def _is_datetime(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not _is_texty(s):
        return False
    sample = s.dropna().head(200)
    if sample.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.9


def prepare_numeric(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    exclude_zeros: bool = False,
) -> pd.DataFrame:
    """Numeric-only copy of `df` with NaN rows removed (original index preserved).

    If `columns` is None, every non-identifier, non-constant numeric column that is at
    most 50% missing is used (mostly-empty columns would otherwise wipe out every row).
    """
    df = coerce_numeric(df)
    if columns is None:
        columns = [
            c for c in df.select_dtypes(include="number").columns
            if not _is_identifier(c, df[c]) and df[c].nunique(dropna=True) > 1
            and df[c].isna().mean() <= MAX_MISSING_DEFAULT
        ]
    columns = list(columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Unknown column(s): {missing}")
    num = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if exclude_zeros:
        num = num[(num != 0).all(axis=1)]
    return num


def profile_dataset(df: pd.DataFrame) -> dict:
    """Describe each column and suggest how to analyse the dataset."""
    df = coerce_numeric(df)
    columns, suggestions, recommended = [], [], []

    for col in df.columns:
        s = df[col]
        info: dict = {
            "name": str(col),
            "dtype": str(s.dtype),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if _is_identifier(col, s) and pd.api.types.is_numeric_dtype(s):
            info["kind"] = "identifier"
        elif pd.api.types.is_numeric_dtype(s):
            info["kind"] = "numeric"
            non_null = s.dropna()
            if not non_null.empty:
                info.update(
                    min=float(non_null.min()), max=float(non_null.max()),
                    mean=round(float(non_null.mean()), 4),
                    zero_pct=round(float((non_null == 0).mean() * 100), 2),
                    skew=round(float(non_null.skew()), 3) if len(non_null) > 2 else 0.0,
                )
            if info["n_unique"] > 1 and info["missing_pct"] <= MAX_MISSING_DEFAULT * 100:
                recommended.append(str(col))
        elif _is_datetime(s):
            info["kind"] = "datetime"
        elif info["n_unique"] <= 50 or info["n_unique"] / max(len(s), 1) < 0.05:
            info["kind"] = "categorical"
        else:
            info["kind"] = "text"
        columns.append(info)

        if info["kind"] == "numeric":
            if info.get("zero_pct", 0) > 50:
                suggestions.append(
                    f"'{col}' is {info['zero_pct']:.0f}% zeros: IQR fences collapse onto zero, "
                    "so consider excluding zeros before detecting."
                )
            if abs(info.get("skew", 0)) > 2:
                suggestions.append(f"'{col}' is heavily skewed (skew={info['skew']}); expect a long tail of flagged values.")
        if info["missing_pct"] > 20:
            suggestions.append(f"'{col}' is {info['missing_pct']:.0f}% missing; rows with gaps are dropped.")

    if not recommended:
        suggestions.append("No usable numeric columns found.")
    elif len(recommended) > 1:
        suggestions.append("Several numeric columns: the ensemble can analyse them jointly (multivariate).")
    if len(df) > 100_000:
        suggestions.append("Large dataset: the AutoEncoder trains on a 100k-row sample.")

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": columns,
        "recommended_columns": recommended,
        "suggestions": suggestions,
    }
