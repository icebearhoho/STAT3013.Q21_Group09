from __future__ import annotations

import pandas as pd
import numpy as np

from metadata_loader import get_missing_rules_for_var_year
from config import NUMERIC_CANDIDATES


def _replace_codes_in_series(series: pd.Series, missing_codes: list[str]) -> pd.Series:
    if not missing_codes:
        return series

    out = series.copy()
    as_str = out.astype(str).str.strip()
    mask = as_str.isin(missing_codes)
    out.loc[mask] = np.nan
    return out


def apply_missing_rules_by_year(df: pd.DataFrame, missing_rules: pd.DataFrame) -> pd.DataFrame:
    out_parts = []

    years = sorted(pd.Series(df["YEAR"]).dropna().unique())
    for year in years:
        sub = df[df["YEAR"] == year].copy()
        for col in sub.columns:
            missing_codes = get_missing_rules_for_var_year(missing_rules, col, year)
            if missing_codes:
                sub[col] = _replace_codes_in_series(sub[col], missing_codes)
        out_parts.append(sub)

    no_year = df[df["YEAR"].isna()].copy()
    if not no_year.empty:
        out_parts.append(no_year)

    if not out_parts:
        return df.copy()

    return pd.concat(out_parts, axis=0, ignore_index=True)


def basic_type_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in NUMERIC_CANDIDATES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "SERIAL" in out.columns:
        out["SERIAL"] = out["SERIAL"].astype(str).str.strip()

    return out