from __future__ import annotations

import pandas as pd
import numpy as np


def summary_statistics(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [c for c in columns if c in df.columns]
    return df[available].describe(include="all").T


def yearly_mean(df: pd.DataFrame, value_col: str, weight_col: str | None = None) -> pd.DataFrame:
    cols = ["YEAR", value_col]
    if weight_col and weight_col in df.columns:
        cols.append(weight_col)

    sub = df[cols].dropna(subset=["YEAR", value_col]).copy()

    if weight_col and weight_col in sub.columns:
        rows = []
        for year, grp in sub.groupby("YEAR"):
            val = np.average(grp[value_col], weights=grp[weight_col])
            rows.append({"YEAR": year, value_col: val})
        return pd.DataFrame(rows)

    return sub.groupby("YEAR", as_index=False)[value_col].mean()