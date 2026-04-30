from __future__ import annotations

import pandas as pd


def simple_asset_score(df: pd.DataFrame, asset_vars: list[str], output_col: str = "asset_score_simple") -> pd.DataFrame:
    out = df.copy()
    available = [c for c in asset_vars if c in out.columns]
    if not available:
        raise ValueError("No asset variables found")

    numeric = out[available].copy()
    for col in numeric.columns:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce").fillna(0)

    out[output_col] = numeric.sum(axis=1)
    return out


def compare_rank_correlation(df: pd.DataFrame, col1: str, col2: str) -> float:
    return float(df[[col1, col2]].dropna().corr(method="spearman").iloc[0, 1])