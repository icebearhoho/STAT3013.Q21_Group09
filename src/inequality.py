from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_gini(values: pd.Series, weights: pd.Series | None = None) -> float:
    if weights is None:
        aligned = pd.DataFrame({"x": values}).dropna()
        x = aligned["x"].to_numpy(dtype=float)
        w = np.ones_like(x, dtype=float)
    else:
        aligned = pd.DataFrame({"x": values, "w": weights}).dropna()
        x = aligned["x"].to_numpy(dtype=float)
        w = aligned["w"].to_numpy(dtype=float)

    if len(x) == 0:
        return float("nan")

    if np.min(x) < 0:
        x = x - np.min(x)

    order = np.argsort(x)
    x = x[order]
    w = w[order]

    cumw = np.cumsum(w)
    cumxw = np.cumsum(x * w)

    if cumw[-1] == 0 or cumxw[-1] == 0:
        return 0.0
    # Gini coefficient formula adapted for weighted data
    g = 1 - 2 * np.sum((cumxw / cumxw[-1]) * (w / cumw[-1])) + np.sum(((x * w) / cumxw[-1]) * (w / cumw[-1]))
    return float(g)


def yearly_gini(df: pd.DataFrame, value_col: str, weight_col: str | None = None) -> pd.DataFrame:
    rows = []
    for year, grp in df.groupby("YEAR"):
        weights = grp[weight_col] if weight_col and weight_col in grp.columns else None
        rows.append({"YEAR": year, "GINI": weighted_gini(grp[value_col], weights)})
    return pd.DataFrame(rows).sort_values("YEAR")