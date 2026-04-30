from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _prepare_numeric_matrix(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    subset = df[variables].copy()
    for col in subset.columns:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
    subset = subset.fillna(subset.median(numeric_only=True))
    return subset


def build_pca_index(df: pd.DataFrame, variables: list[str], output_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    available = [c for c in variables if c in out.columns]
    if len(available) < 3:
        raise ValueError(f"Need at least 3 variables for PCA. Found: {available}")

    x = _prepare_numeric_matrix(out, available)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca = PCA(n_components=1)
    scores = pca.fit_transform(x_scaled).flatten()
    out[output_col] = scores

    loadings = pd.DataFrame({
        "variable": available,
        "loading": pca.components_[0],
    }).sort_values("loading", ascending=False)

    return out, loadings


def create_wealth_class(df: pd.DataFrame, wealth_col: str = "wealth_index") -> pd.DataFrame:
    out = df.copy()
    q30 = out[wealth_col].quantile(0.30)
    q70 = out[wealth_col].quantile(0.70)

    out["wealth_class"] = np.where(
        out[wealth_col] <= q30,
        "poor",
        np.where(out[wealth_col] <= q70, "middle", "rich")
    )
    return out