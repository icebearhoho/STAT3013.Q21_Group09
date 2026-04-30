from __future__ import annotations

from io import StringIO
from pathlib import Path
import random

import numpy as np
import pandas as pd

from config import CHUNK_SIZE, RANDOM_STATE


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def standardize_columns_inplace(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(col).strip().upper() for col in df.columns]
    return df


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_csv_chunks(path: Path, usecols: list[str] | None = None, chunksize: int = CHUNK_SIZE):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    selected_cols = usecols
    if usecols is not None:
        header = pd.read_csv(path, nrows=0)
        available = {str(col).strip().upper(): col for col in header.columns}
        selected_cols = [available[col.upper()] for col in usecols if col.upper() in available]

    for chunk in pd.read_csv(path, usecols=selected_cols, chunksize=chunksize, low_memory=False):
        yield standardize_columns_inplace(chunk)


def load_selected_columns(path: Path, usecols: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    header = pd.read_csv(path, nrows=0)
    available = {str(col).strip().upper(): col for col in header.columns}
    selected_cols = [available[col.upper()] for col in usecols if col.upper() in available]
    df = pd.read_csv(path, usecols=selected_cols, low_memory=False)
    return standardize_columns_inplace(df)


def priority_sample_csv(
    path: Path,
    usecols: list[str],
    max_rows: int,
    random_state: int = RANDOM_STATE,
    chunksize: int = CHUNK_SIZE,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    sampled_parts: list[pd.DataFrame] = []

    for chunk in read_csv_chunks(path, usecols=usecols, chunksize=chunksize):
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk["_PRIORITY"] = rng.random(len(chunk))
        sampled_parts.append(chunk)
        current = pd.concat(sampled_parts, ignore_index=True)
        if len(current) > max_rows:
            current = current.nlargest(max_rows, "_PRIORITY").reset_index(drop=True)
        sampled_parts = [current]

    if not sampled_parts:
        return pd.DataFrame(columns=usecols)

    final = sampled_parts[0].drop(columns="_PRIORITY", errors="ignore")
    return final.reset_index(drop=True)


def ensure_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = safe_to_numeric(out[column])
    return out


def parse_loose_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    lines = []
    for line in raw_text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.startswith("```") or cleaned.startswith("---") or cleaned.startswith("# "):
            continue
        if "," not in cleaned:
            continue
        lines.append(cleaned)

    if not lines:
        return pd.DataFrame()

    return pd.read_csv(StringIO("\n".join(lines)))


def weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    if weights is None:
        return float(values.mean())
    aligned = pd.DataFrame({"value": values, "weight": weights}).dropna()
    if aligned.empty or aligned["weight"].sum() == 0:
        return float("nan")
    return float(np.average(aligned["value"], weights=aligned["weight"]))


def normalize_binary_indicator(series: pd.Series, positive_values: set[str] | set[int] | None = None) -> pd.Series:
    positive_values = positive_values or {1, "1", 2, "2"}
    return series.astype(str).isin({str(value) for value in positive_values}).astype(float)
