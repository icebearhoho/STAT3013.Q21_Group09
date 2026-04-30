from __future__ import annotations

import pandas as pd
import numpy as np

from config import MIN_COLUMNS_REQUIRED, HOUSEHOLD_ID_COLS, PERSON_ID_COLS


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().upper() for c in out.columns]
    return out


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in MIN_COLUMNS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def audit_chunk(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "YEAR" in df.columns:
        years = sorted(pd.Series(df["YEAR"]).dropna().unique())
    else:
        years = [None]

    for col in df.columns:
        for year in years:
            sub = df if year is None else df[df["YEAR"] == year]
            rows.append({
                "YEAR": year,
                "VARIABLE": col,
                "N_ROWS": int(len(sub)),
                "N_MISSING": int(sub[col].isna().sum()),
                "N_UNIQUE": int(sub[col].nunique(dropna=True)),
                "SUM_NON_MISSING": int(sub[col].notna().sum()),
            })
    return pd.DataFrame(rows)


def combine_audit_reports(audit_parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not audit_parts:
        return pd.DataFrame(columns=["YEAR", "VARIABLE", "N_ROWS", "N_MISSING", "N_UNIQUE", "SUM_NON_MISSING"])

    full = pd.concat(audit_parts, axis=0, ignore_index=True)
    grouped = (
        full.groupby(["YEAR", "VARIABLE"], dropna=False, as_index=False)
        .agg({
            "N_ROWS": "sum",
            "N_MISSING": "sum",
            "SUM_NON_MISSING": "sum",
            "N_UNIQUE": "max",
        })
    )
    grouped["MISSING_RATE"] = grouped["N_MISSING"] / grouped["N_ROWS"].replace(0, np.nan)
    return grouped


def year_coverage_from_chunk(df: pd.DataFrame) -> pd.DataFrame:
    if "YEAR" not in df.columns:
        raise ValueError("YEAR column not found")

    years = sorted(pd.Series(df["YEAR"]).dropna().unique())
    out = []
    for col in df.columns:
        row = {"VARIABLE": col}
        for year in years:
            row[f"YEAR_{year}"] = int(df.loc[df["YEAR"] == year, col].notna().any())
        out.append(row)
    return pd.DataFrame(out)


def combine_year_coverage(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()

    full = pd.concat(parts, axis=0, ignore_index=True)
    year_cols = [c for c in full.columns if c.startswith("YEAR_")]
    result = full.groupby("VARIABLE", as_index=False)[year_cols].max()
    return result


def duplicate_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if all(c in df.columns for c in HOUSEHOLD_ID_COLS):
        hh_dups = int(df.duplicated(subset=HOUSEHOLD_ID_COLS, keep=False).sum())
        rows.append({"CHECK": "HOUSEHOLD_ID_DUPLICATES", "VALUE": hh_dups})

    if all(c in df.columns for c in PERSON_ID_COLS):
        person_dups = int(df.duplicated(subset=PERSON_ID_COLS, keep=False).sum())
        rows.append({"CHECK": "PERSON_ID_DUPLICATES", "VALUE": person_dups})

    return pd.DataFrame(rows)


def logical_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks = []

    if {"BEDROOMS", "ROOMS"}.issubset(df.columns):
        b = pd.to_numeric(df["BEDROOMS"], errors="coerce")
        r = pd.to_numeric(df["ROOMS"], errors="coerce")
        checks.append({"CHECK": "BEDROOMS_GT_ROOMS", "N_FAIL": int((b > r).fillna(False).sum())})

    if "AGE" in df.columns:
        age = pd.to_numeric(df["AGE"], errors="coerce")
        checks.append({"CHECK": "AGE_OUT_OF_RANGE", "N_FAIL": int(((age < 0) | (age > 110)).fillna(False).sum())})

    if "LIVEAREA" in df.columns:
        live = pd.to_numeric(df["LIVEAREA"], errors="coerce")
        checks.append({"CHECK": "LIVEAREA_NEGATIVE", "N_FAIL": int((live < 0).fillna(False).sum())})

    if "AUTOS" in df.columns:
        autos = pd.to_numeric(df["AUTOS"], errors="coerce")
        checks.append({"CHECK": "AUTOS_NEGATIVE", "N_FAIL": int((autos < 0).fillna(False).sum())})

    if "MOTORCYCLE" in df.columns:
        moto = pd.to_numeric(df["MOTORCYCLE"], errors="coerce")
        checks.append({"CHECK": "MOTORCYCLE_NEGATIVE", "N_FAIL": int((moto < 0).fillna(False).sum())})

    return pd.DataFrame(checks)


def combine_simple_check_tables(parts: list[pd.DataFrame], group_col: str) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()

    full = pd.concat(parts, axis=0, ignore_index=True)
    value_cols = [c for c in full.columns if c != group_col]
    agg = {c: "sum" for c in value_cols}
    return full.groupby(group_col, as_index=False).agg(agg)