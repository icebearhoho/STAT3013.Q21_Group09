from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analysis_utils import read_csv_chunks
from config import HOUSEHOLD_READY_FILE, OUTPUT_REPORTS_DIR, OUTPUT_TABLES_DIR


def build_region_year_aggregation(
    household_path: Path = HOUSEHOLD_READY_FILE,
    output_dir: Path = OUTPUT_TABLES_DIR,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    usecols = ["YEAR", "GEO1_VN", "WEALTH_INDEX", "HHWT"]
    for chunk in read_csv_chunks(household_path, usecols=usecols):
        chunk = chunk.dropna(subset=["YEAR", "GEO1_VN", "WEALTH_INDEX"])
        if chunk.empty:
            continue
        chunk["HHWT"] = pd.to_numeric(chunk["HHWT"], errors="coerce").fillna(1.0)
        chunk["WEALTH_X_WEIGHT"] = chunk["WEALTH_INDEX"] * chunk["HHWT"]
        grouped = chunk.groupby(["YEAR", "GEO1_VN"], as_index=False).agg(
            wealth_x_weight=("WEALTH_X_WEIGHT", "sum"),
            household_count=("WEALTH_INDEX", "size"),
            weight_sum=("HHWT", "sum"),
        )
        rows.append(grouped)

    region_year = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not region_year.empty:
        region_year = region_year.groupby(["YEAR", "GEO1_VN"], as_index=False).agg(
            wealth_x_weight=("wealth_x_weight", "sum"),
            household_count=("household_count", "sum"),
            weight_sum=("weight_sum", "sum"),
        )
        region_year["weighted_mean_wealth"] = region_year["wealth_x_weight"] / region_year["weight_sum"].replace(0, np.nan)
        region_year = region_year.sort_values(["YEAR", "GEO1_VN"])
    region_year.to_csv(output_dir / "region_year_wealth.csv", index=False)
    return region_year


def sigma_convergence(region_year: pd.DataFrame, output_dir: Path = OUTPUT_TABLES_DIR) -> pd.DataFrame:
    if region_year.empty:
        sigma = pd.DataFrame(columns=["YEAR", "sigma_std", "sigma_cv", "region_count"])
        sigma.to_csv(output_dir / "regional_sigma_convergence.csv", index=False)
        return sigma
    sigma = (
        region_year.groupby("YEAR", as_index=False)
        .agg(
            sigma_std=("weighted_mean_wealth", "std"),
            sigma_cv=("weighted_mean_wealth", lambda x: x.std() / x.mean() if x.mean() else np.nan),
            region_count=("GEO1_VN", "nunique"),
        )
        .sort_values("YEAR")
    )
    sigma.to_csv(output_dir / "regional_sigma_convergence.csv", index=False)
    return sigma


def beta_convergence(region_year: pd.DataFrame, output_dir: Path = OUTPUT_TABLES_DIR, report_dir: Path = OUTPUT_REPORTS_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if region_year.empty:
        return None

    first_year = int(region_year["YEAR"].min())
    last_year = int(region_year["YEAR"].max())
    pivot = region_year.pivot(index="GEO1_VN", columns="YEAR", values="weighted_mean_wealth")
    common = pivot[[first_year, last_year]].dropna().copy()
    if common.empty:
        return None

    min_shift = abs(common.min().min()) + 1.0 if common.min().min() <= 0 else 0.0
    common["initial_log_wealth"] = np.log(common[first_year] + min_shift)
    common["wealth_growth"] = np.log(common[last_year] + min_shift) - np.log(common[first_year] + min_shift)
    common = common.reset_index()

    model = smf.ols("wealth_growth ~ initial_log_wealth", data=common).fit(cov_type="HC1")
    pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "p_value": model.pvalues.values,
        }
    ).to_csv(output_dir / "regional_beta_convergence.csv", index=False)
    with open(output_dir / "regional_beta_convergence.txt", "w", encoding="utf-8") as handle:
        handle.write(str(model.summary()))

    interpretation = pd.DataFrame(
        [
            {
                "period_start": first_year,
                "period_end": last_year,
                "beta_coefficient": float(model.params.get("initial_log_wealth", np.nan)),
                "interpretation": (
                    "Negative beta suggests convergence: poorer regions gained faster."
                    if model.params.get("initial_log_wealth", 0.0) < 0
                    else "Positive beta suggests divergence: richer regions gained faster."
                ),
            }
        ]
    )
    interpretation.to_csv(report_dir / "regional_convergence_interpretation.csv", index=False)
    return model


def run_regional_analysis() -> dict[str, object]:
    region_year = build_region_year_aggregation()
    sigma = sigma_convergence(region_year)
    beta = beta_convergence(region_year)
    return {"region_year": region_year, "sigma": sigma, "beta": beta}
