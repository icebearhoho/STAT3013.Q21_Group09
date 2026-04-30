from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    HOUSEHOLD_ASSET_VARS_BY_YEAR,
    HOUSEHOLD_READY_FILE,
    HOUSING_QUALITY_VARS_BY_YEAR,
    OUTPUT_REPORTS_DIR,
    OUTPUT_TABLES_DIR,
)
from indices import build_pca_index
from robustness import compare_rank_correlation, simple_asset_score


def _period_label(year: int) -> str:
    if year <= 1999:
        return "early"
    if year <= 2009:
        return "middle"
    return "late"


def run_robustness_checks(
    household_df: pd.DataFrame,
    output_dir: Path = OUTPUT_TABLES_DIR,
    report_dir: Path = OUTPUT_REPORTS_DIR,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    results = []
    alt_rows = []
    subgroup_rows = []
    working = household_df.copy()

    for year, year_df in working.groupby("YEAR"):
        asset_vars = [col for col in HOUSEHOLD_ASSET_VARS_BY_YEAR.get(int(year), []) if col in year_df.columns]
        if not asset_vars:
            continue

        year_df = simple_asset_score(year_df, asset_vars, output_col="simple_asset_score")
        if "wealth_index" in year_df.columns:
            corr = compare_rank_correlation(year_df, "wealth_index", "simple_asset_score")
            results.append({"YEAR": int(year), "comparison": "pca_vs_simple_asset_score", "spearman_rho": corr})

        alt_specs = {
            "asset_only": asset_vars,
            "asset_plus_rooms": sorted(set(asset_vars + [col for col in ["ROOMS", "LIVEAREA"] if col in year_df.columns])),
            "housing_quality_alt": [col for col in HOUSING_QUALITY_VARS_BY_YEAR.get(int(year), []) if col in year_df.columns],
        }
        for spec_name, variables in alt_specs.items():
            valid_vars = [col for col in variables if col in year_df.columns and year_df[col].nunique(dropna=True) > 1]
            if len(valid_vars) < 3:
                continue
            alt_df, _ = build_pca_index(year_df, valid_vars, output_col=f"{spec_name}_index")
            corr = compare_rank_correlation(alt_df, "wealth_index", f"{spec_name}_index") if "wealth_index" in alt_df.columns else np.nan
            alt_rows.append(
                {
                    "YEAR": int(year),
                    "specification": spec_name,
                    "n_variables": len(valid_vars),
                    "correlation_with_main_wealth_index": corr,
                }
            )

    if "wealth_index" in working.columns:
        working["PERIOD_GROUP"] = working["YEAR"].astype(int).map(_period_label)
        if "URBAN_H" in working.columns:
            for subgroup, subgroup_df in working.groupby(["URBAN_H", "PERIOD_GROUP"]):
                if len(subgroup_df) < 100:
                    continue
                subgroup_rows.append(
                    {
                        "URBAN_H": subgroup[0],
                        "PERIOD_GROUP": subgroup[1],
                        "mean_wealth_index": subgroup_df["wealth_index"].mean(),
                        "gini_proxy_std": subgroup_df["wealth_index"].std(),
                        "n_households": len(subgroup_df),
                    }
                )

    main_results = pd.DataFrame(results)
    alt_specs_df = pd.DataFrame(alt_rows)
    subgroup_df = pd.DataFrame(subgroup_rows)

    main_results.to_csv(output_dir / "robustness_wealth_measurement.csv", index=False)
    alt_specs_df.to_csv(output_dir / "robustness_alternative_pca_specs.csv", index=False)
    subgroup_df.to_csv(output_dir / "robustness_subgroup_sensitivity.csv", index=False)

    report = pd.DataFrame(
        [
            {
                "check_name": "wealth_measurement_robustness",
                "notes": (
                    "Main wealth PCA is compared against a simple asset sum, alternative PCA specifications, "
                    "and subgroup summaries split by urban/rural status and period."
                ),
            }
        ]
    )
    report.to_csv(report_dir / "robustness_notes.csv", index=False)

    return {
        "main_results": main_results,
        "alternative_specs": alt_specs_df,
        "subgroup_sensitivity": subgroup_df,
    }
