from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analysis_utils import load_selected_columns, priority_sample_csv, safe_to_numeric
from config import HOUSEHOLD_READY_FILE, OUTPUT_REPORTS_DIR, OUTPUT_TABLES_DIR, PERSON_READY_FILE, PERSON_SAMPLE_MAX_ROWS, RANDOM_STATE


def _merge_person_household_sample(sample_rows: int = PERSON_SAMPLE_MAX_ROWS) -> pd.DataFrame:
    household = load_selected_columns(HOUSEHOLD_READY_FILE, usecols=["YEAR", "SERIAL", "WEALTH_INDEX", "WEALTH_CLASS", "URBAN_H"])
    sample = priority_sample_csv(
        PERSON_READY_FILE,
        usecols=["YEAR", "SERIAL", "PERWT", "AGE", "SEX_H", "EDATTAIN_H", "GEO1_VN", "URBAN_H"],
        max_rows=sample_rows,
        random_state=RANDOM_STATE,
    )
    sample = sample.merge(household, on=["YEAR", "SERIAL"], how="left", suffixes=("", "_HH"))
    for column in ["AGE", "EDATTAIN_H", "WEALTH_INDEX", "PERWT"]:
        if column in sample.columns:
            sample[column] = safe_to_numeric(sample[column])
    sample = sample.dropna(subset=["YEAR", "AGE", "EDATTAIN_H"])
    sample["BIRTH_YEAR_PROXY"] = sample["YEAR"] - sample["AGE"]
    sample["COHORT_DECADE"] = (np.floor(sample["BIRTH_YEAR_PROXY"] / 10) * 10).astype("Int64")
    return sample


def run_education_mobility_analysis(
    output_dir: Path = OUTPUT_TABLES_DIR,
    report_dir: Path = OUTPUT_REPORTS_DIR,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    sample = _merge_person_household_sample()
    outputs: dict[str, pd.DataFrame] = {}

    cohort_table = (
        sample.groupby(["COHORT_DECADE", "YEAR"], as_index=False)
        .agg(
            mean_education_attainment=("EDATTAIN_H", "mean"),
            mean_household_wealth=("WEALTH_INDEX", "mean"),
            n_persons=("SERIAL", "size"),
        )
        .sort_values(["COHORT_DECADE", "YEAR"])
    )
    cohort_table.to_csv(output_dir / "education_cohort_profiles.csv", index=False)
    outputs["cohort_profiles"] = cohort_table

    rich_prob = (
        sample.assign(IS_RICH=(sample["WEALTH_CLASS"] == "rich").astype(float))
        .groupby(["YEAR", "EDATTAIN_H"], as_index=False)
        .agg(
            rich_probability=("IS_RICH", "mean"),
            mean_WEALTH_INDEX=("WEALTH_INDEX", "mean"),
            n_persons=("SERIAL", "size"),
        )
        .sort_values(["YEAR", "EDATTAIN_H"])
    )
    rich_prob.to_csv(output_dir / "education_wealth_relationship_over_time.csv", index=False)
    outputs["education_wealth_relationship"] = rich_prob

    returns_df = sample.dropna(subset=["WEALTH_INDEX", "EDATTAIN_H", "YEAR"]).copy()
    if not returns_df.empty:
        returns_formula = "WEALTH_INDEX ~ C(EDATTAIN_H) * C(YEAR) + AGE"
        if "SEX_H" in returns_df.columns:
            returns_formula += " + C(SEX_H)"
        if "URBAN_H" in returns_df.columns:
            returns_formula += " + C(URBAN_H)"
        if "GEO1_VN" in returns_df.columns:
            returns_formula += " + C(GEO1_VN)"
        returns_model = smf.ols(returns_formula, data=returns_df).fit(cov_type="HC1")
        pd.DataFrame(
            {
                "term": returns_model.params.index,
                "coefficient": returns_model.params.values,
                "std_error": returns_model.bse.values,
                "p_value": returns_model.pvalues.values,
            }
        ).to_csv(output_dir / "education_returns_proxy_regression.csv", index=False)
        with open(output_dir / "education_returns_proxy_regression.txt", "w", encoding="utf-8") as handle:
            handle.write(str(returns_model.summary()))

    note = pd.DataFrame(
        [
            {
                "analysis_scope": "Repeated cross-section proxy analysis",
                "interpretation_note": (
                    "These outputs study cohort patterns and changing education-wealth associations over time. "
                    "They are not interpreted as true intergenerational or panel mobility estimates."
                ),
            }
        ]
    )
    note.to_csv(report_dir / "education_mobility_scope_note.csv", index=False)
    outputs["scope_note"] = note
    return outputs
