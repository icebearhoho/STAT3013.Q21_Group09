from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from analysis_utils import load_selected_columns, priority_sample_csv, read_csv_chunks, safe_to_numeric, weighted_mean
from config import (
    HOUSEHOLD_READY_FILE,
    OUTPUT_TABLES_DIR,
    PERSON_READY_FILE,
    PERSON_SAMPLE_MAX_ROWS,
    RANDOM_STATE,
)


def _household_wealth_lookup() -> pd.DataFrame:
    usecols = [col for col in ["YEAR", "SERIAL", "WEALTH_INDEX", "wealth_class"]]
    return load_selected_columns(HOUSEHOLD_READY_FILE, usecols=usecols)


def build_gender_gap_tables(
    person_path: Path = PERSON_READY_FILE,
    output_dir: Path = OUTPUT_TABLES_DIR,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_year = []
    rows_year_urban = []

    wealth_lookup = _household_wealth_lookup()

    usecols = ["YEAR", "SERIAL", "PERWT", "SEX_H", "URBAN_H", "EDATTAIN_H", "CLASSWK_H", "AGE"]
    for chunk in read_csv_chunks(person_path, usecols=usecols):
        chunk = chunk.merge(wealth_lookup, on=["YEAR", "SERIAL"], how="left")
        for column in ["PERWT", "EDATTAIN_H", "AGE", "WEALTH_INDEX"]:
            if column in chunk.columns:
                chunk[column] = safe_to_numeric(chunk[column])

        grouped = chunk.dropna(subset=["YEAR", "SEX_H"]).groupby(["YEAR", "SEX_H"])
        for (year, sex), grp in grouped:
            rows_year.append(
                {
                    "YEAR": int(year),
                    "SEX_H": str(sex),
                    "N_PERSONS": int(len(grp)),
                    "MEAN_EDUCATION_ATTAINMENT": weighted_mean(grp["EDATTAIN_H"], grp.get("PERWT")),
                    "MEAN_AGE": weighted_mean(grp["AGE"], grp.get("PERWT")),
                    "MEAN_HOUSEHOLD_WEALTH": weighted_mean(grp["WEALTH_INDEX"], grp.get("PERWT")),
                    "CLASSWK_OBSERVED_RATE": float(grp["CLASSWK_H"].notna().mean()),
                }
            )

        grouped_urban = chunk.dropna(subset=["YEAR", "SEX_H", "URBAN_H"]).groupby(["YEAR", "URBAN_H", "SEX_H"])
        for (year, urban, sex), grp in grouped_urban:
            rows_year_urban.append(
                {
                    "YEAR": int(year),
                    "URBAN_H": str(urban),
                    "SEX_H": str(sex),
                    "N_PERSONS": int(len(grp)),
                    "MEAN_EDUCATION_ATTAINMENT": weighted_mean(grp["EDATTAIN_H"], grp.get("PERWT")),
                    "MEAN_AGE": weighted_mean(grp["AGE"], grp.get("PERWT")),
                    "MEAN_HOUSEHOLD_WEALTH": weighted_mean(grp["WEALTH_INDEX"], grp.get("PERWT")), # Fixed case              
                }
            )

    by_year = pd.DataFrame(rows_year).sort_values(["YEAR", "SEX_H"])
    by_year_urban = pd.DataFrame(rows_year_urban).sort_values(["YEAR", "URBAN_H", "SEX_H"])
    by_year.to_csv(output_dir / "gender_gaps_by_year.csv", index=False)
    by_year_urban.to_csv(output_dir / "gender_gaps_by_year_urban.csv", index=False)
    return {"gender_gaps_by_year": by_year, "gender_gaps_by_year_urban": by_year_urban}


def run_gender_regressions(
    output_dir: Path = OUTPUT_TABLES_DIR,
    sample_rows: int = PERSON_SAMPLE_MAX_ROWS,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wealth_lookup = _household_wealth_lookup()
    usecols = ["YEAR", "SERIAL", "PERWT", "SEX_H", "URBAN_H", "EDATTAIN_H", "CLASSWK_H", "AGE", "GEO1_VN"]
    sample = priority_sample_csv(
        PERSON_READY_FILE,
        usecols=usecols,
        max_rows=sample_rows,
        random_state=RANDOM_STATE,
    )
    sample = sample.merge(wealth_lookup, on=["YEAR", "SERIAL"], how="left")
    for column in ["EDATTAIN_H", "AGE", "WEALTH_INDEX"]:
        if column in sample.columns:
            sample[column] = safe_to_numeric(sample[column])
    sample["AGE2"] = sample["AGE"] ** 2
    sample["HAS_CLASSWK"] = sample["CLASSWK_H"].notna().astype(float)

    results = {}
    education_df = sample.dropna(subset=["EDATTAIN_H", "SEX_H", "YEAR"]).copy()
    if not education_df.empty:
        formula = "EDATTAIN_H ~ C(SEX_H) * C(YEAR) + AGE + AGE2"
        if "URBAN_H" in education_df.columns:
            formula += " + C(URBAN_H)"
        if "GEO1_VN" in education_df.columns:
            formula += " + C(GEO1_VN)"
        edu_model = smf.ols(formula, data=education_df).fit(cov_type="HC1")
        pd.DataFrame(
            {
                "term": edu_model.params.index,
                "coefficient": edu_model.params.values,
                "std_error": edu_model.bse.values,
                "p_value": edu_model.pvalues.values,
            }
        ).to_csv(output_dir / "gender_education_regression.csv", index=False)
        with open(output_dir / "gender_education_regression.txt", "w", encoding="utf-8") as handle:
            handle.write(str(edu_model.summary()))
        results["education_gap_model"] = edu_model

    wealth_df = sample.dropna(subset=["WEALTH_INDEX", "SEX_H", "YEAR"]).copy()
    if not wealth_df.empty:
        formula = "WEALTH_INDEX ~ C(SEX_H) * C(URBAN_H) + C(YEAR)"
        if "EDATTAIN_H" in wealth_df.columns:
            formula += " + C(EDATTAIN_H)"
        if "AGE" in wealth_df.columns:
            formula += " + AGE + AGE2"
        if "GEO1_VN" in wealth_df.columns:
            formula += " + C(GEO1_VN)"
        wealth_model = smf.ols(formula, data=wealth_df).fit(cov_type="HC1")
        pd.DataFrame(
            {
                "term": wealth_model.params.index,
                "coefficient": wealth_model.params.values,
                "std_error": wealth_model.bse.values,
                "p_value": wealth_model.pvalues.values,
            }
        ).to_csv(output_dir / "gender_wealth_regression.csv", index=False)
        with open(output_dir / "gender_wealth_regression.txt", "w", encoding="utf-8") as handle:
            handle.write(str(wealth_model.summary()))
        results["wealth_gap_model"] = wealth_model

    return results


def run_gender_analysis() -> dict[str, object]:
    outputs = {}
    outputs.update(build_gender_gap_tables())
    outputs.update(run_gender_regressions())
    return outputs
