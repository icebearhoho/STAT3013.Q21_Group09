from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from config import OUTPUT_TABLES_DIR


def _fit_ols(formula: str, data: pd.DataFrame):
    return smf.ols(formula=formula, data=data).fit(cov_type="HC1")


def _result_to_table(result, model_name: str) -> pd.DataFrame:
    conf = result.conf_int()
    table = pd.DataFrame(
        {
            "term": result.params.index,
            "coefficient": result.params.values,
            "std_error": result.bse.values,
            "p_value": result.pvalues.values,
            "ci_lower": conf[0].values,
            "ci_upper": conf[1].values,
            "model": model_name,
            "nobs": result.nobs,
            "r_squared": result.rsquared,
            "adj_r_squared": result.rsquared_adj,
        }
    )
    return table


def run_household_models(df: pd.DataFrame, output_dir: Path = OUTPUT_TABLES_DIR) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    model_tables: list[pd.DataFrame] = []

    urban_col = "URBAN_H" if "URBAN_H" in df.columns else "URBAN"
    region_col = "GEO1_VN" if "GEO1_VN" in df.columns else None

    base_terms = ["HOUSEHOLD_SIZE"]
    for optional in ["AREA_PER_PERSON", "ROOMS_PER_PERSON", "VEHICLE_COUNT"]:
        if optional in df.columns:
            base_terms.append(optional)
    if urban_col in df.columns:
        base_terms.append(f"C({urban_col}) * C(YEAR)")
    else:
        base_terms.append("C(YEAR)")
    if region_col:
        base_terms.append(f"C({region_col})")
    if "OWNERSHIP_H" in df.columns:
        base_terms.append("C(OWNERSHIP_H)")

    for dependent in ["wealth_index", "housing_quality_index"]:
        if dependent not in df.columns:
            continue
        model_df = df.dropna(subset=[dependent, "HOUSEHOLD_SIZE"]).copy()
        formula = f"{dependent} ~ " + " + ".join(base_terms)
        result = _fit_ols(formula, model_df)
        results[dependent] = result
        model_tables.append(_result_to_table(result, dependent))
        with open(output_dir / f"{dependent}_household_fe_ols.txt", "w", encoding="utf-8") as handle:
            handle.write(str(result.summary()))

    if model_tables:
        pd.concat(model_tables, ignore_index=True).to_csv(output_dir / "household_regression_coefficients.csv", index=False)

    return results


def run_person_returns_model(df: pd.DataFrame, output_dir: Path = OUTPUT_TABLES_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {"wealth_index", "EDATTAIN_H", "SEX_H", "YEAR"}
    if not required.issubset(df.columns):
        return None

    model_df = df.dropna(subset=["wealth_index", "EDATTAIN_H", "SEX_H", "YEAR"]).copy()
    if "AGE" in model_df.columns:
        model_df["AGE2"] = model_df["AGE"] ** 2

    terms = ["C(EDATTAIN_H) * C(SEX_H)", "C(YEAR)"]
    if "AGE" in model_df.columns:
        terms.extend(["AGE", "AGE2"])
    if "URBAN_H" in model_df.columns:
        terms.append("C(URBAN_H)")
    if "GEO1_VN" in model_df.columns:
        terms.append("C(GEO1_VN)")

    formula = "wealth_index ~ " + " + ".join(terms)
    result = _fit_ols(formula, model_df)
    _result_to_table(result, "person_education_gender_returns").to_csv(
        output_dir / "person_education_gender_returns_coefficients.csv",
        index=False,
    )
    with open(output_dir / "person_education_gender_returns.txt", "w", encoding="utf-8") as handle:
        handle.write(str(result.summary()))
    return result
