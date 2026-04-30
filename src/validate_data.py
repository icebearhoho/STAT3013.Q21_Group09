from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import parse_loose_csv, priority_sample_csv, read_csv_chunks, safe_to_numeric
from config import (
    CORE_CATEGORICAL_VARS,
    CORE_CONTINUOUS_VARS,
    HARMONIZE_VARS,
    HARMONIZED_MASTER_FILE,
    OUTPUT_REPORTS_DIR,
    RANDOM_STATE,
    VALIDATION_SAMPLE_MAX_ROWS,
    YEAR_COVERAGE_FILE,
)
from metadata_loader import (
    HARMONIZATION_RULE_COLUMNS,
    MISSING_RULE_COLUMNS,
    metadata_schema_report,
)


def _safe_year_values(series: pd.Series) -> list[int]:
    years = pd.to_numeric(series, errors="coerce").dropna().astype(int).unique().tolist()
    return sorted(years)


def _write_report(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def audit_metadata_usage(
    missing_rules: pd.DataFrame,
    harm_rules: pd.DataFrame,
    variable_dict: pd.DataFrame,
    coverage: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    reports: dict[str, pd.DataFrame] = {}
    coverage = coverage if coverage is not None else pd.DataFrame()

    reports["missing_rules_schema"] = metadata_schema_report(
        missing_rules,
        MISSING_RULE_COLUMNS,
        "missing_rules.csv",
    )
    reports["harmonization_rules_schema"] = metadata_schema_report(
        harm_rules,
        HARMONIZATION_RULE_COLUMNS,
        "harmonization_rules.csv",
    )

    unresolved_rows = []
    if not coverage.empty:
        coverage = coverage.copy()
        coverage.columns = [str(col).strip().upper() for col in coverage.columns]
        variables = sorted(set(coverage["VARIABLE"].astype(str).str.upper()))

        missing_vars = set()
        if not missing_rules.empty and "variable_name" in missing_rules.columns:
            missing_vars = set(missing_rules["variable_name"].astype(str).str.upper())
        harm_vars = set()
        if not harm_rules.empty and "variable_name" in harm_rules.columns:
            harm_vars = set(harm_rules["variable_name"].astype(str).str.upper())
        dict_vars = set()
        if not variable_dict.empty and "variable_name" in variable_dict.columns:
            dict_vars = set(variable_dict["variable_name"].astype(str).str.upper())

        for variable in variables:
            unresolved_rows.append(
                {
                    "VARIABLE": variable,
                    "IN_VARIABLE_DICTIONARY": int(variable in dict_vars),
                    "HAS_MISSING_RULE": int(variable in missing_vars),
                    "HAS_HARMONIZATION_RULE": int(variable in harm_vars),
                    "IS_TARGET_HARMONIZE_VAR": int(variable in {var.upper() for var in HARMONIZE_VARS}),
                }
            )

    reports["metadata_usage_audit"] = pd.DataFrame(unresolved_rows)
    return reports


def build_missingness_summary(harmonized_path: Path = HARMONIZED_MASTER_FILE) -> pd.DataFrame:
    rows = []
    for chunk in read_csv_chunks(harmonized_path):
        if "YEAR" not in chunk.columns:
            continue
        years = _safe_year_values(chunk["YEAR"])
        for column in chunk.columns:
            for year in years:
                subset = chunk.loc[chunk["YEAR"] == year, column]
                rows.append(
                    {
                        "YEAR": year,
                        "VARIABLE": column,
                        "N_ROWS": int(len(subset)),
                        "N_MISSING": int(subset.isna().sum()),
                        "MISSING_RATE": float(subset.isna().mean()) if len(subset) else np.nan,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["YEAR", "VARIABLE", "N_ROWS", "N_MISSING", "MISSING_RATE"])

    summary = pd.DataFrame(rows)
    return (
        summary.groupby(["YEAR", "VARIABLE"], as_index=False)
        .agg({"N_ROWS": "sum", "N_MISSING": "sum"})
        .assign(MISSING_RATE=lambda df: df["N_MISSING"] / df["N_ROWS"].replace(0, np.nan))
        .sort_values(["VARIABLE", "YEAR"])
    )


def build_category_consistency_report(harmonized_path: Path = HARMONIZED_MASTER_FILE) -> pd.DataFrame:
    counters: dict[tuple[str, int], Counter] = defaultdict(Counter)

    for chunk in read_csv_chunks(harmonized_path):
        if "YEAR" not in chunk.columns:
            continue
        years = _safe_year_values(chunk["YEAR"])
        variables = [col for col in CORE_CATEGORICAL_VARS if col in chunk.columns]
        for variable in variables:
            for year in years:
                values = (
                    chunk.loc[chunk["YEAR"] == year, variable]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                if not values.empty:
                    counters[(variable, year)].update(values.tolist())

    rows = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for variable, year in counters:
        grouped[variable].append(year)

    for variable, years in grouped.items():
        ordered_years = sorted(years)
        previous_codes: set[str] | None = None
        for year in ordered_years:
            code_counts = counters[(variable, year)]
            code_set = set(code_counts.keys())
            new_codes = sorted(code_set - previous_codes) if previous_codes is not None else sorted(code_set)
            disappeared_codes = sorted(previous_codes - code_set) if previous_codes is not None else []
            rows.append(
                {
                    "VARIABLE": variable,
                    "YEAR": year,
                    "N_CODES": len(code_set),
                    "TOP_CODES": " | ".join(f"{code}:{count}" for code, count in code_counts.most_common(10)),
                    "NEW_CODES_VS_PREVIOUS_YEAR": " | ".join(new_codes[:10]),
                    "DISAPPEARED_CODES_VS_PREVIOUS_YEAR": " | ".join(disappeared_codes[:10]),
                    "FLAG_CATEGORY_SHIFT": int(bool(new_codes or disappeared_codes) and previous_codes is not None),
                }
            )
            previous_codes = code_set

    return pd.DataFrame(rows).sort_values(["VARIABLE", "YEAR"]) if rows else pd.DataFrame()


def build_harmonization_coverage_report(
    harm_rules: pd.DataFrame,
    harmonized_path: Path = HARMONIZED_MASTER_FILE,
) -> pd.DataFrame:
    rules = harm_rules.copy()
    rules.columns = [str(col).strip().lower() for col in rules.columns]
    mapping_lookup: dict[tuple[str, str], set[str]] = defaultdict(set)

    if not rules.empty and {"variable_name", "year", "original_code"}.issubset(rules.columns):
        for _, row in rules.iterrows():
            variable = str(row["variable_name"]).strip().upper()
            year = str(row["year"]).strip().upper()
            mapping_lookup[(variable, year)].add(str(row["original_code"]).strip())

    coverage_rows = []
    unresolved_examples: dict[tuple[str, int], Counter] = defaultdict(Counter)

    for chunk in read_csv_chunks(harmonized_path):
        years = _safe_year_values(chunk["YEAR"]) if "YEAR" in chunk.columns else []
        for variable in HARMONIZE_VARS:
            harm_col = f"{variable}_H"
            if variable not in chunk.columns:
                continue
            for year in years:
                mask = chunk["YEAR"] == year
                raw_values = chunk.loc[mask, variable].dropna().astype(str).str.strip()
                if raw_values.empty:
                    continue
                mapped_codes = mapping_lookup.get((variable, str(year)), set()) or mapping_lookup.get((variable, "ALL"), set())
                covered = raw_values.isin(mapped_codes).sum() if mapped_codes else 0
                unresolved = raw_values[~raw_values.isin(mapped_codes)] if mapped_codes else raw_values
                unresolved_examples[(variable, year)].update(unresolved.tolist())
                coverage_rows.append(
                    {
                        "VARIABLE": variable,
                        "YEAR": year,
                        "NON_MISSING_RAW": int(raw_values.shape[0]),
                        "COVERED_BY_RULES": int(covered),
                        "COVERAGE_RATE": float(covered / raw_values.shape[0]) if raw_values.shape[0] else np.nan,
                        "HAS_HARMONIZED_COLUMN": int(harm_col in chunk.columns),
                        "HAS_RULES": int(bool(mapped_codes)),
                    }
                )

    if not coverage_rows:
        return pd.DataFrame()

    coverage = pd.DataFrame(coverage_rows)
    coverage = (
        coverage.groupby(["VARIABLE", "YEAR"], as_index=False)
        .agg(
            {
                "NON_MISSING_RAW": "sum",
                "COVERED_BY_RULES": "sum",
                "HAS_HARMONIZED_COLUMN": "max",
                "HAS_RULES": "max",
            }
        )
    )
    coverage["COVERAGE_RATE"] = coverage["COVERED_BY_RULES"] / coverage["NON_MISSING_RAW"].replace(0, np.nan)
    coverage["TOP_UNRESOLVED_CODES"] = coverage.apply(
        lambda row: " | ".join(
            f"{code}:{count}"
            for code, count in unresolved_examples[(row["VARIABLE"], int(row["YEAR"]))].most_common(10)
        ),
        axis=1,
    )
    return coverage.sort_values(["VARIABLE", "YEAR"])


def build_distribution_break_report(
    harmonized_path: Path = HARMONIZED_MASTER_FILE,
    sample_max_rows: int = VALIDATION_SAMPLE_MAX_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_cols = ["YEAR"] + [var for var in CORE_CONTINUOUS_VARS if var not in {"wealth_index", "housing_quality_index"}]
    sample = priority_sample_csv(
        harmonized_path,
        usecols=sample_cols,
        max_rows=sample_max_rows,
        random_state=RANDOM_STATE,
    )
    sample = sample[[col for col in sample.columns if col in sample_cols]].copy()
    for column in sample.columns:
        if column != "YEAR":
            sample[column] = safe_to_numeric(sample[column])

    outlier_rows = []
    break_rows = []

    available_numeric = [col for col in sample.columns if col in CORE_CONTINUOUS_VARS and col != "YEAR"]
    for variable in available_numeric:
        grouped = []
        for year, grp in sample.groupby("YEAR"):
            series = grp[variable].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            grouped.append(
                {
                    "YEAR": int(year),
                    "VARIABLE": variable,
                    "MEAN": float(series.mean()),
                    "MEDIAN": float(series.median()),
                    "STD": float(series.std()),
                    "P01": float(series.quantile(0.01)),
                    "P99": float(series.quantile(0.99)),
                    "OUTLIER_RATE_IQR": float(((series < lower) | (series > upper)).mean()),
                }
            )
            outlier_rows.append(grouped[-1])

        grouped = sorted(grouped, key=lambda row: row["YEAR"])
        for prev, curr in zip(grouped, grouped[1:]):
            pooled_std = np.nanmean([prev["STD"], curr["STD"]])
            break_rows.append(
                {
                    "VARIABLE": variable,
                    "FROM_YEAR": prev["YEAR"],
                    "TO_YEAR": curr["YEAR"],
                    "MEAN_CHANGE": curr["MEAN"] - prev["MEAN"],
                    "MEDIAN_CHANGE": curr["MEDIAN"] - prev["MEDIAN"],
                    "STANDARDIZED_MEAN_CHANGE": (
                        (curr["MEAN"] - prev["MEAN"]) / pooled_std if pooled_std and not np.isnan(pooled_std) else np.nan
                    ),
                    "FLAG_STRUCTURAL_BREAK": int(
                        abs((curr["MEAN"] - prev["MEAN"]) / pooled_std) >= 1.0 if pooled_std and not np.isnan(pooled_std) else 0
                    ),
                }
            )

    return (
        pd.DataFrame(outlier_rows).sort_values(["VARIABLE", "YEAR"]) if outlier_rows else pd.DataFrame(),
        pd.DataFrame(break_rows).sort_values(["VARIABLE", "FROM_YEAR"]) if break_rows else pd.DataFrame(),
    )


def build_variable_documentation_artifact(
    variable_dict: pd.DataFrame,
    coverage: pd.DataFrame,
    harm_rules: pd.DataFrame,
) -> pd.DataFrame:
    coverage = coverage.copy()
    if not coverage.empty:
        coverage.columns = [str(col).strip().upper() for col in coverage.columns]

    dictionary = variable_dict.copy()
    if not dictionary.empty:
        dictionary.columns = [str(col).strip().lower() for col in dictionary.columns]

    rows = []
    variables = sorted(set(coverage["VARIABLE"].astype(str).str.upper())) if not coverage.empty else []
    dict_lookup = (
        dictionary.set_index(dictionary["variable_name"].astype(str).str.upper()).to_dict(orient="index")
        if not dictionary.empty and "variable_name" in dictionary.columns
        else {}
    )
    harm_vars = (
        set(harm_rules["variable_name"].astype(str).str.upper())
        if not harm_rules.empty and "variable_name" in harm_rules.columns
        else set()
    )

    for variable in variables:
        coverage_row = coverage.loc[coverage["VARIABLE"] == variable]
        years_available = [
            int(col.replace("YEAR_", ""))
            for col in coverage_row.columns
            if col.startswith("YEAR_") and int(coverage_row.iloc[0][col]) == 1
        ] if not coverage_row.empty else []
        meta = dict_lookup.get(variable, {})
        rows.append(
            {
                "variable_name": variable,
                "level": meta.get("level"),
                "dtype": meta.get("dtype"),
                "description": meta.get("description"),
                "use_in_analysis": meta.get("use_in_analysis"),
                "years_available": " | ".join(map(str, years_available)),
                "has_harmonization_rules": int(variable in harm_vars),
                "has_harmonized_version": int(variable.endswith("_H") or f"{variable}_H" in variables),
                "documented_in_metadata": int(variable in dict_lookup),
            }
        )

    return pd.DataFrame(rows)


def run_validation_suite(
    missing_rules: pd.DataFrame,
    harm_rules: pd.DataFrame,
    variable_dict: pd.DataFrame,
    coverage_path: Path = YEAR_COVERAGE_FILE,
) -> dict[str, Path]:
    OUTPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    coverage = parse_loose_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()

    paths: dict[str, Path] = {}
    metadata_reports = audit_metadata_usage(missing_rules, harm_rules, variable_dict, coverage=coverage)
    for report_name, df in metadata_reports.items():
        path = OUTPUT_REPORTS_DIR / f"{report_name}.csv"
        _write_report(df, path)
        paths[report_name] = path

    missingness = build_missingness_summary()
    path = OUTPUT_REPORTS_DIR / "missingness_summary_by_variable_year.csv"
    _write_report(missingness, path)
    paths["missingness_summary"] = path

    category_report = build_category_consistency_report()
    path = OUTPUT_REPORTS_DIR / "category_consistency_by_year.csv"
    _write_report(category_report, path)
    paths["category_consistency"] = path

    harmonization_coverage = build_harmonization_coverage_report(harm_rules)
    path = OUTPUT_REPORTS_DIR / "harmonization_coverage_report.csv"
    _write_report(harmonization_coverage, path)
    paths["harmonization_coverage"] = path

    outliers, structural_breaks = build_distribution_break_report()
    outlier_path = OUTPUT_REPORTS_DIR / "outlier_summary.csv"
    break_path = OUTPUT_REPORTS_DIR / "structural_break_flags.csv"
    _write_report(outliers, outlier_path)
    _write_report(structural_breaks, break_path)
    paths["outlier_summary"] = outlier_path
    paths["structural_breaks"] = break_path

    variable_doc = build_variable_documentation_artifact(variable_dict, coverage, harm_rules)
    path = OUTPUT_REPORTS_DIR / "variable_documentation.csv"
    _write_report(variable_doc, path)
    paths["variable_documentation"] = path

    return paths
