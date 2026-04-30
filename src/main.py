from __future__ import annotations

import pandas as pd

from analysis_utils import load_selected_columns, priority_sample_csv, set_global_seed
from bayesian_network import run_bayesian_network_analysis
from build_household_dataset import build_household_dataset
from build_person_dataset import build_person_dataset
from clean_master_data import apply_missing_rules_by_year, basic_type_cleaning
from config import (
    AUDIT_REPORT_FILE,
    CLEANED_MASTER_FILE,
    HARMONIZE_VARS,
    HARMONIZED_MASTER_FILE,
    HOUSEHOLD_ASSET_VARS_BY_YEAR,
    HOUSEHOLD_READY_FILE,
    HOUSEHOLD_SAMPLE_MAX_ROWS,
    HOUSING_QUALITY_VARS_BY_YEAR,
    LOGICAL_CHECKS_FILE,
    OUTPUT_FIGURES_DIR,
    OUTPUT_MODELS_DIR,
    OUTPUT_REPORTS_DIR,
    OUTPUT_TABLES_DIR,
    PERSON_READY_FILE,
    PERSON_SAMPLE_MAX_ROWS,
    RAW_DATA_FILE,
    YEAR_COVERAGE_FILE,
)
from deep_learning import run_deep_learning_analysis
from descriptive import summary_statistics, yearly_mean
from econometrics import run_household_models, run_person_returns_model
from education_mobility import run_education_mobility_analysis
from gender_analysis import run_gender_analysis
from harmonize import harmonize_master
from indices import build_pca_index, create_wealth_class
from inequality import yearly_gini
from io_utils import ensure_parent_dirs, load_harmonization_rules, load_missing_rules, load_variable_dictionary, read_raw_in_chunks
from logging_utils import setup_logger
from ml_models import (
    model_feature_importance,
    run_unsupervised_tabular_analysis,
    subgroup_model_performance,
    temporal_wealth_validation,
    train_wealth_models,
)
from regional_analysis import run_regional_analysis
from robustness_checks import run_robustness_checks
from validate_data import run_validation_suite
from validators import (
    audit_chunk,
    combine_audit_reports,
    combine_simple_check_tables,
    combine_year_coverage,
    duplicate_checks,
    logical_checks,
    standardize_columns,
    validate_required_columns,
    year_coverage_from_chunk,
)
from visualize import build_research_dashboard, generate_figures


logger = setup_logger()


def _delete_if_exists(path):
    if path.exists():
        path.unlink()


def _append_csv(df: pd.DataFrame, path, first_write: bool) -> bool:
    df.to_csv(path, mode="w" if first_write else "a", header=first_write, index=False)
    return False


def process_raw_streaming(missing_rules, harm_rules):
    audit_parts = []
    coverage_parts = []
    dup_parts = []
    logic_parts = []

    first_chunk_checked = False
    first_cleaned_write = True
    first_harmonized_write = True
    first_household_write = True
    first_person_write = True

    for path in [
        CLEANED_MASTER_FILE,
        HARMONIZED_MASTER_FILE,
        HOUSEHOLD_READY_FILE,
        PERSON_READY_FILE,
        AUDIT_REPORT_FILE,
        YEAR_COVERAGE_FILE,
        LOGICAL_CHECKS_FILE,
    ]:
        _delete_if_exists(path)

    for i, chunk in enumerate(read_raw_in_chunks(), start=1):
        logger.info("[chunk %s] standardizing and auditing", i)
        chunk = standardize_columns(chunk)

        if not first_chunk_checked:
            validate_required_columns(chunk)
            first_chunk_checked = True

        audit_parts.append(audit_chunk(chunk))
        coverage_parts.append(year_coverage_from_chunk(chunk))
        dup_parts.append(duplicate_checks(chunk))
        logic_parts.append(logical_checks(chunk))

        cleaned = apply_missing_rules_by_year(chunk, missing_rules)
        cleaned = basic_type_cleaning(cleaned)
        first_cleaned_write = _append_csv(cleaned, CLEANED_MASTER_FILE, first_cleaned_write)

        harmonized = harmonize_master(cleaned, harm_rules, HARMONIZE_VARS)
        first_harmonized_write = _append_csv(harmonized, HARMONIZED_MASTER_FILE, first_harmonized_write)

        hh_chunk = build_household_dataset(harmonized)
        person_chunk = build_person_dataset(harmonized)
        first_household_write = _append_csv(hh_chunk, HOUSEHOLD_READY_FILE, first_household_write)
        first_person_write = _append_csv(person_chunk, PERSON_READY_FILE, first_person_write)

    audit = combine_audit_reports(audit_parts)
    coverage = combine_year_coverage(coverage_parts)
    dups = combine_simple_check_tables(dup_parts, "CHECK")
    logic = combine_simple_check_tables(logic_parts, "CHECK")
    checks = pd.concat([logic, dups], ignore_index=True) if not dups.empty else logic

    audit.to_csv(AUDIT_REPORT_FILE, index=False)
    coverage.to_csv(YEAR_COVERAGE_FILE, index=False)
    checks.to_csv(LOGICAL_CHECKS_FILE, index=False)
    logger.info("Streaming pipeline completed")


def deduplicate_household_file() -> pd.DataFrame:
    logger.info("Deduplicating household analysis file")
    hh = pd.read_csv(HOUSEHOLD_READY_FILE, low_memory=False)
    hh.columns = [str(col).strip().upper() for col in hh.columns]
    hh = hh.drop_duplicates(subset=["YEAR", "SERIAL"], keep="first")
    hh.to_csv(HOUSEHOLD_READY_FILE, index=False)
    return hh


def build_household_indices(hh: pd.DataFrame) -> pd.DataFrame:
    processed = []
    years = sorted(pd.to_numeric(hh["YEAR"], errors="coerce").dropna().astype(int).unique())
    for year in years:
        year_df = hh.loc[hh["YEAR"] == year].copy()
        wealth_vars = [col for col in HOUSEHOLD_ASSET_VARS_BY_YEAR.get(year, []) if col in year_df.columns]
        housing_vars = [col for col in HOUSING_QUALITY_VARS_BY_YEAR.get(year, []) if col in year_df.columns]

        wealth_vars = [col for col in wealth_vars if year_df[col].nunique(dropna=True) > 1]
        housing_vars = [col for col in housing_vars if year_df[col].nunique(dropna=True) > 1]

        if len(wealth_vars) >= 3:
            year_df, loadings = build_pca_index(year_df, wealth_vars, "wealth_index")
            year_df = create_wealth_class(year_df, "wealth_index")
            loadings.to_csv(OUTPUT_TABLES_DIR / f"wealth_pca_loadings_{year}.csv", index=False)

        if len(housing_vars) >= 3:
            year_df, loadings = build_pca_index(year_df, housing_vars, "housing_quality_index")
            loadings.to_csv(OUTPUT_TABLES_DIR / f"housing_pca_loadings_{year}.csv", index=False)

        processed.append(year_df)

    enriched = pd.concat(processed, ignore_index=True)
    enriched.to_csv(HOUSEHOLD_READY_FILE, index=False)
    return enriched


def build_urban_rural_summary(hh: pd.DataFrame) -> pd.DataFrame:
    urban_col = "URBAN_H" if "URBAN_H" in hh.columns else "URBAN"
    if "wealth_index" not in hh.columns or urban_col not in hh.columns:
        return pd.DataFrame()
    summary = (
        hh.groupby(["YEAR", urban_col], as_index=False)
        .agg(mean_wealth_index=("wealth_index", "mean"), n_households=("SERIAL", "size"))
        .rename(columns={urban_col: "URBAN_CODE"})
    )
    summary["URBAN_LABEL"] = summary["URBAN_CODE"].map({"1": "Urban", "0": "Rural", 1: "Urban", 0: "Rural"}).fillna(summary["URBAN_CODE"].astype(str))
    summary.to_csv(OUTPUT_TABLES_DIR / "urban_rural_wealth_by_year.csv", index=False)
    return summary


def build_person_regression_sample(hh: pd.DataFrame) -> pd.DataFrame:
    wealth_lookup = hh[[col for col in ["YEAR", "SERIAL", "wealth_index"] if col in hh.columns]].copy()
    sample = priority_sample_csv(
        PERSON_READY_FILE,
        usecols=["YEAR", "SERIAL", "PERWT", "AGE", "SEX_H", "EDATTAIN_H", "URBAN_H", "GEO1_VN"],
        max_rows=PERSON_SAMPLE_MAX_ROWS,
    )
    return sample.merge(wealth_lookup, on=["YEAR", "SERIAL"], how="left")


def run_analysis(missing_rules, harm_rules, variable_dict):
    logger.info("Running validation suite")
    run_validation_suite(missing_rules, harm_rules, variable_dict)

    logger.info("Loading household dataset")
    hh = deduplicate_household_file()
    hh = build_household_indices(hh)

    summary_cols = [
        "YEAR",
        "HOUSEHOLD_SIZE",
        "LIVEAREA",
        "AREA_PER_PERSON",
        "ROOMS",
        "ROOMS_PER_PERSON",
        "wealth_index",
        "housing_quality_index",
    ]
    summary_statistics(hh, [col for col in summary_cols if col in hh.columns]).to_csv(
        OUTPUT_TABLES_DIR / "household_summary_statistics.csv"
    )

    if "wealth_index" in hh.columns:
        yearly_mean(hh, "wealth_index", weight_col="HHWT").to_csv(
            OUTPUT_TABLES_DIR / "yearly_weighted_wealth_mean.csv",
            index=False,
        )
        yearly_gini(hh, "wealth_index", weight_col="HHWT").to_csv(
            OUTPUT_TABLES_DIR / "yearly_weighted_gini.csv",
            index=False,
        )
        build_urban_rural_summary(hh)

    logger.info("Running econometrics")
    run_household_models(hh)
    person_sample = build_person_regression_sample(hh)
    run_person_returns_model(person_sample)

    logger.info("Running gender, mobility, and regional analyses")
    run_gender_analysis()
    run_education_mobility_analysis()
    run_regional_analysis()

    logger.info("Running traditional ML")
    model_input = hh.sample(n=min(HOUSEHOLD_SAMPLE_MAX_ROWS, len(hh)), random_state=42) if len(hh) > HOUSEHOLD_SAMPLE_MAX_ROWS else hh.copy()
    ml_results = train_wealth_models(model_input, target_col="wealth_class")
    temporal_wealth_validation(model_input, target_col="wealth_class")
    model_feature_importance(model_input, target_col="wealth_class")
    subgroup_model_performance(model_input, target_col="wealth_class")
    with open(OUTPUT_TABLES_DIR / "ml_results.txt", "w", encoding="utf-8") as handle:
        for model_name, result in ml_results.items():
            handle.write(f"===== {model_name.upper()} =====\n")
            handle.write(f"Accuracy: {result['accuracy']:.4f}\n")
            handle.write(f"Macro F1: {result['macro_f1']:.4f}\n")
            handle.write(result["report"])
            handle.write("\n\n")

    logger.info("Running deep learning")
    deep_results = run_deep_learning_analysis(model_input)

    logger.info("Running Bayesian network")
    bn_results = run_bayesian_network_analysis(model_input)

    logger.info("Running clustering and anomaly models")
    unsupervised_results = run_unsupervised_tabular_analysis(model_input)

    comparison = pd.read_csv(OUTPUT_TABLES_DIR / "traditional_model_comparison.csv")
    deep_path = OUTPUT_TABLES_DIR / "deep_learning_model_comparison.csv"
    bn_path = OUTPUT_TABLES_DIR / "bayesian_network_prediction.csv"
    comparison_frames = [comparison]
    if deep_path.exists():
        deep_df = pd.read_csv(deep_path)
        comparison_frames.append(deep_df)
    if bn_path.exists():
        bn_df = pd.read_csv(bn_path)[["model", "accuracy", "macro_f1"]]
        comparison_frames.append(bn_df)
    pd.concat(comparison_frames, ignore_index=True).to_csv(OUTPUT_TABLES_DIR / "model_comparison_all.csv", index=False)

    logger.info("Running robustness checks")
    run_robustness_checks(hh)

    logger.info("Generating figures")
    generate_figures()
    build_research_dashboard()

    summary = pd.DataFrame(
        [
            {
                "streaming_stage_used": int(RAW_DATA_FILE.exists()),
                "households_analyzed": len(hh),
                "person_sample_for_regressions": len(person_sample),
                "traditional_models_run": len(ml_results),
                "deep_learning_components_run": len(deep_results),
                "bayesian_network_run": int(bool(bn_results)),
                "unsupervised_models_run": len(unsupervised_results),
            }
        ]
    )
    summary.to_csv(OUTPUT_REPORTS_DIR / "run_summary.csv", index=False)


def main():
    set_global_seed()
    ensure_parent_dirs(
        [
            CLEANED_MASTER_FILE,
            HARMONIZED_MASTER_FILE,
            AUDIT_REPORT_FILE,
            YEAR_COVERAGE_FILE,
            LOGICAL_CHECKS_FILE,
            HOUSEHOLD_READY_FILE,
            PERSON_READY_FILE,
            OUTPUT_TABLES_DIR / "placeholder.txt",
            OUTPUT_FIGURES_DIR / "placeholder.txt",
            OUTPUT_REPORTS_DIR / "placeholder.txt",
            OUTPUT_MODELS_DIR / "placeholder.txt",
        ]
    )

    logger.info("Loading metadata")
    missing_rules = load_missing_rules()
    harm_rules = load_harmonization_rules()
    variable_dict = load_variable_dictionary()

    if RAW_DATA_FILE.exists():
        logger.info("Raw file found. Running streaming data pipeline")
        process_raw_streaming(missing_rules, harm_rules)
    elif HOUSEHOLD_READY_FILE.exists() and PERSON_READY_FILE.exists() and HARMONIZED_MASTER_FILE.exists():
        logger.info("Raw file not found. Using existing processed datasets and harmonized master")
    else:
        raise FileNotFoundError(
            "Neither raw input nor the required processed datasets are available. "
            "Provide data/raw/ipumsi_00002.csv or restore the processed outputs."
        )

    run_analysis(missing_rules, harm_rules, variable_dict)
    logger.info("Pipeline completed")


if __name__ == "__main__":
    main()
