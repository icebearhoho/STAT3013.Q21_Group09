from __future__ import annotations
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

METADATA_DIR = BASE_DIR / "metadata"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUT_TABLES_DIR = OUTPUTS_DIR / "tables"
OUTPUT_FIGURES_DIR = OUTPUTS_DIR / "figures"
OUTPUT_LOGS_DIR = OUTPUTS_DIR / "logs"
OUTPUT_MODELS_DIR = OUTPUTS_DIR / "models"
OUTPUT_REPORTS_DIR = OUTPUTS_DIR / "reports"

RAW_DATA_FILE = RAW_DIR / "ipumsi_00002.csv"

VARIABLE_DICTIONARY_FILE = METADATA_DIR / "variable_dictionary.csv"
MISSING_RULES_FILE = METADATA_DIR / "missing_rules.csv"
HARMONIZATION_RULES_FILE = METADATA_DIR / "harmonization_rules.csv"

CLEANED_MASTER_FILE = INTERIM_DIR / "cleaned_master.csv"
HARMONIZED_MASTER_FILE = INTERIM_DIR / "harmonized_master.csv"
AUDIT_REPORT_FILE = INTERIM_DIR / "audit_report.csv"
YEAR_COVERAGE_FILE = INTERIM_DIR / "year_coverage_matrix.csv"
LOGICAL_CHECKS_FILE = INTERIM_DIR / "logical_checks.csv"

HOUSEHOLD_READY_FILE = PROCESSED_DIR / "household_analysis_ready.csv"
PERSON_READY_FILE = PROCESSED_DIR / "person_analysis_ready.csv"

RANDOM_STATE = 42
CHUNK_SIZE = 100_000

HOUSEHOLD_SAMPLE_MAX_ROWS = 250_000
PERSON_SAMPLE_MAX_ROWS = 300_000
VALIDATION_SAMPLE_MAX_ROWS = 200_000
DEEP_LEARNING_MAX_ROWS = 150_000
BAYESIAN_NETWORK_MAX_ROWS = 60_000
SVM_MAX_ROWS = 40_000
CLUSTERING_MAX_ROWS = 50_000
ANOMALY_MAX_ROWS = 75_000

HOUSEHOLD_ID_COLS = ["YEAR", "SERIAL"]
PERSON_ID_COLS = ["YEAR", "SERIAL", "PERNUM"]
MIN_COLUMNS_REQUIRED = ["YEAR", "SERIAL", "PERNUM"]

HARMONIZE_VARS = [
    "URBAN",
    "OWNERSHIP",
    "ELECTRIC",
    "WATSUP",
    "SEWAGE",
    "WALL",
    "ROOF",
    "SEX",
    "MARST",
    "EDATTAIN",
    "CLASSWK",
]

CORE_CONTINUOUS_VARS = [
    "HHWT",
    "PERWT",
    "AUTOS",
    "MOTORCYCLE",
    "ROOMS",
    "BEDROOMS",
    "LIVEAREA",
    "NCHILD",
    "AGE",
    "CHBORN",
    "HOUSEHOLD_SIZE",
    "AREA_PER_PERSON",
    "wealth_index",
    "housing_quality_index",
]

CORE_CATEGORICAL_VARS = [
    "URBAN",
    "URBAN_H",
    "OWNERSHIP",
    "OWNERSHIP_H",
    "ELECTRIC",
    "ELECTRIC_H",
    "WATSUP",
    "WATSUP_H",
    "SEWAGE",
    "SEWAGE_H",
    "WALL",
    "WALL_H",
    "ROOF",
    "ROOF_H",
    "SEX",
    "SEX_H",
    "MARST",
    "MARST_H",
    "EDATTAIN",
    "EDATTAIN_H",
    "CLASSWK",
    "CLASSWK_H",
    "GEO1_VN",
    "wealth_class",
]

HOUSEHOLD_ASSET_VARS_BY_YEAR = {
    1989: ["LIVEAREA", "ELECTRIC_H", "WATSUP_H", "SEWAGE_H", "OWNERSHIP_H"],
    1999: ["LIVEAREA", "ELECTRIC_H", "WATSUP_H", "SEWAGE_H", "OWNERSHIP_H"],
    2009: ["MOTORCYCLE", "LIVEAREA", "ELECTRIC_H", "WATSUP_H", "AUTOS", "OWNERSHIP_H"],
    2019: ["MOTORCYCLE", "LIVEAREA", "ELECTRIC_H", "WATSUP_H", "SEWAGE_H", "AUTOS", "OWNERSHIP_H"],
}

HOUSING_QUALITY_VARS_BY_YEAR = {
    1989: ["ELECTRIC_H", "WATSUP_H", "SEWAGE_H", "LIVEAREA"],
    1999: ["ELECTRIC_H", "WATSUP_H", "SEWAGE_H", "LIVEAREA"],
    2009: ["ELECTRIC_H", "WATSUP_H", "LIVEAREA"],
    2019: ["ELECTRIC_H", "WATSUP_H", "SEWAGE_H", "LIVEAREA"],
}

PERSON_CORE_VARS = [
    "AGE",
    "SEX_H",
    "MARST_H",
    "EDATTAIN_H",
    "CLASSWK_H",
]

NUMERIC_CANDIDATES = [
    "YEAR",
    "HHWT",
    "PERWT",
    "AUTOS",
    "MOTORCYCLE",
    "ROOMS",
    "BEDROOMS",
    "LIVEAREA",
    "NCHILD",
    "AGE",
    "CHBORN",
    "PERNUM",
]


PERSON_MODEL_FEATURES = [
    "AGE",
    "SEX_H",
    "URBAN_H",
    "EDATTAIN_H",
    "CLASSWK_H",
    "YEAR",
    "GEO1_VN",
]

HOUSEHOLD_MODEL_FEATURES = [
    "URBAN_H",
    "ELECTRIC_H",
    "WATSUP_H",
    "YEAR",
    "GEO1_VN",
]

BAYESIAN_NETWORK_FEATURES = [
    "YEAR",
    "GEO1_VN",
    "URBAN_H",
    "ELECTRIC_H",
    "WATSUP_H",
    "wealth_class",
]