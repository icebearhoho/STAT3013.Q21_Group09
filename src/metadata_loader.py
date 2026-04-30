from __future__ import annotations

import pandas as pd


MISSING_RULE_COLUMNS = {"variable_name", "year", "missing_codes"}
HARMONIZATION_RULE_COLUMNS = {"variable_name", "year", "original_code", "harmonized_code"}


def parse_missing_codes(code_str) -> list[str]:
    if pd.isna(code_str):
        return []
    return [x.strip() for x in str(code_str).split("|") if x.strip()]


def _normalize_rules(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    return out


def get_missing_rules_for_var_year(missing_rules: pd.DataFrame, variable_name: str, year) -> list[str]:
    rules = _normalize_rules(missing_rules)
    if rules.empty or not MISSING_RULE_COLUMNS.issubset(rules.columns):
        return []

    rules["variable_name"] = rules["variable_name"].astype(str).str.upper()
    rules["year"] = rules["year"].astype(str).str.upper()

    variable_name = str(variable_name).upper()
    year = str(year).upper()

    specific = rules[(rules["variable_name"] == variable_name) & (rules["year"] == year)]
    if not specific.empty:
        return parse_missing_codes(specific.iloc[0]["missing_codes"])

    fallback = rules[(rules["variable_name"] == variable_name) & (rules["year"] == "ALL")]
    if not fallback.empty:
        return parse_missing_codes(fallback.iloc[0]["missing_codes"])

    return []


def get_harmonization_map_for_var_year(harm_rules: pd.DataFrame, variable_name: str, year) -> dict[str, str]:
    rules = _normalize_rules(harm_rules)
    if rules.empty or not HARMONIZATION_RULE_COLUMNS.issubset(rules.columns):
        return {}

    rules["variable_name"] = rules["variable_name"].astype(str).str.upper()
    rules["year"] = rules["year"].astype(str).str.upper()

    variable_name = str(variable_name).upper()
    year = str(year).upper()

    subset = rules[(rules["variable_name"] == variable_name) & (rules["year"] == year)]
    if subset.empty:
        subset = rules[(rules["variable_name"] == variable_name) & (rules["year"] == "ALL")]

    mapping = {}
    for _, row in subset.iterrows():
        raw_code = str(row["original_code"]).strip()
        mapping[raw_code] = row["harmonized_code"]
    return mapping


def metadata_schema_report(df: pd.DataFrame, required_columns: set[str], name: str) -> pd.DataFrame:
    normalized = _normalize_rules(df)
    present = set(normalized.columns)
    rows = []
    for column in sorted(required_columns):
        rows.append(
            {
                "metadata_file": name,
                "column_name": column,
                "present": int(column in present),
            }
        )
    return pd.DataFrame(rows)
