from __future__ import annotations

import pandas as pd

from metadata_loader import get_harmonization_map_for_var_year


def harmonize_variable_by_year(df: pd.DataFrame, harm_rules: pd.DataFrame, variable_name: str) -> pd.DataFrame:
    out = df.copy()
    variable_name = variable_name.upper()

    if variable_name not in out.columns:
        return out

    harm_col = f"{variable_name}_H"
    out[harm_col] = out[variable_name]

    years = sorted(pd.Series(out["YEAR"]).dropna().unique())
    for year in years:
        mapping = get_harmonization_map_for_var_year(harm_rules, variable_name, year)
        if not mapping:
            continue

        mask = out["YEAR"] == year
        raw_vals = out.loc[mask, variable_name].astype(str).str.strip()
        mapped = raw_vals.map(mapping)
        out.loc[mask, harm_col] = mapped.where(mapped.notna(), out.loc[mask, variable_name])

    return out


def harmonize_master(df: pd.DataFrame, harm_rules: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    out = df.copy()
    for var in variables:
        out = harmonize_variable_by_year(out, harm_rules, var)
    return out