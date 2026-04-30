from __future__ import annotations

import pandas as pd


def build_person_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required = ["YEAR", "SERIAL", "PERNUM"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing person id columns: {missing}")

    keep_cols = [
        "YEAR",
        "SERIAL",
        "PERNUM",
        "PERWT",
        "AGE",
        "SEX",
        "SEX_H",
        "MARST",
        "MARST_H",
        "CHBORN",
        "NCHILD",
        "EDATTAIN",
        "EDATTAIN_H",
        "CLASSWK",
        "CLASSWK_H",
        "URBAN",
        "URBAN_H",
        "GEO1_VN",
    ]

    available = [c for c in keep_cols if c in df.columns]
    person = df[available].copy()

    if "AGE" in person.columns:
        person["IS_WORKING_AGE"] = ((person["AGE"] >= 15) & (person["AGE"] <= 64)).astype(float)

    return person




