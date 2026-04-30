from __future__ import annotations

import pandas as pd
import numpy as np


def build_household_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if not {"YEAR", "SERIAL"}.issubset(df.columns):
        raise ValueError("YEAR and SERIAL are required")

    working = df.copy()

    hh_size = (
        working.groupby(["YEAR", "SERIAL"])
        .size()
        .rename("HOUSEHOLD_SIZE")
        .reset_index()
    )

    first_cols = [
        "HHWT",
        "URBAN",
        "URBAN_H",
        "GEO1_VN",
        "OWNERSHIP",
        "OWNERSHIP_H",
        "ELECTRIC",
        "ELECTRIC_H",
        "WATSUP",
        "WATSUP_H",
        "SEWAGE",
        "SEWAGE_H",
        "AUTOS",
        "MOTORCYCLE",
        "ROOMS",
        "BEDROOMS",
        "WALL",
        "WALL_H",
        "ROOF",
        "ROOF_H",
        "LIVEAREA",
    ]

    agg_dict = {}
    for col in first_cols:
        if col in working.columns:
            agg_dict[col] = "first"

    hh = working.groupby(["YEAR", "SERIAL"], as_index=False).agg(agg_dict)
    hh = hh.merge(hh_size, on=["YEAR", "SERIAL"], how="left")

    if "LIVEAREA" in hh.columns:
        hh["AREA_PER_PERSON"] = hh["LIVEAREA"] / hh["HOUSEHOLD_SIZE"].replace(0, np.nan)

    if "ROOMS" in hh.columns:
        hh["ROOMS_PER_PERSON"] = hh["ROOMS"] / hh["HOUSEHOLD_SIZE"].replace(0, np.nan)

    if {"AUTOS", "MOTORCYCLE"}.issubset(hh.columns):
        hh["VEHICLE_COUNT"] = hh[["AUTOS", "MOTORCYCLE"]].fillna(0).sum(axis=1)

    return hh