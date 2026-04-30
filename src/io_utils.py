from __future__ import annotations

from pathlib import Path
import pandas as pd

from analysis_utils import parse_loose_csv
from config import (
    CHUNK_SIZE,
    HARMONIZATION_RULES_FILE,
    MISSING_RULES_FILE,
    RAW_DATA_FILE,
    VARIABLE_DICTIONARY_FILE,
)


def ensure_parent_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def read_raw_in_chunks(chunksize: int | None = None):
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_DATA_FILE}")

    return pd.read_csv(
        RAW_DATA_FILE,
        chunksize=chunksize or CHUNK_SIZE,
        low_memory=False,
    )


def _load_metadata_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception:
        return parse_loose_csv(path)


def load_variable_dictionary() -> pd.DataFrame:
    df = _load_metadata_csv(VARIABLE_DICTIONARY_FILE)
    if not df.empty:
        df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def load_missing_rules() -> pd.DataFrame:
    df = _load_metadata_csv(MISSING_RULES_FILE)
    if not df.empty:
        df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def load_harmonization_rules() -> pd.DataFrame:
    df = _load_metadata_csv(HARMONIZATION_RULES_FILE)
    if not df.empty:
        df.columns = [str(col).strip().lower() for col in df.columns]
    return df
