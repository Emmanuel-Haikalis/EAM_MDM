import pandas as pd
from pathlib import Path

ASSET_REQUIRED = ["asset_id", "asset_name"]
CLASSIFICATION_REQUIRED = ["classification_code", "classification_name"]

_ASSET_TEXT_COLS = ["asset_type", "description", "location", "condition", "technical_specs"]
_CLASS_TEXT_COLS = ["classification_description", "category", "subcategory"]


def _load_file(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    raise ValueError(f"Unsupported format '{suffix}' — use CSV or Excel (.csv/.xlsx/.xls)")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    )
    return df


def _check_required(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_asset_register(path: str) -> pd.DataFrame:
    df = _normalise_columns(_load_file(path))
    _check_required(df, ASSET_REQUIRED, f"Asset register '{path}'")
    for col in _ASSET_TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").str.strip()
        else:
            df[col] = ""
    df["asset_id"] = df["asset_id"].fillna("").str.strip()
    df["asset_name"] = df["asset_name"].fillna("").str.strip()
    return df.reset_index(drop=True)


def load_classification_table(path: str, system_name: str) -> pd.DataFrame:
    df = _normalise_columns(_load_file(path))
    _check_required(df, CLASSIFICATION_REQUIRED, f"Classification table '{path}' ({system_name})")
    for col in _CLASS_TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").str.strip()
        else:
            df[col] = ""
    df["classification_code"] = df["classification_code"].fillna("").str.strip()
    df["classification_name"] = df["classification_name"].fillna("").str.strip()
    return df.reset_index(drop=True)
