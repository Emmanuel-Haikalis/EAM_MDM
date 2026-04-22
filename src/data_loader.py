import pandas as pd
from pathlib import Path
from typing import Any, Dict, Tuple

ASSET_REQUIRED = ["asset_id", "asset_name"]
CLASSIFICATION_REQUIRED = ["classification_code", "classification_name"]

# Optional columns loaded into the asset text for similarity matching.
# Includes legacy fields common in real EAM systems.
_ASSET_TEXT_COLS = [
    "asset_type", "description", "location", "condition",
    "technical_specs", "existing_classification", "notes", "manufacturer_model",
]

# Optional columns loaded into the classification text for matching.
_CLASS_TEXT_COLS = [
    "classification_description", "category", "subcategory", "keywords",
]

# Columns whose absence significantly reduces match quality — warn but do not error.
_HIGH_VALUE_COLS = ["description", "technical_specs"]


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
        raise ValueError(
            f"{source} is missing required columns: {missing}\n"
            f"  Found columns : {list(df.columns)}\n"
            f"  Tip: run with --generate-templates to get a pre-formatted CSV template."
        )


def load_asset_register(path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and validate an asset register file.

    Returns
    -------
    (df, quality_stats)
        df           — cleaned DataFrame ready for classification
        quality_stats — dict with data-quality diagnostics
    """
    df = _normalise_columns(_load_file(path))
    _check_required(df, ASSET_REQUIRED, f"Asset register '{path}'")

    for col in _ASSET_TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").str.strip()
        else:
            df[col] = ""

    df["asset_id"] = df["asset_id"].fillna("").str.strip()
    df["asset_name"] = df["asset_name"].fillna("").str.strip()

    # Warn about columns that strongly affect match quality
    for col in _HIGH_VALUE_COLS:
        if col not in df.columns or (df[col] == "").all():
            print(f"  WARNING: '{col}' column is missing or empty — match quality will be reduced.")

    quality_stats = _compute_quality(df)

    if quality_stats["duplicate_count"] > 0:
        dupes = quality_stats["duplicate_asset_ids"]
        print(f"  WARNING: {quality_stats['duplicate_count']} duplicate asset_id(s) found: {dupes}")

    return df.reset_index(drop=True), quality_stats


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


def _compute_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute data-quality statistics for the asset register."""
    dup_ids = df[df.duplicated(subset=["asset_id"], keep=False)]["asset_id"].unique().tolist()
    monitored = [
        "asset_name", "asset_type", "description", "technical_specs",
        "location", "condition", "existing_classification", "notes",
    ]
    fields: Dict[str, Any] = {}
    for col in monitored:
        if col in df.columns:
            populated = (df[col].str.strip() != "").mean()
            fields[col] = {
                "pct_populated": round(populated * 100, 1),
                "pct_empty": round((1 - populated) * 100, 1),
            }
    return {
        "total_rows": len(df),
        "duplicate_count": len(dup_ids),
        "duplicate_asset_ids": dup_ids,
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------

_ASSET_TEMPLATE_ROWS = [
    {
        "asset_id": "AST001",
        "asset_name": "Main Street Bridge",
        "asset_type": "Bridge Structure",
        "description": "Reinforced concrete bridge over Main Creek carrying dual carriageway road traffic",
        "location": "Main Street over Main Creek",
        "condition": "Good",
        "technical_specs": "Span: 45m; Width: 12m; Load rating: Class A; Deck: RC slab",
        "existing_classification": "BRG-RC-45",
        "notes": "Last inspected 2023; minor spalling on soffit",
        "manufacturer_model": "",
    },
    {
        "asset_id": "AST002",
        "asset_name": "CBD Traffic Signal Controller TC-014",
        "asset_type": "Traffic Control",
        "description": "Adaptive traffic signal controller managing 4-way intersection with pedestrian phases",
        "location": "Elizabeth St and George St intersection",
        "condition": "Good",
        "technical_specs": "SCATS compatible; 8-phase; LED signal heads; loop detectors",
        "existing_classification": "TMG-SIGNAL-CB",
        "notes": "SCATS connected; TMC integration active",
        "manufacturer_model": "Siemens SEPAC",
    },
]

_CLASS_TEMPLATE_ROWS = [
    {
        "classification_code": "Ss_25_13_15",
        "classification_name": "Bridge Deck and Superstructure Systems",
        "classification_description": "Superstructure systems for bridge decks including reinforced concrete prestressed concrete and steel beam deck systems",
        "category": "Civil Engineering Systems",
        "subcategory": "Bridge and Tunnel Systems",
        "keywords": "bridge, deck, superstructure, beam, girder, span, viaduct",
        "parent_code": "Ss_25_13",
    },
    {
        "classification_code": "Ss_25_14_15",
        "classification_name": "Traffic Signal Control Systems",
        "classification_description": "Traffic signal systems at intersections including adaptive controllers signal heads loop detectors and SCATS",
        "category": "Civil Engineering Systems",
        "subcategory": "Traffic Management Systems",
        "keywords": "traffic signal, controller, SCATS, intersection, signal head, loop detector",
        "parent_code": "Ss_25_14",
    },
]


def write_templates(output_dir: str) -> Dict[str, str]:
    """Write CSV template files for asset register and classification table."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    asset_path = out / "template_asset_register.csv"
    class_path = out / "template_classification_table.csv"

    pd.DataFrame(_ASSET_TEMPLATE_ROWS).to_csv(asset_path, index=False)
    pd.DataFrame(_CLASS_TEMPLATE_ROWS).to_csv(class_path, index=False)

    return {"asset_register": str(asset_path), "classification_table": str(class_path)}
