"""
Edge-case test suite — converted from the original test_edge_cases.py script.
Each test maps 1-to-1 with an EC0N identifier for traceability.
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.classifier import classify_all, _calibrate_scores
from src.classifier_ml import classify_all_ml
from src.data_loader import load_asset_register, load_classification_table
from src.report_generator import generate_summary, save_results


# ---------------------------------------------------------------------------
# EC01 — Empty asset register
# ---------------------------------------------------------------------------

def test_ec01_empty_register_fast(single_cls):
    result = classify_all(pd.DataFrame(), single_cls, top_n=3)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_ec01_empty_register_ml(single_cls):
    result = classify_all_ml(pd.DataFrame(), single_cls, top_n=3)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# EC02 — Single asset, single system → top_n rows
# ---------------------------------------------------------------------------

def test_ec02_single_asset_single_system(single_cls):
    df = pd.DataFrame([{
        "asset_id": "A1", "asset_name": "Bridge",
        "asset_type": "Bridge Structure",
        "description": "Concrete bridge over creek",
    }])
    result = classify_all(df, single_cls, top_n=3)
    assert len(result) == 3
    assert set(result["match_rank"]) == {"1st", "2nd", "3rd"}


# ---------------------------------------------------------------------------
# EC03 — Duplicate asset_id → independent scores per row
# ---------------------------------------------------------------------------

def test_ec03_duplicate_asset_id_independent_scores(single_cls):
    df = pd.DataFrame([
        {
            "asset_id": "DUP", "asset_name": "Main Street Bridge",
            "asset_type": "Bridge",
            "description": "reinforced concrete bridge deck superstructure span",
        },
        {
            "asset_id": "DUP", "asset_name": "Hume Highway Pavement",
            "asset_type": "Pavement",
            "description": "flexible asphalt road pavement surface layer rutting",
        },
    ])
    result = classify_all(df, single_cls, top_n=1)
    rank1_codes = result[result["match_rank"] == "1st"]["matched_classification_code"].tolist()
    assert len(rank1_codes) == 2
    assert rank1_codes[0] != rank1_codes[1], (
        f"Data corruption: both duplicate rows produced code {rank1_codes[0]}"
    )


# ---------------------------------------------------------------------------
# EC04 — Blank asset_id → independent scores per row
# ---------------------------------------------------------------------------

def test_ec04_blank_asset_id_independent_scores(single_cls):
    df = pd.DataFrame([
        {
            "asset_id": "", "asset_name": "Bridge", "asset_type": "Bridge",
            "description": "concrete bridge deck superstructure span",
        },
        {
            "asset_id": "", "asset_name": "Pavement", "asset_type": "Pavement",
            "description": "asphalt flexible pavement layer rutting road",
        },
    ])
    result = classify_all(df, single_cls, top_n=1)
    rank1_codes = result[result["match_rank"] == "1st"]["matched_classification_code"].tolist()
    assert len(rank1_codes) == 2
    assert rank1_codes[0] != rank1_codes[1], (
        f"Blank asset_id collision: both rows got code {rank1_codes[0]}"
    )


# ---------------------------------------------------------------------------
# EC05 — Sparse asset (only id + name)
# ---------------------------------------------------------------------------

def test_ec05_sparse_asset(single_cls):
    df = pd.DataFrame([{"asset_id": "SPARSE", "asset_name": "Unknown Asset"}])
    result = classify_all(df, single_cls, top_n=1)
    assert len(result) == 1
    assert result.iloc[0]["confidence_flag"] in ("low", "medium", "high")
    assert result.iloc[0]["reasoning"]


# ---------------------------------------------------------------------------
# EC06 — Fully empty asset text → graceful low-confidence output
# ---------------------------------------------------------------------------

def test_ec06_empty_asset_text(single_cls):
    df = pd.DataFrame([{
        "asset_id": "EMPTY", "asset_name": "",
        "asset_type": "", "description": "", "technical_specs": "",
        "location": "", "notes": "", "existing_classification": "",
    }])
    result = classify_all(df, single_cls, top_n=1)
    assert len(result) == 1
    row = result.iloc[0]
    assert "Insufficient" in row["reasoning"] or row["confidence_flag"] in ("low", "medium")


# ---------------------------------------------------------------------------
# EC07 — top_n = 1
# ---------------------------------------------------------------------------

def test_ec07_top_n_one(single_cls):
    df = pd.DataFrame([{
        "asset_id": "T1", "asset_name": "Bridge",
        "description": "concrete bridge deck",
    }])
    result = classify_all(df, single_cls, top_n=1)
    assert len(result) == 1
    assert result.iloc[0]["match_rank"] == "1st"


# ---------------------------------------------------------------------------
# EC08 — top_n > table size → capped at table size
# ---------------------------------------------------------------------------

def test_ec08_top_n_exceeds_table_size(single_cls):
    df = pd.DataFrame([{
        "asset_id": "T1", "asset_name": "Bridge",
        "description": "concrete bridge deck",
    }])
    result = classify_all(df, single_cls, top_n=30)
    assert len(result) == 3  # capped at 3-entry table


# ---------------------------------------------------------------------------
# EC09 — Ordinal rank labels (21st not 21th)
# ---------------------------------------------------------------------------

def test_ec09_ordinal_rank_labels():
    big_cls = pd.DataFrame([
        {
            "classification_code": f"X{i:03d}",
            "classification_name": f"Class {i}",
            "classification_description": f"description for class {i}",
            "category": "General",
            "subcategory": "Sub",
        }
        for i in range(1, 25)
    ])
    df = pd.DataFrame([{
        "asset_id": "T1", "asset_name": "Test asset",
        "description": "test description",
    }])
    result = classify_all(df, {"BIG": big_cls}, top_n=22)
    ranks = result["match_rank"].tolist()
    assert "21th" not in ranks, f"Found '21th' — should be '21st'"
    assert "22th" not in ranks, f"Found '22th' — should be '22nd'"


# ---------------------------------------------------------------------------
# EC10 — Single system → no cross_system_consistency in summary
# ---------------------------------------------------------------------------

def test_ec10_single_system_no_cross_system_summary(single_cls):
    df = pd.DataFrame([{
        "asset_id": "T1", "asset_name": "Bridge",
        "description": "bridge deck concrete",
    }])
    result = classify_all(df, single_cls, top_n=1)
    summary = generate_summary(result, ["CLS"])
    assert "cross_system_consistency" not in summary


# ---------------------------------------------------------------------------
# EC11 — Special characters (unicode, punctuation)
# ---------------------------------------------------------------------------

def test_ec11_special_characters(single_cls):
    df = pd.DataFrame([{
        "asset_id": "SPECIAL",
        "asset_name": "Brücke über den Fluss",
        "asset_type": "Bridge/Culvert & Structure",
        "description": 'Reinforced "concrete" bridge; width=12m, load≥50t — Class A',
        "technical_specs": "Span: 45m | RC Slab | Load: Class A",
    }])
    result = classify_all(df, single_cls, top_n=1)
    assert len(result) == 1
    assert isinstance(result.iloc[0]["similarity_score"], float)


# ---------------------------------------------------------------------------
# EC12 — Single-entry classification table → calibrated score = 50.0
# ---------------------------------------------------------------------------

def test_ec12_single_entry_classification():
    one_cls = pd.DataFrame([{
        "classification_code": "ONLY",
        "classification_name": "Only Option",
        "classification_description": "The only classification",
        "category": "X",
        "subcategory": "",
    }])
    df = pd.DataFrame([{
        "asset_id": "T1", "asset_name": "Something",
        "description": "something",
    }])
    result = classify_all(df, {"ONE": one_cls}, top_n=1)
    assert len(result) == 1
    assert result.iloc[0]["similarity_score"] == 50.0, (
        f"Expected 50.0 when single-entry table, got {result.iloc[0]['similarity_score']}"
    )


# ---------------------------------------------------------------------------
# EC13 — Very long description → completes in <5 s
# ---------------------------------------------------------------------------

def test_ec13_long_description(single_cls):
    import time
    long_desc = " ".join(["bridge concrete reinforced structure deck span culvert"] * 150)
    df = pd.DataFrame([{
        "asset_id": "LONG", "asset_name": "Long asset",
        "description": long_desc,
    }])
    t0 = time.time()
    result = classify_all(df, single_cls, top_n=3)
    elapsed = time.time() - t0
    assert len(result) == 3
    assert elapsed < 5.0, f"Too slow for single asset with long desc: {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# EC14 — CSV with Windows CRLF + column name whitespace
# ---------------------------------------------------------------------------

def test_ec14_windows_csv_whitespace():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("asset_id ,asset_name ,asset_type \r\n")
        f.write("W001 ,Windows Bridge ,Bridge \r\n")
        fname = f.name
    try:
        df, _ = load_asset_register(fname)
        assert "asset_id" in df.columns
        assert df.iloc[0]["asset_id"] == "W001"
    finally:
        os.unlink(fname)


# ---------------------------------------------------------------------------
# EC15 — _calibrate_scores: all-identical input → 50.0
# ---------------------------------------------------------------------------

def test_ec15_calibration_all_identical():
    arr = np.array([30.0, 30.0, 30.0])
    cal = _calibrate_scores({"SYS": {"A1": arr}})
    assert np.allclose(cal["SYS"]["A1"], 50.0), (
        f"Expected all 50.0, got {cal['SYS']['A1']}"
    )


# ---------------------------------------------------------------------------
# EC16 — generate_summary with empty DataFrame
# ---------------------------------------------------------------------------

def test_ec16_generate_summary_empty():
    empty_df = pd.DataFrame(columns=[
        "asset_id", "asset_name", "classification_system",
        "matched_classification_code", "matched_classification_name",
        "similarity_score", "match_rank", "confidence_flag",
        "reasoning", "matched_category", "score_breakdown",
    ])
    summary = generate_summary(empty_df, ["SYS1", "SYS2"])
    assert summary["total_assets_processed"] == 0
    assert summary["manual_review_count"] == 0


# ---------------------------------------------------------------------------
# EC17 — save_results with empty DataFrame
# ---------------------------------------------------------------------------

def test_ec17_save_results_empty():
    tmpdir = tempfile.mkdtemp()
    try:
        empty_df = pd.DataFrame(columns=[
            "asset_id", "asset_name", "classification_system",
            "matched_classification_code", "matched_classification_name",
            "similarity_score", "match_rank", "confidence_flag",
            "reasoning", "matched_category", "score_breakdown",
        ])
        paths = save_results(empty_df, tmpdir)
        assert os.path.exists(paths["csv"])
        assert os.path.exists(paths["excel"])
    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# EC18 — Multi-system summary with a system that has no rows
# ---------------------------------------------------------------------------

def test_ec18_summary_missing_system():
    rows = [
        {
            "asset_id": "X1", "asset_name": "X",
            "classification_system": "S1", "match_rank": "1st",
            "similarity_score": 75.0, "confidence_flag": "high",
            "matched_category": "Structures",
            "matched_classification_code": "A001",
        },
        {
            "asset_id": "X1", "asset_name": "X",
            "classification_system": "S2", "match_rank": "1st",
            "similarity_score": 60.0, "confidence_flag": "medium",
            "matched_category": "Structures",
            "matched_classification_code": "A001",
        },
    ]
    df = pd.DataFrame(rows)
    summary = generate_summary(df, ["S1", "S2", "S3"])
    assert "S1" in summary["systems"]
    assert "S2" in summary["systems"]
    assert "S3" not in summary["systems"]


# ---------------------------------------------------------------------------
# EC19 — ML mode: duplicate asset_id → independent scores
# ---------------------------------------------------------------------------

def test_ec19_ml_duplicate_asset_id(single_cls):
    df = pd.DataFrame([
        {
            "asset_id": "DUP", "asset_name": "Main Street Bridge",
            "asset_type": "Bridge",
            "description": "reinforced concrete bridge deck superstructure",
        },
        {
            "asset_id": "DUP", "asset_name": "Hume Highway Pavement",
            "asset_type": "Pavement",
            "description": "flexible asphalt pavement layer rutting fatigue",
        },
    ])
    result = classify_all_ml(df, single_cls, top_n=1)
    rank1_codes = result[result["match_rank"] == "1st"]["matched_classification_code"].tolist()
    assert len(rank1_codes) == 2
    assert rank1_codes[0] != rank1_codes[1], (
        f"ML mode corruption: both duplicate rows got code {rank1_codes[0]}"
    )


# ---------------------------------------------------------------------------
# EC20 — ML mode: empty asset register
# ---------------------------------------------------------------------------

def test_ec20_ml_empty_register(single_cls):
    result = classify_all_ml(pd.DataFrame(), single_cls, top_n=3)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# EC21 — Excel (.xlsx) asset register loading
# ---------------------------------------------------------------------------

def test_ec21_excel_asset_register():
    tmpdir = tempfile.mkdtemp()
    try:
        xlsx = os.path.join(tmpdir, "test.xlsx")
        pd.DataFrame([{
            "asset_id": "E001", "asset_name": "Excel Bridge",
            "asset_type": "Bridge", "description": "Excel-loaded bridge",
        }]).to_excel(xlsx, index=False)
        df, _ = load_asset_register(xlsx)
        assert len(df) == 1
        assert df.iloc[0]["asset_id"] == "E001"
    finally:
        shutil.rmtree(tmpdir)


# ---------------------------------------------------------------------------
# EC22 — Unsupported file format raises clear error
# ---------------------------------------------------------------------------

def test_ec22_unsupported_file_format():
    with pytest.raises((ValueError, FileNotFoundError)) as exc_info:
        load_asset_register("/tmp/test.json")
    msg = str(exc_info.value).lower()
    assert "json" in msg or "not found" in msg


# ---------------------------------------------------------------------------
# EC23 — Missing required column raises ValueError naming the column
# ---------------------------------------------------------------------------

def test_ec23_missing_required_column():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("asset_name,description\n")
        f.write("Bridge,A concrete bridge\n")
        fname = f.name
    try:
        with pytest.raises(ValueError, match="asset_id"):
            load_asset_register(fname)
    finally:
        os.unlink(fname)
