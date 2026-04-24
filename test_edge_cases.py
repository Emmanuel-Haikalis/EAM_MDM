"""
Edge-case test suite for the asset classification tool.
Run:  python test_edge_cases.py
Each test prints PASS / FAIL and the failure reason.
"""
import io, sys, os, traceback, tempfile, shutil
import pandas as pd
import numpy as np

sys.path.insert(0, "/home/user/EAM_MDM")
from src.classifier import classify_all, _calibrate_scores, _RANK_LABEL
from src.classifier_ml import classify_all_ml
from src.data_loader import load_asset_register, load_classification_table
from src.report_generator import generate_summary, save_summary, save_results

PASS = "PASS"
FAIL = "FAIL"
results = []


def test(name, fn):
    try:
        fn()
        results.append((PASS, name, ""))
        print(f"  {PASS}  {name}")
    except AssertionError as e:
        results.append((FAIL, name, str(e)))
        print(f"  {FAIL}  {name}: {e}")
    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()[-1]
        results.append((FAIL, name, tb))
        print(f"  {FAIL}  {name}: {tb}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_cls_df():
    """3-entry classification table (deliberately small to test edge cases)."""
    return pd.DataFrame([
        {"classification_code": "A001", "classification_name": "Bridge Structures",
         "classification_description": "Reinforced concrete bridges and culverts",
         "category": "Structures", "subcategory": "Bridges"},
        {"classification_code": "A002", "classification_name": "Road Pavement Systems",
         "classification_description": "Flexible and rigid pavement layers",
         "category": "Pavements", "subcategory": "Roads"},
        {"classification_code": "A003", "classification_name": "Drainage Systems",
         "classification_description": "Stormwater pipes culverts and channels",
         "category": "Drainage", "subcategory": "Stormwater"},
    ])

def make_asset_df(rows):
    return pd.DataFrame(rows)

SINGLE_CLS = {"CLS": make_cls_df()}
MULTI_CLS  = {
    "SYS1": make_cls_df(),
    "SYS2": make_cls_df(),
}

# ── EC01: Empty asset register ────────────────────────────────────────────────
def ec01():
    empty = make_asset_df([])
    result = classify_all(empty, SINGLE_CLS, top_n=3)
    assert isinstance(result, pd.DataFrame), "Expected DataFrame"
    assert len(result) == 0, f"Expected 0 rows, got {len(result)}"

test("EC01 empty asset register → empty DataFrame (fast)", ec01)

def ec01_ml():
    empty = make_asset_df([])
    result = classify_all_ml(empty, SINGLE_CLS, top_n=3)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0

test("EC01 empty asset register → empty DataFrame (ml)", ec01_ml)

# ── EC02: Single asset, single system ─────────────────────────────────────────
def ec02():
    df = make_asset_df([{"asset_id": "A1", "asset_name": "Bridge",
                          "asset_type": "Bridge Structure",
                          "description": "Concrete bridge over creek"}])
    result = classify_all(df, SINGLE_CLS, top_n=3)
    assert len(result) == 3, f"Expected 3 rows (top_n=3 × 1 asset × 1 system), got {len(result)}"
    assert set(result["match_rank"]) == {"1st", "2nd", "3rd"}

test("EC02 single asset single system → 3 rows", ec02)

# ── EC03: Duplicate asset_id — each asset must get its OWN scores ─────────────
def ec03():
    """
    Two assets with the same asset_id but completely different descriptions.
    After classification, their top matches must differ (bridge vs pavement).
    If scores are overwritten, both would have the same rank-1 match.
    """
    df = make_asset_df([
        {"asset_id": "DUP", "asset_name": "Main Street Bridge",
         "asset_type": "Bridge", "description": "reinforced concrete bridge deck superstructure span"},
        {"asset_id": "DUP", "asset_name": "Hume Highway Pavement",
         "asset_type": "Pavement", "description": "flexible asphalt road pavement surface layer rutting"},
    ])
    result = classify_all(df, SINGLE_CLS, top_n=1)
    rank1 = result[result["match_rank"] == "1st"]["matched_classification_code"].tolist()
    assert len(rank1) == 2, f"Expected 2 rank-1 rows, got {len(rank1)}"
    # Bridge asset should match A001, pavement asset should match A002
    assert rank1[0] != rank1[1], (
        f"Duplicate asset_id data corruption: both rows got same code {rank1[0]}"
    )

test("EC03 duplicate asset_id → each row gets independent scores", ec03)

# ── EC04: Blank asset_id ──────────────────────────────────────────────────────
def ec04():
    """Two assets with blank asset_id must each get independent scores."""
    df = make_asset_df([
        {"asset_id": "", "asset_name": "Bridge", "asset_type": "Bridge",
         "description": "concrete bridge deck superstructure span"},
        {"asset_id": "", "asset_name": "Pavement", "asset_type": "Pavement",
         "description": "asphalt flexible pavement layer rutting road"},
    ])
    result = classify_all(df, SINGLE_CLS, top_n=1)
    rank1 = result[result["match_rank"] == "1st"]["matched_classification_code"].tolist()
    assert len(rank1) == 2, f"Expected 2 rank-1 rows, got {len(rank1)}"
    assert rank1[0] != rank1[1], (
        f"Blank asset_id collision: both rows got same code {rank1[0]}"
    )

test("EC04 blank asset_id → each row gets independent scores", ec04)

# ── EC05: Asset with only asset_id + asset_name (all other fields empty) ──────
def ec05():
    df = make_asset_df([{"asset_id": "SPARSE", "asset_name": "Unknown Asset"}])
    result = classify_all(df, SINGLE_CLS, top_n=1)
    assert len(result) == 1
    assert result.iloc[0]["confidence_flag"] in ("low", "medium", "high")
    assert result.iloc[0]["reasoning"]  # non-empty reasoning

test("EC05 sparse asset (only id + name) → no crash, produces output", ec05)

# ── EC06: Asset with entirely empty description fields ────────────────────────
def ec06():
    df = make_asset_df([{
        "asset_id": "EMPTY", "asset_name": "",
        "asset_type": "", "description": "", "technical_specs": "",
        "location": "", "notes": "", "existing_classification": "",
    }])
    result = classify_all(df, SINGLE_CLS, top_n=1)
    assert len(result) == 1
    # Should flag as "low" confidence with an informative reasoning
    assert "Insufficient" in result.iloc[0]["reasoning"] or result.iloc[0]["confidence_flag"] in ("low", "medium")

test("EC06 fully empty asset text → graceful low-confidence output", ec06)

# ── EC07: top_n = 1 ───────────────────────────────────────────────────────────
def ec07():
    df = make_asset_df([{"asset_id": "T1", "asset_name": "Bridge",
                          "description": "concrete bridge deck"}])
    result = classify_all(df, SINGLE_CLS, top_n=1)
    assert len(result) == 1
    assert result.iloc[0]["match_rank"] == "1st"

test("EC07 top_n=1 → exactly 1 result", ec07)

# ── EC08: top_n > number of classifications (30 > 3) ─────────────────────────
def ec08():
    df = make_asset_df([{"asset_id": "T1", "asset_name": "Bridge",
                          "description": "concrete bridge deck"}])
    result = classify_all(df, SINGLE_CLS, top_n=30)
    # Should cap at 3 (size of classification table)
    assert len(result) == 3, f"Expected 3 (capped at table size), got {len(result)}"

test("EC08 top_n > classification table size → capped at table size", ec08)

# ── EC09: top_n > 3 → ordinal rank labels ────────────────────────────────────
def ec09():
    """Rank labels for 4+: must be proper ordinals, not '4th', '5th' (those are
    correct), but NOT '21th', '22th' etc."""
    big_cls = pd.DataFrame([
        {"classification_code": f"X{i:03d}",
         "classification_name": f"Class {i}",
         "classification_description": f"description for class {i}",
         "category": "General", "subcategory": "Sub"}
        for i in range(1, 25)   # 24 entries
    ])
    df = make_asset_df([{"asset_id": "T1", "asset_name": "Test asset",
                          "description": "test description"}])
    result = classify_all(df, {"BIG": big_cls}, top_n=22)
    ranks = result["match_rank"].tolist()
    # Spot-check problematic ordinals
    assert "21th" not in ranks, f"Found '21th' in ranks — should be '21st': {ranks}"
    assert "22th" not in ranks, f"Found '22th' in ranks — should be '22nd': {ranks}"

test("EC09 rank labels ordinal correctness (21st not 21th)", ec09)

# ── EC10: Single classification system → no cross-system block in summary ─────
def ec10():
    df = make_asset_df([{"asset_id": "T1", "asset_name": "Bridge",
                          "description": "bridge deck concrete"}])
    result = classify_all(df, SINGLE_CLS, top_n=1)
    summary = generate_summary(result, ["CLS"])
    assert "cross_system_consistency" not in summary, "Should not have cross-system block with 1 system"

test("EC10 single system → no cross-system block in summary", ec10)

# ── EC11: Special characters in descriptions ──────────────────────────────────
def ec11():
    df = make_asset_df([{
        "asset_id": "SPECIAL",
        "asset_name": "Brücke über den Fluss",   # German, umlaut
        "asset_type": "Bridge/Culvert & Structure",
        "description": 'Reinforced "concrete" bridge; width=12m, load≥50t — Class A',
        "technical_specs": "Span: 45m | RC Slab | Load: Class A",
    }])
    result = classify_all(df, SINGLE_CLS, top_n=1)
    assert len(result) == 1
    assert isinstance(result.iloc[0]["similarity_score"], float)

test("EC11 special characters (unicode, punctuation) → no crash", ec11)

# ── EC12: Classification table with single entry ──────────────────────────────
def ec12():
    """Calibration edge case: hi == lo → all scores → 50.0."""
    one_cls = pd.DataFrame([
        {"classification_code": "ONLY", "classification_name": "Only Option",
         "classification_description": "The only classification", "category": "X", "subcategory": ""}
    ])
    df = make_asset_df([{"asset_id": "T1", "asset_name": "Something",
                          "description": "something"}])
    result = classify_all(df, {"ONE": one_cls}, top_n=1)
    assert len(result) == 1
    assert result.iloc[0]["similarity_score"] == 50.0, (
        f"Expected 50.0 when single-entry table (hi==lo), got {result.iloc[0]['similarity_score']}"
    )

test("EC12 single-entry classification table → score = 50.0", ec12)

# ── EC13: Very long description (performance + no crash) ─────────────────────
def ec13():
    long_desc = " ".join(["bridge concrete reinforced structure deck span culvert"] * 150)
    df = make_asset_df([{"asset_id": "LONG", "asset_name": "Long asset",
                          "description": long_desc}])
    import time
    t0 = time.time()
    result = classify_all(df, SINGLE_CLS, top_n=3)
    elapsed = time.time() - t0
    assert len(result) == 3
    assert elapsed < 5.0, f"Too slow for single asset with long desc: {elapsed:.1f}s"

test("EC13 very long description → no crash, completes in <5s", ec13)

# ── EC14: CSV with Windows line endings / trailing whitespace in column names ─
def ec14():
    import tempfile, os
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

test("EC14 CSV with Windows CRLF + column name whitespace → loads correctly", ec14)

# ── EC15: _calibrate_scores with single-asset (boundary) ─────────────────────
def ec15():
    """Single asset: all scores identical for that system → map to 50.0."""
    arr = np.array([30.0, 30.0, 30.0])  # hi == lo
    cal = _calibrate_scores({"SYS": {"A1": arr}})
    assert np.allclose(cal["SYS"]["A1"], 50.0), f"Expected all 50.0, got {cal['SYS']['A1']}"

test("EC15 calibration with all-identical scores → 50.0", ec15)

# ── EC16: generate_summary with empty results_df ─────────────────────────────
def ec16():
    empty_df = pd.DataFrame(columns=[
        "asset_id", "asset_name", "classification_system",
        "matched_classification_code", "matched_classification_name",
        "similarity_score", "match_rank", "confidence_flag",
        "reasoning", "matched_category", "score_breakdown",
    ])
    summary = generate_summary(empty_df, ["SYS1", "SYS2"])
    assert summary["total_assets_processed"] == 0
    assert summary["manual_review_count"] == 0

test("EC16 generate_summary with empty results → no crash, zero counts", ec16)

# ── EC17: save_results with empty DataFrame ───────────────────────────────────
def ec17():
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

test("EC17 save_results with empty DataFrame → files created without crash", ec17)

# ── EC18: Multi-system summary consistency with mismatched row counts ─────────
def ec18():
    """If one system classifies fewer assets than others, summary should not crash."""
    rows = [
        {"asset_id": "X1", "classification_system": "S1", "match_rank": "1st",
         "similarity_score": 75.0, "confidence_flag": "high",
         "matched_category": "Structures", "matched_classification_code": "A001",
         "asset_name": "X"},
        {"asset_id": "X1", "classification_system": "S2", "match_rank": "1st",
         "similarity_score": 60.0, "confidence_flag": "medium",
         "matched_category": "Structures", "matched_classification_code": "A001",
         "asset_name": "X"},
    ]
    df = pd.DataFrame(rows)
    # S3 is listed but has no rows (no assets classified)
    summary = generate_summary(df, ["S1", "S2", "S3"])
    assert "S1" in summary["systems"]
    assert "S2" in summary["systems"]
    assert "S3" not in summary["systems"]  # should be skipped, not crash

test("EC18 summary with a system that has no classified assets → skipped cleanly", ec18)

# ── EC19: ML mode with duplicate asset_id ─────────────────────────────────────
def ec19():
    df = make_asset_df([
        {"asset_id": "DUP", "asset_name": "Main Street Bridge",
         "asset_type": "Bridge", "description": "reinforced concrete bridge deck superstructure"},
        {"asset_id": "DUP", "asset_name": "Hume Highway Pavement",
         "asset_type": "Pavement", "description": "flexible asphalt pavement layer rutting fatigue"},
    ])
    result = classify_all_ml(df, SINGLE_CLS, top_n=1)
    rank1 = result[result["match_rank"] == "1st"]["matched_classification_code"].tolist()
    assert len(rank1) == 2, f"Expected 2 rank-1 rows, got {len(rank1)}"
    assert rank1[0] != rank1[1], (
        f"ML mode duplicate asset_id corruption: both rows got code {rank1[0]}"
    )

test("EC19 ML mode duplicate asset_id → each row gets independent scores", ec19)

# ── EC20: ML mode empty asset register ───────────────────────────────────────
def ec20():
    empty = make_asset_df([])
    result = classify_all_ml(empty, SINGLE_CLS, top_n=3)
    assert len(result) == 0

test("EC20 ML mode empty asset register → empty DataFrame", ec20)

# ── EC21: Excel file loading ──────────────────────────────────────────────────
def ec21():
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    try:
        xlsx = os.path.join(tmpdir, "test.xlsx")
        pd.DataFrame([
            {"asset_id": "E001", "asset_name": "Excel Bridge",
             "asset_type": "Bridge", "description": "Excel-loaded bridge"}
        ]).to_excel(xlsx, index=False)
        df, _ = load_asset_register(xlsx)
        assert len(df) == 1
        assert df.iloc[0]["asset_id"] == "E001"
    finally:
        shutil.rmtree(tmpdir)

test("EC21 Excel (.xlsx) asset register → loads correctly", ec21)

# ── EC22: Unsupported file format error message ───────────────────────────────
def ec22():
    try:
        load_asset_register("/tmp/test.json")
        assert False, "Should have raised an error"
    except (ValueError, FileNotFoundError) as e:
        assert "json" in str(e).lower() or "not found" in str(e).lower()

test("EC22 unsupported file format → clear error message", ec22)

# ── EC23: Missing required columns in asset register ─────────────────────────
def ec23():
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("asset_name,description\n")
        f.write("Bridge,A concrete bridge\n")
        fname = f.name
    try:
        load_asset_register(fname)
        assert False, "Should have raised ValueError for missing asset_id"
    except ValueError as e:
        assert "asset_id" in str(e)
    finally:
        os.unlink(fname)

test("EC23 missing required column (asset_id) → clear ValueError", ec23)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
print(f"{'='*60}")
print(f"  Results: {passed} passed, {failed} failed  ({len(results)} total)")
print(f"{'='*60}")
if failed:
    print("\nFailed tests:")
    for s, name, reason in results:
        if s == FAIL:
            print(f"  ✗ {name}")
            print(f"    {reason}")
