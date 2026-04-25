"""Unit tests for classifier internals."""
import numpy as np
import pandas as pd
import pytest

from src.classifier import (
    _calibrate_scores,
    _confidence_flag,
    _rank_label,
    _reasoning,
    classify_all,
)


class TestRankLabel:
    def test_first_three(self):
        assert _rank_label(1) == "1st"
        assert _rank_label(2) == "2nd"
        assert _rank_label(3) == "3rd"

    def test_fourth_to_tenth(self):
        assert _rank_label(4) == "4th"
        assert _rank_label(10) == "10th"

    def test_teens_use_th(self):
        # 11–13 must always use "th"
        assert _rank_label(11) == "11th"
        assert _rank_label(12) == "12th"
        assert _rank_label(13) == "13th"

    def test_twenty_first(self):
        assert _rank_label(21) == "21st"
        assert _rank_label(22) == "22nd"
        assert _rank_label(23) == "23rd"
        assert _rank_label(24) == "24th"


class TestConfidenceFlag:
    def test_high(self):
        assert _confidence_flag(70) == "high"
        assert _confidence_flag(95) == "high"

    def test_medium(self):
        assert _confidence_flag(45) == "medium"
        assert _confidence_flag(69) == "medium"

    def test_low(self):
        assert _confidence_flag(44) == "low"
        assert _confidence_flag(5) == "low"


class TestCalibrateScores:
    def test_range_maps_to_5_95(self):
        arr = np.array([0.0, 50.0, 100.0])
        cal = _calibrate_scores({"S": {"a1": arr}})
        out = cal["S"]["a1"]
        assert pytest.approx(out[0], abs=0.1) == 5.0
        assert pytest.approx(out[2], abs=0.1) == 95.0

    def test_identical_scores_map_to_50(self):
        arr = np.array([30.0, 30.0, 30.0])
        cal = _calibrate_scores({"S": {"a1": arr}})
        assert np.allclose(cal["S"]["a1"], 50.0)

    def test_multiple_assets_normalised_together(self):
        """All assets in the same system share the same min/max."""
        a1 = np.array([0.0, 0.0])
        a2 = np.array([100.0, 100.0])
        cal = _calibrate_scores({"S": {"a1": a1, "a2": a2}})
        assert np.allclose(cal["S"]["a1"], 5.0)
        assert np.allclose(cal["S"]["a2"], 95.0)


class TestReasoning:
    def test_high_score_includes_high_confidence_text(self):
        text = _reasoning("bridge concrete", "Bridge", "bridge structure", "Structures", 80, 60, 10, 10)
        assert "high" in text.lower() or "strong" in text.lower() or "confidence" in text.lower()

    def test_low_score_mentions_manual_review(self):
        text = _reasoning("bridge", "Pavement", "pavement road", "Pavements", 20, 5, 5, 5)
        assert "review" in text.lower()

    def test_score_gap_included_for_rank1(self):
        text = _reasoning("bridge concrete", "Bridge", "bridge", "Structures", 80, 60, 10, 10, score_2nd=60.0)
        assert "gap" in text.lower() or "+20" in text or "20" in text


class TestClassifyAll:
    def test_output_columns_present(self, single_cls, bridge_asset_df):
        result = classify_all(bridge_asset_df, single_cls, top_n=1)
        for col in [
            "asset_id", "asset_name", "classification_system",
            "matched_classification_code", "matched_classification_name",
            "similarity_score", "match_rank", "confidence_flag",
            "reasoning", "matched_category", "score_breakdown",
            "composite_confidence",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_scores_within_bounds(self, single_cls, bridge_asset_df):
        result = classify_all(bridge_asset_df, single_cls, top_n=3)
        assert result["similarity_score"].between(5, 95).all()

    def test_multi_system_produces_correct_row_count(self, multi_cls):
        df = pd.DataFrame([{"asset_id": "X1", "asset_name": "Bridge",
                             "description": "concrete bridge"}])
        result = classify_all(df, multi_cls, top_n=2)
        # 1 asset × 2 systems × 2 ranks = 4 rows
        assert len(result) == 4
