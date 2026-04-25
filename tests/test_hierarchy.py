"""Unit tests for HierarchyIndex."""
import numpy as np
import pandas as pd
import pytest

from src.hierarchy import HierarchyIndex


@pytest.fixture
def hierarchy_cls_df():
    return pd.DataFrame({
        "classification_code": ["A", "A1", "A2", "B"],
        "classification_name": ["Structures", "Bridge", "Tunnel", "Pavement"],
        "classification_description": [
            "Load-bearing structures",
            "Bridge over waterway",
            "Underground tunnel",
            "Road pavement systems",
        ],
        "category": ["Civil", "Civil", "Civil", "Transport"],
        "subcategory": ["", "Bridges", "Tunnels", "Roads"],
        "parent_code": ["", "A", "A", ""],
        "hierarchy_level": [1, 2, 2, 1],
    })


class TestHierarchyIndexBuild:
    def test_inactive_without_parent_code_column(self):
        df = pd.DataFrame({
            "classification_code": ["A", "B"],
            "classification_name": ["Bridge", "Road"],
        })
        hi = HierarchyIndex()
        hi.build(df)
        assert not hi._active

    def test_active_with_parent_code_column(self, hierarchy_cls_df):
        hi = HierarchyIndex()
        hi.build(hierarchy_cls_df)
        assert hi._active

    def test_leaf_codes_identified(self, hierarchy_cls_df):
        hi = HierarchyIndex()
        hi.build(hierarchy_cls_df)
        assert hi.is_leaf("A1")
        assert hi.is_leaf("A2")
        assert not hi.is_leaf("A")   # A has children


class TestEnrichTexts:
    def test_child_text_includes_parent_vocab(self, hierarchy_cls_df):
        hi = HierarchyIndex()
        hi.build(hierarchy_cls_df)
        texts = hi.enrich_texts(hierarchy_cls_df)
        # texts[1] is "A1 - Bridge"; its enriched text should include parent "Structures"
        assert "Structures" in texts[1] or "structures" in texts[1]

    def test_root_text_unchanged(self, hierarchy_cls_df):
        hi = HierarchyIndex()
        hi.build(hierarchy_cls_df)
        texts = hi.enrich_texts(hierarchy_cls_df)
        # Root node "A" should not get extra text from non-existent parent
        from src.similarity_engine import build_classification_text
        base = build_classification_text(hierarchy_cls_df.iloc[0])
        assert texts[0] == base

    def test_no_op_without_parent_code(self):
        df = pd.DataFrame({
            "classification_code": ["X"],
            "classification_name": ["Something"],
            "classification_description": ["Some desc"],
            "category": ["Cat"],
            "subcategory": ["Sub"],
        })
        hi = HierarchyIndex()
        hi.build(df)
        texts = hi.enrich_texts(df)
        from src.similarity_engine import build_classification_text
        assert texts[0] == build_classification_text(df.iloc[0])


class TestLeafBoost:
    def test_no_boost_when_score_high(self, hierarchy_cls_df):
        hi = HierarchyIndex()
        hi.build(hierarchy_cls_df)
        calibrated = np.array([90.0, 80.0, 70.0, 60.0])
        score = hi.leaf_boost(80.0, "A1", calibrated, hierarchy_cls_df)
        assert score == 80.0  # already above threshold, no boost

    def test_boost_applied_when_score_low(self, hierarchy_cls_df):
        hi = HierarchyIndex()
        hi.build(hierarchy_cls_df)
        # A1 is a leaf (child of A at index 0). Set parent score high.
        calibrated = np.array([90.0, 30.0, 30.0, 30.0])
        score = hi.leaf_boost(30.0, "A1", calibrated, hierarchy_cls_df)
        # Expected: max(30.0, 90.0 × 0.90) = max(30.0, 81.0) = 81.0
        assert score == pytest.approx(81.0)

    def test_no_boost_for_non_leaf(self, hierarchy_cls_df):
        hi = HierarchyIndex()
        hi.build(hierarchy_cls_df)
        calibrated = np.array([90.0, 30.0, 30.0, 30.0])
        # "A" is not a leaf
        score = hi.leaf_boost(30.0, "A", calibrated, hierarchy_cls_df)
        assert score == 30.0

    def test_no_boost_when_inactive(self):
        df = pd.DataFrame({
            "classification_code": ["X"],
            "classification_name": ["Something"],
        })
        hi = HierarchyIndex()
        hi.build(df)
        calibrated = np.array([90.0])
        assert hi.leaf_boost(30.0, "X", calibrated, df) == 30.0
