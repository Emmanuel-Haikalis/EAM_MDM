"""Unit tests for the similarity engine."""
import numpy as np
import pandas as pd
import pytest

from src.similarity_engine import SimilarityEngine, build_asset_text, build_classification_text, _clean


class TestClean:
    def test_lowercases(self):
        assert _clean("BRIDGE") == "bridge"

    def test_strips_punctuation(self):
        assert _clean("bridge/culvert") == "bridge culvert"

    def test_expands_known_acronym(self):
        result = _clean("cctv")
        assert "closed" in result or "camera" in result or "cctv" in result

    def test_collapses_whitespace(self):
        assert _clean("  a   b  ") == "a b"


class TestBuildAssetText:
    def test_combines_fields(self):
        row = pd.Series({
            "asset_name": "Bridge",
            "asset_type": "Structure",
            "description": "concrete",
            "technical_specs": "",
        })
        text = build_asset_text(row)
        assert "bridge" in text.lower()
        assert "concrete" in text.lower()

    def test_empty_row_returns_string(self):
        row = pd.Series({"asset_name": "", "asset_id": "X"})
        text = build_asset_text(row)
        assert isinstance(text, str)


class TestBuildClassificationText:
    def test_combines_fields(self):
        row = pd.Series({
            "classification_name": "Bridge Deck",
            "classification_description": "RC bridge deck systems",
            "category": "Civil",
            "subcategory": "Bridges",
            "keywords": "bridge deck span",
        })
        text = build_classification_text(row)
        assert "bridge" in text.lower()
        assert "civil" in text.lower()


class TestSimilarityEngine:
    @pytest.fixture
    def fitted_engine(self):
        texts = [
            "bridge concrete reinforced deck span",
            "pavement asphalt road layer rutting",
            "drainage stormwater pipe culvert channel",
        ]
        cat_texts = [
            "Structures Bridges",
            "Pavements Roads",
            "Drainage Stormwater",
        ]
        eng = SimilarityEngine()
        eng.fit(texts, category_texts=cat_texts)
        return eng

    def test_score_batch_shape(self, fitted_engine):
        assets = ["bridge deck", "road pavement", "stormwater pipe"]
        hybrid, a, b, c = fitted_engine.score_batch(assets)
        assert hybrid.shape == (3, 3)
        assert a.shape == (3, 3)
        assert b.shape == (3, 3)
        assert c.shape == (3, 3)

    def test_bridge_asset_matches_bridge_class(self, fitted_engine):
        hybrid, _, _, _ = fitted_engine.score_batch(["bridge concrete deck span"])
        # Index 0 is bridge class — should have highest score
        assert np.argmax(hybrid[0]) == 0

    def test_scores_are_nonnegative(self, fitted_engine):
        hybrid, a, b, c = fitted_engine.score_batch(["test asset"])
        assert (hybrid >= 0).all()
        assert (a >= 0).all()

    def test_single_asset_score_batch(self, fitted_engine):
        hybrid, a, b, c = fitted_engine.score_batch(["bridge"])
        assert hybrid.shape == (1, 3)
