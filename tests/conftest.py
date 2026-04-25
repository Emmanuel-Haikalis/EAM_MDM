"""Shared fixtures for the asset classification test suite."""
import pandas as pd
import pytest


@pytest.fixture
def cls_df():
    """3-entry classification table — small enough to expose edge cases fast."""
    return pd.DataFrame([
        {
            "classification_code": "A001",
            "classification_name": "Bridge Structures",
            "classification_description": "Reinforced concrete bridges and culverts",
            "category": "Structures",
            "subcategory": "Bridges",
        },
        {
            "classification_code": "A002",
            "classification_name": "Road Pavement Systems",
            "classification_description": "Flexible and rigid pavement layers",
            "category": "Pavements",
            "subcategory": "Roads",
        },
        {
            "classification_code": "A003",
            "classification_name": "Drainage Systems",
            "classification_description": "Stormwater pipes culverts and channels",
            "category": "Drainage",
            "subcategory": "Stormwater",
        },
    ])


@pytest.fixture
def single_cls(cls_df):
    return {"CLS": cls_df}


@pytest.fixture
def multi_cls(cls_df):
    return {"SYS1": cls_df.copy(), "SYS2": cls_df.copy()}


@pytest.fixture
def bridge_asset_df():
    return pd.DataFrame([{
        "asset_id": "A1",
        "asset_name": "Bridge",
        "asset_type": "Bridge Structure",
        "description": "Concrete bridge over creek",
    }])
