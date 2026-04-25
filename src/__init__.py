"""
Asset Classification Library

Public API::

    from src import classify_assets, classify_assets_ml
    from src import load_asset_register, load_classification_table

    asset_df, quality = load_asset_register("assets.csv")
    class_tables = {
        "UNICLASS": load_classification_table("uniclass.csv", "UNICLASS"),
    }
    results = classify_assets(asset_df, class_tables)

Configure logging before calling any function::

    import logging
    logging.basicConfig(level=logging.INFO)
"""

from .classifier import classify_all as classify_assets
from .data_loader import (
    load_asset_register,
    load_classification_table,
    write_templates,
)
from .report_generator import save_results, generate_summary, save_summary

__all__ = [
    "classify_assets",
    "classify_assets_ml",
    "load_asset_register",
    "load_classification_table",
    "write_templates",
    "save_results",
    "generate_summary",
    "save_summary",
]


def classify_assets_ml(
    asset_df,
    classification_tables,
    top_n: int = 3,
    top_categories: int = 2,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    equiv_threshold: float = 0.65,
):
    """ML-mode classification (sentence-transformers required).

    Requires: ``pip install sentence-transformers``
    """
    from .classifier_ml import classify_all_ml
    return classify_all_ml(
        asset_df,
        classification_tables,
        top_n=top_n,
        top_categories=top_categories,
        embedding_model_name=embedding_model_name,
        equiv_threshold=equiv_threshold,
    )
