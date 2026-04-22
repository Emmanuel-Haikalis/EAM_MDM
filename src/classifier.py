import re
import pandas as pd
from typing import List, Dict, Any

from .similarity_engine import SimilarityEngine, build_asset_text, build_classification_text, _clean

_STOP = {
    "the", "and", "for", "are", "with", "this", "that", "from", "used", "use",
    "have", "has", "its", "not", "but", "can", "all", "any", "may", "been",
    "also", "into", "over", "such", "than", "when", "these", "those", "which",
    "include", "including", "within", "along", "road", "system", "systems",
}

_RANK_LABEL = {1: "1st", 2: "2nd", 3: "3rd"}


def _confidence_flag(score: float) -> str:
    # TF-IDF cosine between short asset queries and classification descriptions
    # naturally tops out at ~65.  Thresholds are calibrated accordingly:
    #   high   >= 55  (strong term overlap, reliable match)
    #   medium >= 35  (partial overlap, generally correct but verify)
    #   low    <  35  (ambiguous or insufficient data, manual review needed)
    if score >= 55:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def _keywords(text: str, n: int = 8) -> list[str]:
    words = re.findall(r"\b[a-z]{3,}\b", _clean(text))
    seen: set[str] = set()
    result = []
    for w in words:
        if w not in _STOP and w not in seen:
            seen.add(w)
            result.append(w)
            if len(result) == n:
                break
    return result


def _reasoning(asset_text: str, class_name: str, class_text: str, score: float) -> str:
    # Thresholds mirror _confidence_flag (55 / 35)
    shared = set(_keywords(asset_text)) & set(_keywords(class_text))
    if score >= 55:
        if shared:
            terms = ", ".join(sorted(shared)[:3])
            return f"High confidence — shared key terms: {terms}"
        return f"High semantic similarity to '{class_name}'"
    if score >= 35:
        if shared:
            terms = ", ".join(sorted(shared)[:3])
            return f"Moderate match on terms: {terms}; manual review recommended"
        return f"Partial alignment with '{class_name}'; manual review recommended"
    return f"Low confidence match to '{class_name}'; manual review required"


def _classify_one(
    asset_row: pd.Series,
    system_name: str,
    class_df: pd.DataFrame,
    engine: SimilarityEngine,
    top_n: int,
) -> List[Dict[str, Any]]:
    asset_text = build_asset_text(asset_row)
    asset_id = str(asset_row.get("asset_id", ""))
    asset_name = str(asset_row.get("asset_name", ""))

    top = engine.score(asset_text, top_n=top_n)
    results = []

    for rank, (idx, score) in enumerate(top, start=1):
        crow = class_df.iloc[idx]
        class_text = build_classification_text(crow)
        reasoning = (
            "Insufficient asset data for reliable matching"
            if not asset_text.strip()
            else _reasoning(asset_text, str(crow["classification_name"]), class_text, score)
        )
        results.append(
            {
                "asset_id": asset_id,
                "asset_name": asset_name,
                "classification_system": system_name,
                "matched_classification_code": str(crow["classification_code"]),
                "matched_classification_name": str(crow["classification_name"]),
                "similarity_score": score,
                "match_rank": _RANK_LABEL.get(rank, f"{rank}th"),
                "confidence_flag": _confidence_flag(score),
                "reasoning": reasoning,
            }
        )

    return results


def classify_all(
    asset_df: pd.DataFrame,
    classification_tables: Dict[str, pd.DataFrame],
    top_n: int = 3,
) -> pd.DataFrame:
    engines: Dict[str, SimilarityEngine] = {}
    for system, cdf in classification_tables.items():
        eng = SimilarityEngine()
        eng.fit([build_classification_text(r) for _, r in cdf.iterrows()])
        engines[system] = eng

    records: List[Dict[str, Any]] = []
    total = len(asset_df)

    for i, (_, asset_row) in enumerate(asset_df.iterrows(), start=1):
        label = asset_row.get("asset_name") or asset_row.get("asset_id") or f"row {i}"
        print(f"  [{i}/{total}] {label}")
        for system, cdf in classification_tables.items():
            records.extend(_classify_one(asset_row, system, cdf, engines[system], top_n))

    return pd.DataFrame(records)
