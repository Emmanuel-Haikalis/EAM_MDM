import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

from .similarity_engine import SimilarityEngine, build_asset_text, build_classification_text, _clean, _STOP_WORDS

_RANK_LABEL = {1: "1st", 2: "2nd", 3: "3rd"}

# Confidence thresholds are calibrated for post-normalisation scores [5, 95]:
#   high   >= 70  (strong term + hierarchy overlap, reliable match)
#   medium >= 45  (partial overlap, generally correct but verify)
#   low    <  45  (ambiguous or insufficient data, manual review needed)
_HIGH_THRESH = 70
_MED_THRESH = 45


def _confidence_flag(score: float) -> str:
    if score >= _HIGH_THRESH:
        return "high"
    if score >= _MED_THRESH:
        return "medium"
    return "low"


def _keywords(text: str, n: int = 10) -> list[str]:
    """Extract up to n meaningful keywords (3+ chars, non-stop) from cleaned text."""
    words = re.findall(r"\b[a-z]{3,}\b", _clean(text))
    seen: set[str] = set()
    result = []
    for w in words:
        if w not in _STOP_WORDS and w not in seen:
            seen.add(w)
            result.append(w)
            if len(result) == n:
                break
    return result


def _dominant_component(a: float, b: float, c: float) -> str:
    """Return a label for the component that contributed most to the score."""
    best = max(("TF-IDF", a), ("Category", b), ("Jaccard", c), key=lambda t: t[1])
    return best[0]


def _reasoning(
    asset_text: str,
    class_name: str,
    class_text: str,
    matched_category: str,
    score: float,
    a: float,
    b: float,
    c: float,
) -> str:
    shared = set(_keywords(asset_text)) & set(_keywords(class_text))
    dominant = _dominant_component(a, b, c)

    if score >= _HIGH_THRESH:
        if dominant == "Category" and matched_category:
            base = f"Strong category alignment: '{matched_category}'"
        elif shared:
            terms = ", ".join(sorted(shared)[:3])
            base = f"High confidence — key terms: {terms}"
        else:
            base = f"High semantic similarity to '{class_name}'"
    elif score >= _MED_THRESH:
        if dominant == "Category" and matched_category:
            base = f"Category match: '{matched_category}'"
        elif shared:
            terms = ", ".join(sorted(shared)[:3])
            base = f"Moderate match on terms: {terms}; review recommended"
        else:
            base = f"Partial alignment with '{class_name}'; review recommended"
    else:
        base = f"Low confidence match to '{class_name}'; manual review required"

    return base


def _classify_one(
    asset_row: pd.Series,
    system_name: str,
    class_df: pd.DataFrame,
    calibrated: np.ndarray,   # shape (n_classes,) — post-calibration hybrid scores
    a_arr: np.ndarray,         # shape (n_classes,) — pre-calib Component A
    b_arr: np.ndarray,         # shape (n_classes,) — pre-calib Component B
    c_arr: np.ndarray,         # shape (n_classes,) — pre-calib Component C
    top_n: int,
) -> List[Dict[str, Any]]:
    asset_text = build_asset_text(asset_row)
    asset_id = str(asset_row.get("asset_id", ""))
    asset_name = str(asset_row.get("asset_name", ""))

    n = min(top_n, len(class_df))
    top_idx = np.argsort(calibrated)[::-1][:n]

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        crow = class_df.iloc[int(idx)]
        class_text = build_classification_text(crow)
        matched_category = str(crow.get("category", ""))
        cal_score = float(calibrated[idx])
        a, b, c = float(a_arr[idx]), float(b_arr[idx]), float(c_arr[idx])

        reasoning = (
            "Insufficient asset data for reliable matching"
            if not asset_text.strip()
            else _reasoning(asset_text, str(crow["classification_name"]),
                            class_text, matched_category, cal_score, a, b, c)
        )

        results.append({
            "asset_id": asset_id,
            "asset_name": asset_name,
            "classification_system": system_name,
            "matched_classification_code": str(crow["classification_code"]),
            "matched_classification_name": str(crow["classification_name"]),
            "similarity_score": round(cal_score, 1),
            "match_rank": _RANK_LABEL.get(rank, f"{rank}th"),
            "confidence_flag": _confidence_flag(cal_score),
            "reasoning": reasoning,
            "matched_category": matched_category,
            "score_breakdown": f"TF-IDF: {a:.1f} | Category: {b:.1f} | Jaccard: {c:.1f}",
        })

    return results


def _calibrate_scores(
    raw_scores: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Apply per-system min-max normalisation to [5, 95].

    raw_scores[system][asset_id] = hybrid_array(n_classes,)
    Returns calibrated copy with same structure.
    """
    calibrated: Dict[str, Dict[str, np.ndarray]] = {}
    for system, asset_map in raw_scores.items():
        all_vals = np.concatenate(list(asset_map.values()))
        lo, hi = float(all_vals.min()), float(all_vals.max())
        calibrated[system] = {}
        for asset_id, arr in asset_map.items():
            if hi > lo:
                cal = np.clip((arr - lo) / (hi - lo) * 90 + 5, 5.0, 95.0)
            else:
                cal = np.full_like(arr, 50.0)
            calibrated[system][asset_id] = np.round(cal, 1)
    return calibrated


def classify_all(
    asset_df: pd.DataFrame,
    classification_tables: Dict[str, pd.DataFrame],
    top_n: int = 3,
) -> pd.DataFrame:
    """Two-pass classification with per-system score calibration.

    Pass 1: Score every asset against every system, collect raw arrays.
    Calibrate: per-system min-max normalisation to [5, 95].
    Pass 2: Build output records from calibrated scores.
    """
    # Build engines (one per system) with full + category texts
    engines: Dict[str, SimilarityEngine] = {}
    for system, cdf in classification_tables.items():
        eng = SimilarityEngine()
        full_texts = [build_classification_text(r) for _, r in cdf.iterrows()]
        # Include classification_name in category text so the vocabulary
        # contains both singular and plural forms (e.g. "bridge" from
        # "Bridge Deck…" in addition to "bridges" from subcategory).
        cat_texts = [
            " ".join(filter(None, [
                str(r.get("classification_name", "")).strip(),
                str(r.get("category", "")).strip(),
                str(r.get("subcategory", "")).strip(),
            ]))
            for _, r in cdf.iterrows()
        ]
        eng.fit(full_texts, category_texts=cat_texts)
        engines[system] = eng

    total = len(asset_df)
    asset_ids = asset_df["asset_id"].tolist()

    # ---------- Pass 1: collect raw scores --------------------------------
    # raw[system][asset_id] = (hybrid_arr, a_arr, b_arr, c_arr)
    raw: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {
        s: {} for s in classification_tables
    }
    n_classes_per_system: Dict[str, int] = {
        s: len(cdf) for s, cdf in classification_tables.items()
    }

    for i, (_, asset_row) in enumerate(asset_df.iterrows(), start=1):
        label = asset_row.get("asset_name") or asset_row.get("asset_id") or f"row {i}"
        print(f"  [{i}/{total}] {label}")
        asset_text = build_asset_text(asset_row)
        aid = str(asset_row.get("asset_id", f"_row{i}"))

        for system, eng in engines.items():
            n_cls = n_classes_per_system[system]
            results = eng.score_raw(asset_text, top_n=n_cls)
            # Re-expand to full n_classes arrays (score_raw already returns all)
            hybrid_arr = np.zeros(n_cls)
            a_arr = np.zeros(n_cls)
            b_arr = np.zeros(n_cls)
            c_arr = np.zeros(n_cls)
            for idx, h, a, b, c in results:
                hybrid_arr[idx] = h
                a_arr[idx] = a
                b_arr[idx] = b
                c_arr[idx] = c
            raw[system][aid] = (hybrid_arr, a_arr, b_arr, c_arr)

    # ---------- Calibrate ------------------------------------------------
    raw_hybrid = {s: {aid: arrs[0] for aid, arrs in am.items()} for s, am in raw.items()}
    cal_hybrid = _calibrate_scores(raw_hybrid)

    # ---------- Pass 2: build records ------------------------------------
    records: List[Dict[str, Any]] = []
    for i, (_, asset_row) in enumerate(asset_df.iterrows(), start=1):
        aid = str(asset_row.get("asset_id", f"_row{i}"))
        for system, cdf in classification_tables.items():
            _, a_arr, b_arr, c_arr = raw[system][aid]
            calibrated = cal_hybrid[system][aid]
            records.extend(
                _classify_one(asset_row, system, cdf, calibrated, a_arr, b_arr, c_arr, top_n)
            )

    return pd.DataFrame(records)
