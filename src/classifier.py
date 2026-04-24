import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .similarity_engine import SimilarityEngine, build_asset_text, build_classification_text, _clean, _STOP_WORDS
from .hierarchy import HierarchyIndex

_RANK_LABEL = {1: "1st", 2: "2nd", 3: "3rd"}


def _rank_label(n: int) -> str:
    """Return the correct ordinal string for rank n (1st, 2nd, 21st, 22nd …)."""
    if n in _RANK_LABEL:
        return _RANK_LABEL[n]
    # Handles 11th/12th/13th correctly (not 11st/12nd/13rd)
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{('th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th')[n % 10]}"


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
    score_2nd: Optional[float] = None,
) -> str:
    """Generate human-readable match reasoning including score gap to 2nd candidate."""
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

    if score_2nd is not None:
        gap = round(score - score_2nd, 1)
        base = f"{base}. Score gap to 2nd: +{gap:.0f} pts"

    return base


def _classify_one(
    asset_row: pd.Series,
    system_name: str,
    class_df: pd.DataFrame,
    calibrated: np.ndarray,
    a_arr: np.ndarray,
    b_arr: np.ndarray,
    c_arr: np.ndarray,
    top_n: int,
    hierarchy: Optional[HierarchyIndex] = None,
) -> List[Dict[str, Any]]:
    asset_text = build_asset_text(asset_row)
    asset_id = str(asset_row.get("asset_id", ""))
    asset_name = str(asset_row.get("asset_name", ""))

    n = min(top_n, len(class_df))
    top_idx = np.argsort(calibrated)[::-1][:n]

    score_2nd_val = float(calibrated[top_idx[1]]) if n >= 2 else None

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        crow = class_df.iloc[int(idx)]
        class_text = build_classification_text(crow)
        matched_category = str(crow.get("category", ""))
        matched_code = str(crow["classification_code"])
        cal_score = float(calibrated[idx])
        a, b, c = float(a_arr[idx]), float(b_arr[idx]), float(c_arr[idx])

        # Leaf score boost (hierarchy-aware, no-op when hierarchy absent)
        if hierarchy is not None and rank == 1:
            cal_score = hierarchy.leaf_boost(cal_score, matched_code, calibrated, class_df)

        reasoning = (
            "Insufficient asset data for reliable matching"
            if not asset_text.strip()
            else _reasoning(
                asset_text, str(crow["classification_name"]),
                class_text, matched_category, cal_score, a, b, c,
                score_2nd=score_2nd_val if rank == 1 else None,
            )
        )

        results.append({
            "asset_id": asset_id,
            "asset_name": asset_name,
            "classification_system": system_name,
            "matched_classification_code": matched_code,
            "matched_classification_name": str(crow["classification_name"]),
            "similarity_score": round(cal_score, 1),
            "match_rank": _rank_label(rank),
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


def _compute_composite_confidence(
    df: pd.DataFrame,
    n_systems: int,
    code_to_domain: Optional[Dict[str, str]] = None,
) -> pd.Series:
    """Compute multi-signal composite confidence for every row.

    composite = 0.60 × similarity_score
              + 0.25 × cross_system_agreement_bonus   (100 agree / 50 single / 0 diverge)
              + 0.15 × failure_alignment_score         (0 in fast mode)
    """
    agreement: Dict[str, float] = {}
    if n_systems > 1 and code_to_domain:
        top1 = df[df["match_rank"] == "1st"]
        for asset_id, grp in top1.groupby("asset_id"):
            domains = [
                code_to_domain.get(str(row["matched_classification_code"]),
                                   str(row.get("matched_category", "")))
                for _, row in grp.iterrows()
            ]
            domains = [d for d in domains if d]
            if not domains:
                agreement[str(asset_id)] = 50.0
                continue
            most_common_count = Counter(domains).most_common(1)[0][1]
            agreement[str(asset_id)] = 100.0 if most_common_count > n_systems / 2 else 0.0
    else:
        for asset_id in df["asset_id"].unique():
            agreement[str(asset_id)] = 50.0

    f_align = pd.to_numeric(
        df.get("failure_alignment_score", pd.Series(0.0, index=df.index)),
        errors="coerce",
    ).fillna(0.0)

    bonus = df["asset_id"].astype(str).map(agreement).fillna(50.0)
    return (0.60 * df["similarity_score"] + 0.25 * bonus + 0.15 * f_align).round(1)


def classify_all(
    asset_df: pd.DataFrame,
    classification_tables: Dict[str, pd.DataFrame],
    top_n: int = 3,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Two-pass classification with per-system score calibration.

    Pass 1 (batch): Score ALL assets against ALL systems in vectorized matrix ops.
    Calibrate: per-system min-max normalisation to [5, 95].
    Pass 2: Build output records from calibrated scores.
    """
    if asset_df.empty:
        return pd.DataFrame()

    try:
        from .ml_engine import _CODE_TO_DOMAIN as code_to_domain
    except ImportError:
        code_to_domain = {}

    from .cache import FitCache
    cache = FitCache(cache_dir)

    # Build engines and hierarchy indices per system
    engines: Dict[str, SimilarityEngine] = {}
    hierarchy_indices: Dict[str, HierarchyIndex] = {}

    for system, cdf in classification_tables.items():
        hi = HierarchyIndex()
        hi.build(cdf)
        hierarchy_indices[system] = hi

        cached_eng = cache.load_tfidf(system, cdf)
        if cached_eng is not None:
            engines[system] = cached_eng
            continue

        full_texts = hi.enrich_texts(cdf)
        cat_texts = [
            " ".join(filter(None, [
                str(r.get("classification_name", "")).strip(),
                str(r.get("category", "")).strip(),
                str(r.get("subcategory", "")).strip(),
            ]))
            for _, r in cdf.iterrows()
        ]
        eng = SimilarityEngine()
        eng.fit(full_texts, category_texts=cat_texts)
        cache.save_tfidf(system, cdf, eng)
        engines[system] = eng

    total = len(asset_df)
    all_texts = [build_asset_text(row) for _, row in asset_df.iterrows()]

    # ---------- Pass 1: batch-score all assets (one matrix multiply per system) --
    # raw[system][i] = (hybrid_arr, a_arr, b_arr, c_arr)  keyed by 1-based row index
    raw: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {
        s: {} for s in classification_tables
    }

    for system, eng in engines.items():
        hybrid_mat, a_mat, b_mat, c_mat = eng.score_batch(all_texts)
        for i in range(total):
            raw[system][i + 1] = (hybrid_mat[i], a_mat[i], b_mat[i], c_mat[i])

    for i, (_, asset_row) in enumerate(asset_df.iterrows(), start=1):
        label = asset_row.get("asset_name") or asset_row.get("asset_id") or f"row {i}"
        print(f"  [{i}/{total}] {label}")

    # ---------- Calibrate ------------------------------------------------
    raw_hybrid = {s: {i: arrs[0] for i, arrs in am.items()} for s, am in raw.items()}
    cal_hybrid = _calibrate_scores(raw_hybrid)

    # ---------- Pass 2: build records ------------------------------------
    records: List[Dict[str, Any]] = []
    for i, (_, asset_row) in enumerate(asset_df.iterrows(), start=1):
        for system, cdf in classification_tables.items():
            _, a_arr, b_arr, c_arr = raw[system][i]
            calibrated = cal_hybrid[system][i]
            hi = hierarchy_indices[system]
            records.extend(
                _classify_one(asset_row, system, cdf, calibrated, a_arr, b_arr, c_arr, top_n, hi)
            )

    df = pd.DataFrame(records)

    if not df.empty:
        df["composite_confidence"] = _compute_composite_confidence(
            df, n_systems=len(classification_tables), code_to_domain=code_to_domain
        )

    return df
