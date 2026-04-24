"""
ML-mode classification pipeline.

classify_all_ml() is the entry point.  It reuses _calibrate_scores,
_confidence_flag, _reasoning, and _RANK_LABEL from classifier.py unchanged,
then layers in:
  - Failure-mode-enriched classification texts
  - Sentence-transformer Component D (embedding cosine similarity)
  - Two-stage recursive category filtering
  - Cross-register semantic equivalence index
  - Per-rank-1 failure alignment scoring
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .classifier import (
    _RANK_LABEL,
    _rank_label,
    _calibrate_scores,
    _confidence_flag,
    _reasoning,
)
from .similarity_engine import (
    SimilarityEngine,
    build_asset_text,
    build_classification_text,
)
from .ml_engine import (
    CrossRegisterIndex,
    EmbeddingEngine,
    FailureModeEngine,
    TwoStageClassifier,
    TwoStageResult,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reasoning_ml(
    base: str,
    two_stage: TwoStageResult,
    category_mismatch: bool,
    embedding_available: bool,
) -> str:
    parts = [base]
    if two_stage.stage1_available:
        cats = ", ".join(two_stage.top_categories[:2])
        parts.append(f"[Stage1 domain: {cats}]")
    if category_mismatch:
        parts.append("[WARNING: matched category outside Stage1 prediction — review]")
    if not embedding_available:
        parts.append("[Embed: unavailable, TF-IDF weights used]")
    return " ".join(parts)


def _classify_one_ml(
    asset_row: pd.Series,
    system_name: str,
    class_df: pd.DataFrame,
    calibrated: np.ndarray,
    a_arr: np.ndarray,
    b_arr: np.ndarray,
    c_arr: np.ndarray,
    d_arr: np.ndarray,
    two_stage: TwoStageResult,
    top_n: int,
    cross_reg: CrossRegisterIndex,
    failure_engine: FailureModeEngine,
    embedding_engine: EmbeddingEngine,
) -> List[Dict[str, Any]]:
    asset_text = build_asset_text(asset_row)
    asset_id   = str(asset_row.get("asset_id", ""))
    asset_name = str(asset_row.get("asset_name", ""))
    n = min(top_n, len(class_df))

    # --- Apply two-stage candidate mask ---
    calibrated_masked = calibrated.copy()
    if two_stage.stage1_available and two_stage.candidate_indices:
        mask = np.zeros(len(calibrated), dtype=bool)
        for idx in two_stage.candidate_indices:
            mask[idx] = True
        calibrated_masked[~mask] = -np.inf

    top_idx = np.argsort(calibrated_masked)[::-1][:n]

    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(top_idx, start=1):
        crow         = class_df.iloc[int(idx)]
        matched_code = str(crow["classification_code"])
        matched_cat  = str(crow.get("category", ""))
        cal_score    = float(calibrated[int(idx)])     # unmasked for display
        a, b, c, d   = (float(a_arr[int(idx)]), float(b_arr[int(idx)]),
                        float(c_arr[int(idx)]), float(d_arr[int(idx)]))

        # --- Base reasoning (reuse fast-mode logic) ---
        class_text = build_classification_text(crow)
        if not asset_text.strip():
            base_r = "Insufficient asset data for reliable matching"
        else:
            base_r = _reasoning(
                asset_text, str(crow["classification_name"]),
                class_text, matched_cat, cal_score, a, b, c,
            )

        # --- Category mismatch detection ---
        category_mismatch = (
            two_stage.stage1_available
            and matched_cat not in two_stage.top_categories
        )

        reasoning = _reasoning_ml(
            base_r, two_stage, category_mismatch, embedding_engine.available
        )

        # --- Rank-1-only ML fields ---
        if rank == 1:
            equiv_list  = cross_reg.lookup_equivalents(matched_code, system_name, top_k=2)
            cross_equiv = (
                " | ".join(f"{s}:{c} ({sc:.2f})" for s, c, sc in equiv_list)
                if equiv_list else ""
            )
            failure_domain = failure_engine.get_domain_for_code(matched_code) or ""
            if failure_domain:
                f_align = failure_engine.compute_alignment_score(
                    asset_text, failure_domain, embedding_engine
                )
                f_modes = failure_engine.top_failure_modes(failure_domain)
            else:
                f_align = ""
                f_modes = ""
        else:
            cross_equiv    = ""
            failure_domain = ""
            f_align        = ""
            f_modes        = ""

        # --- Stage-1 score string (top-3 categories by score, descending) ---
        if two_stage.stage1_available and two_stage.category_scores:
            sorted_cats = sorted(
                two_stage.category_scores.items(), key=lambda x: -x[1]
            )[:3]
            stage1_score_str = ", ".join(f"{k}:{v:.1f}" for k, v in sorted_cats)
        else:
            stage1_score_str = ""

        s2_match = (
            (matched_cat in two_stage.top_categories)
            if two_stage.stage1_available else ""
        )

        results.append({
            # --- 11 base columns (identical layout to fast mode) ---
            "asset_id":                    asset_id,
            "asset_name":                  asset_name,
            "classification_system":       system_name,
            "matched_classification_code": matched_code,
            "matched_classification_name": str(crow["classification_name"]),
            "similarity_score":            round(cal_score, 1),
            "match_rank":                  _rank_label(rank),
            "confidence_flag":             _confidence_flag(cal_score),
            "reasoning":                   reasoning,
            "matched_category":            matched_cat,
            "score_breakdown": (
                f"TF-IDF: {a:.1f} | Category: {b:.1f} | "
                f"Jaccard: {c:.1f} | Embed: {d:.1f}"
            ),
            # --- 10 ML-only columns ---
            "stage1_categories":         (
                ", ".join(two_stage.top_categories)
                if two_stage.stage1_available else ""
            ),
            "stage1_category_scores":    stage1_score_str,
            "stage2_category_match":     s2_match,
            "category_mismatch_flag":    category_mismatch if two_stage.stage1_available else "",
            "embedding_score":           round(d, 1),
            "cross_system_equivalents":  cross_equiv,
            "cross_system_category_vote": "",       # filled in post-processing
            "failure_domain":            failure_domain,
            "failure_alignment_score":   f_align,
            "failure_modes_expected":    f_modes,
        })

    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_all_ml(
    asset_df: pd.DataFrame,
    classification_tables: Dict[str, pd.DataFrame],
    top_n: int = 3,
    top_categories: int = 2,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    equiv_threshold: float = 0.65,
) -> pd.DataFrame:
    """Two-pass ML classification with embeddings, two-stage filtering,
    cross-register equivalence, and failure-mode alignment.

    Falls back gracefully to TF-IDF weights if sentence-transformers is absent.
    """
    if asset_df.empty:
        return pd.DataFrame()

    # ── A: Create shared engines ─────────────────────────────────────────────
    embedding_engine = EmbeddingEngine(embedding_model_name)
    failure_engine   = FailureModeEngine()

    total = len(asset_df)

    # ── B: Fit per-system engines ────────────────────────────────────────────
    print("  Fitting ML engines …")
    sim_engines:    Dict[str, SimilarityEngine]     = {}
    two_stage_clfs: Dict[str, TwoStageClassifier]   = {}
    system_matrices: Dict[str, np.ndarray]          = {}
    n_classes_per_system: Dict[str, int] = {
        s: len(cdf) for s, cdf in classification_tables.items()
    }

    for system, cdf in classification_tables.items():
        # Enrich classification texts with failure-mode vocabulary
        enriched_full: List[str] = []
        for _, row in cdf.iterrows():
            code   = str(row.get("classification_code", ""))
            domain = failure_engine.get_domain_for_code(code)
            base   = build_classification_text(row)
            enriched_full.append(failure_engine.enrich_classification_text(base, domain))

        cat_texts = [
            " ".join(filter(None, [
                str(row.get("classification_name", "")).strip(),
                str(row.get("category", "")).strip(),
                str(row.get("subcategory", "")).strip(),
            ]))
            for _, row in cdf.iterrows()
        ]

        # SimilarityEngine fitted on enriched texts (TF-IDF components A, B, C)
        eng = SimilarityEngine()
        eng.fit(enriched_full, category_texts=cat_texts)
        sim_engines[system] = eng

        # EmbeddingEngine fitted on enriched texts (component D)
        sys_matrix = embedding_engine.fit(enriched_full)
        system_matrices[system] = sys_matrix

        # TwoStageClassifier for this system
        ts = TwoStageClassifier(embedding_engine)
        ts.fit(cdf, failure_engine)
        two_stage_clfs[system] = ts

    # ── C: Build cross-register index ────────────────────────────────────────
    cross_reg = CrossRegisterIndex()
    cross_reg.build(classification_tables, system_matrices, threshold=equiv_threshold)

    # ── D: PASS 1 — collect raw scores ───────────────────────────────────────
    # raw_ml[system][row_idx] = (hybrid_ml, a_arr, b_arr, c_arr, d_arr)
    # Keyed by row index (not asset_id) so duplicate asset_ids don't collide.
    raw_ml: Dict[str, Dict[int, Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]]] = {s: {} for s in classification_tables}

    for i, (_, asset_row) in enumerate(asset_df.iterrows(), start=1):
        label = (asset_row.get("asset_name")
                 or asset_row.get("asset_id") or f"row {i}")
        print(f"  [{i}/{total}] {label}")
        asset_text = build_asset_text(asset_row)

        for system, eng in sim_engines.items():
            n_cls = n_classes_per_system[system]

            # Components A, B, C from enriched SimilarityEngine
            results_raw = eng.score_raw(asset_text, top_n=n_cls)
            hybrid_arr = np.zeros(n_cls)
            a_arr      = np.zeros(n_cls)
            b_arr      = np.zeros(n_cls)
            c_arr      = np.zeros(n_cls)
            for r_idx, h, ra, rb, rc in results_raw:
                hybrid_arr[r_idx] = h
                a_arr[r_idx]      = ra
                b_arr[r_idx]      = rb
                c_arr[r_idx]      = rc

            # Component D — embedding cosine
            if embedding_engine.available:
                asset_emb = embedding_engine.transform([asset_text])   # (1, D)
                d_arr = (asset_emb @ system_matrices[system].T).flatten() * 100
                hybrid_ml = (
                    0.35 * a_arr
                    + 0.15 * b_arr
                    + 0.15 * c_arr
                    + 0.35 * d_arr
                )
            else:
                d_arr     = np.zeros(n_cls)
                hybrid_ml = 0.50 * a_arr + 0.25 * b_arr + 0.25 * c_arr

            raw_ml[system][i] = (hybrid_ml, a_arr, b_arr, c_arr, d_arr)

    # ── E: Calibrate per system ───────────────────────────────────────────────
    raw_hybrid = {
        s: {i: arrs[0] for i, arrs in am.items()}
        for s, am in raw_ml.items()
    }
    cal_hybrid = _calibrate_scores(raw_hybrid)

    # ── F: PASS 2 — build output records ─────────────────────────────────────
    records: List[Dict[str, Any]] = []
    for i, (_, asset_row) in enumerate(asset_df.iterrows(), start=1):
        asset_text = build_asset_text(asset_row)

        for system, cdf in classification_tables.items():
            _, a_arr, b_arr, c_arr, d_arr = raw_ml[system][i]
            calibrated = cal_hybrid[system][i]

            two_stage = two_stage_clfs[system].score_two_stage(
                asset_text, top_categories=top_categories
            )

            records.extend(
                _classify_one_ml(
                    asset_row, system, cdf,
                    calibrated, a_arr, b_arr, c_arr, d_arr,
                    two_stage, top_n,
                    cross_reg, failure_engine, embedding_engine,
                )
            )

    # ── G: Post-process — cross-system category vote ─────────────────────────
    df = pd.DataFrame(records)
    if not df.empty:
        top1 = df[df["match_rank"] == "1st"]
        for asset_id_val, grp in top1.groupby("asset_id"):
            top1_codes = dict(zip(grp["classification_system"],
                                  grp["matched_classification_code"]))
            _, is_majority = cross_reg.majority_domain_vote(top1_codes, failure_engine)
            vote = "majority_agree" if is_majority else "no_consensus"
            df.loc[df["asset_id"] == asset_id_val, "cross_system_category_vote"] = vote

    return df
