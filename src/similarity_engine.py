import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Tuple

# Minimal stop-word list: only remove truly generic connective words so that
# technical domain terms (road, bridge, tunnel, pipe…) keep their IDF weight.
_STOP_WORDS = frozenset(
    "a an the and or but if in on at to for of with from by is are was were be been "
    "being have has had do does did will would could should may might shall not no "
    "this that these those it its also about as up out all any into over such than "
    "when which who what how where".split()
)

# Domain acronyms common in EAM asset descriptions that should be expanded
# to their full technical phrases before TF-IDF matching.  Expansion is
# applied to BOTH asset text and classification text for consistency.
_EXPANSIONS = {
    "cctv":  "closed circuit television camera surveillance",
    "ptz":   "pan tilt zoom camera",
    "vms":   "variable message sign display board",
    "scats": "traffic signal control adaptive system intersection",
    "bms":   "building management system control automation",
    "tmc":   "traffic management centre monitoring operations",
    "hps":   "high pressure sodium lamp luminaire street lighting",
    "dn":    "diameter nominal pipe bore",
    "rc":    "reinforced concrete structure",
    "ac":    "asphalt concrete bituminous pavement surfacing",
    "dgb":   "dense graded base granular pavement subbase",
    "led":   "light emitting diode luminaire energy efficient",
    "its":   "intelligent transport system technology",
    "scada": "supervisory control data acquisition automation",
    "rms":   "roads maritime services transport authority",
    "anpr":  "automatic number plate recognition camera",
    "vds":   "vehicle detection system loop radar",
    "atms":  "advanced traffic management system",
}

# Pre-compile acronym pattern for efficiency (whole-word matches only)
_EXPANSION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _EXPANSIONS) + r")\b"
)


def _expand_acronyms(text: str) -> str:
    """Replace each domain acronym token with itself plus its expansion."""
    def _replace(m: re.Match) -> str:
        token = m.group(0)
        return f"{token} {_EXPANSIONS[token]}"
    return _EXPANSION_PATTERN.sub(_replace, text)


def _clean(text: str) -> str:
    text = str(text).lower()
    text = _expand_acronyms(text)                   # O2: expand EAM acronyms
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _term_set(text: str) -> frozenset:
    """Return a frozenset of 3+-character non-stop-word tokens from cleaned text."""
    return frozenset(
        w for w in re.findall(r"\b[a-z]{3,}\b", _clean(text))
        if w not in _STOP_WORDS
    )


def build_asset_text(row: pd.Series) -> str:
    name = str(row.get("asset_name", "")).strip()
    atype = str(row.get("asset_type", "")).strip()
    desc = str(row.get("description", "")).strip()
    specs = str(row.get("technical_specs", "")).strip()
    location = str(row.get("location", "")).strip()
    existing_cls = str(row.get("existing_classification", "")).strip()
    notes = str(row.get("notes", "")).strip()
    mfr_model = str(row.get("manufacturer_model", "")).strip()

    parts: list[str] = []
    # asset_type is the cleanest domain signal (structured field, no location noise)
    # name gets ×2 not ×3 to reduce pollution from location tokens like "Main Street"
    if name:
        parts += [name] * 2
    if atype:
        parts += [atype] * 3
    if existing_cls:
        parts += [existing_cls] * 2   # legacy classification is a strong domain prior
    if desc:
        parts.append(desc)
    if specs:
        parts.append(specs)
    if notes:
        parts.append(notes)
    if mfr_model:
        parts.append(mfr_model)
    if location:
        parts.append(location)
    return " ".join(parts)


def build_classification_text(row: pd.Series) -> str:
    parts: list[str] = []
    for col in ["classification_name", "classification_description",
                "category", "subcategory", "keywords", "classification_code"]:
        val = str(row.get(col, "")).strip()
        if val:
            parts.append(val)
    return " ".join(parts)


def _build_category_text(row: pd.Series) -> str:
    """Build a short text string from only the category + subcategory fields."""
    parts = []
    for col in ["category", "subcategory"]:
        val = str(row.get(col, "")).strip()
        if val:
            parts.append(val)
    return " ".join(parts)


class SimilarityEngine:
    """Hybrid 3-component similarity engine.

    Score = 0.50 × TF-IDF(full_text)  [Component A — semantic overlap]
          + 0.25 × TF-IDF(category)   [Component B — domain hierarchy]
          + 0.25 × Jaccard(keywords)  [Component C — term precision]
    """

    def __init__(self) -> None:
        self._full_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
            min_df=1, stop_words=list(_STOP_WORDS),
        )
        self._cat_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
            min_df=1, stop_words=list(_STOP_WORDS),
        )
        self._full_matrix = None
        self._cat_matrix = None
        self._class_term_sets: list[frozenset] = []
        self._n_classes = 0

    def fit(
        self,
        classification_texts: List[str],
        category_texts: Optional[List[str]] = None,
    ) -> None:
        """Fit the hybrid engine on classification descriptions and categories.

        Parameters
        ----------
        classification_texts : full concatenated text per classification entry
        category_texts       : category+subcategory strings; if omitted, Component B
                               contributes zero and weights redistribute to A and C.
        """
        self._n_classes = len(classification_texts)
        cleaned_full = [_clean(t) for t in classification_texts]

        self._full_matrix = self._full_vec.fit_transform(cleaned_full)
        self._class_term_sets = [_term_set(t) for t in classification_texts]

        if category_texts and len(category_texts) == len(classification_texts):
            cleaned_cats = [_clean(t) for t in category_texts]
            # Guard against degenerate empty-corpus case
            non_empty = [t for t in cleaned_cats if t.strip()]
            if non_empty:
                self._cat_matrix = self._cat_vec.fit_transform(cleaned_cats)
            else:
                self._cat_matrix = None
        else:
            self._cat_matrix = None

    def score_raw(
        self, asset_text: str, top_n: int = 3
    ) -> List[Tuple[int, float, float, float, float]]:
        """Return uncalibrated (idx, hybrid, A, B, C) tuples for top-N matches.

        Scores are in 0–100 range before calibration.
        """
        n = min(top_n, self._n_classes)
        cleaned = _clean(asset_text)

        if not cleaned or self._full_matrix is None:
            return [(i, 0.0, 0.0, 0.0, 0.0) for i in range(n)]

        query_vec = self._full_vec.transform([cleaned])
        if query_vec.nnz == 0:
            return [(i, 0.0, 0.0, 0.0, 0.0) for i in range(n)]

        # Component A — full-text TF-IDF cosine
        raw_a = cosine_similarity(query_vec, self._full_matrix).flatten()
        a_scores = np.nan_to_num(raw_a, nan=0.0) * 100

        # Component B — category TF-IDF cosine
        if self._cat_matrix is not None:
            try:
                cat_vec = self._cat_vec.transform([cleaned])
                raw_b = cosine_similarity(cat_vec, self._cat_matrix).flatten()
                b_scores = np.nan_to_num(raw_b, nan=0.0) * 100
            except Exception:
                b_scores = np.zeros(self._n_classes)
        else:
            b_scores = np.zeros(self._n_classes)

        # Component C — Keyword Jaccard overlap
        asset_terms = _term_set(asset_text)
        c_scores = np.zeros(self._n_classes)
        if asset_terms:
            for i, cls_terms in enumerate(self._class_term_sets):
                union = asset_terms | cls_terms
                if union:
                    c_scores[i] = len(asset_terms & cls_terms) / len(union) * 100

        # Weighted hybrid
        if self._cat_matrix is not None:
            hybrid = 0.50 * a_scores + 0.25 * b_scores + 0.25 * c_scores
        else:
            # Redistribute category weight to A and C when category data absent
            hybrid = 0.625 * a_scores + 0.375 * c_scores

        hybrid = np.round(hybrid, 1)
        top_idx = np.argsort(hybrid)[::-1][:n]
        return [
            (int(i), float(hybrid[i]), float(a_scores[i]), float(b_scores[i]), float(c_scores[i]))
            for i in top_idx
        ]

    def score(self, asset_text: str, top_n: int = 3) -> List[Tuple[int, float]]:
        """Backward-compatible interface returning (idx, hybrid_score) tuples."""
        return [(idx, h) for idx, h, *_ in self.score_raw(asset_text, top_n)]
