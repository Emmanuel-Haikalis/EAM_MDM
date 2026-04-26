import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Tuple

from .config_loader import load_config

# Minimal stop-word list: only remove truly generic connective words so that
# technical domain terms (road, bridge, tunnel, pipe…) keep their IDF weight.
_STOP_WORDS = frozenset(
    "a an the and or but if in on at to for of with from by is are was were be been "
    "being have has had do does did will would could should may might shall not no "
    "this that these those it its also about as up out all any into over such than "
    "when which who what how where".split()
)

# Domain acronyms loaded from config/acronyms.yaml.
# Falls back to hardcoded dict if config is absent (e.g. during unit tests).
_EXPANSIONS: dict = load_config("acronyms") or {
    "cctv":  "closed circuit television camera surveillance",
    "ptz":   "pan tilt zoom camera",
    "vms":   "variable message sign display board",
    "scats": "traffic signal control adaptive system intersection",
    "led":   "light emitting diode luminaire energy efficient",
    "its":   "intelligent transport system technology",
    "scada": "supervisory control data acquisition automation",
    "anpr":  "automatic number plate recognition camera",
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
    text = _expand_acronyms(text)
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
    if name:
        parts += [name] * 2
    if atype:
        parts += [atype] * 3
    if existing_cls:
        parts += [existing_cls] * 2
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

    Score = w_A × TF-IDF(full_text)  [Component A — semantic overlap]
          + w_B × TF-IDF(category)   [Component B — domain hierarchy]
          + w_C × Jaccard(keywords)  [Component C — term precision]

    Default weights (fast mode): A=0.50, B=0.25, C=0.25.
    Per-column domain-adaptive weights can be set via set_domain_weights().
    score_batch() processes N assets in one vectorized pass.
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
        # Precomputed binary term matrix for vectorized Jaccard (set at fit time)
        self._jaccard_vocab: list[str] = []
        self._jaccard_matrix: Optional[np.ndarray] = None  # (M, V_j) bool
        # Per-column domain-adaptive weights (set by set_domain_weights; None = use defaults)
        self._w_a: Optional[np.ndarray] = None
        self._w_b: Optional[np.ndarray] = None
        self._w_c: Optional[np.ndarray] = None
        self._w_d: Optional[np.ndarray] = None

    def fit(
        self,
        classification_texts: List[str],
        category_texts: Optional[List[str]] = None,
    ) -> None:
        """Fit the hybrid engine on classification descriptions and categories."""
        self._n_classes = len(classification_texts)
        cleaned_full = [_clean(t) for t in classification_texts]

        self._full_matrix = self._full_vec.fit_transform(cleaned_full)
        self._class_term_sets = [_term_set(t) for t in classification_texts]

        # Precompute Jaccard binary matrix for batch scoring
        all_terms = sorted(set().union(*self._class_term_sets))
        self._jaccard_vocab = all_terms
        if all_terms:
            vocab_idx = {w: j for j, w in enumerate(all_terms)}
            jmat = np.zeros((self._n_classes, len(all_terms)), dtype=np.float32)
            for i, term_set in enumerate(self._class_term_sets):
                for w in term_set:
                    j = vocab_idx.get(w)
                    if j is not None:
                        jmat[i, j] = 1.0
            self._jaccard_matrix = jmat
            self._jaccard_vocab_idx = vocab_idx
        else:
            self._jaccard_matrix = None
            self._jaccard_vocab_idx = {}

        if category_texts and len(category_texts) == len(classification_texts):
            cleaned_cats = [_clean(t) for t in category_texts]
            non_empty = [t for t in cleaned_cats if t.strip()]
            if non_empty:
                self._cat_matrix = self._cat_vec.fit_transform(cleaned_cats)
            else:
                self._cat_matrix = None
        else:
            self._cat_matrix = None

    def _compute_jaccard_batch(self, asset_term_sets: List[frozenset]) -> np.ndarray:
        """Compute Jaccard scores for N assets vs M classifications.

        Returns (N, M) float32 array in [0, 100].
        """
        N = len(asset_term_sets)
        M = self._n_classes
        if self._jaccard_matrix is None or M == 0:
            return np.zeros((N, M), dtype=np.float32)

        V = len(self._jaccard_vocab)
        # Build (N, V) asset binary matrix
        asset_mat = np.zeros((N, V), dtype=np.float32)
        for i, terms in enumerate(asset_term_sets):
            for w in terms:
                j = self._jaccard_vocab_idx.get(w)
                if j is not None:
                    asset_mat[i, j] = 1.0

        # Intersection = (N, V) @ (V, M) = (N, M)
        intersection = asset_mat @ self._jaccard_matrix.T  # (N, M)
        asset_counts = asset_mat.sum(axis=1, keepdims=True)          # (N, 1)
        class_counts = self._jaccard_matrix.sum(axis=1, keepdims=False)  # (M,)
        union = asset_counts + class_counts - intersection            # (N, M)
        c_mat = np.where(union > 0, intersection / union * 100, 0.0)
        return c_mat.astype(np.float32)

    def score_batch(
        self, asset_texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Score N assets against all M classifications in one vectorized pass.

        Returns (hybrid, A, B, C) as (N, M) float32 arrays in [0, 100].
        Use this instead of calling score_raw() in a loop for large registers.
        """
        N = len(asset_texts)
        M = self._n_classes

        if M == 0 or self._full_matrix is None:
            z = np.zeros((N, M), dtype=np.float32)
            return z, z, z, z

        cleaned = [_clean(t) for t in asset_texts]

        # Component A — full-text TF-IDF cosine (N, M)
        query_mat = self._full_vec.transform(cleaned)       # (N, V)
        raw_a = cosine_similarity(query_mat, self._full_matrix)  # (N, M)
        a_mat = np.nan_to_num(raw_a, nan=0.0).astype(np.float32) * 100

        # Component B — category TF-IDF cosine (N, M)
        if self._cat_matrix is not None:
            try:
                cat_query = self._cat_vec.transform(cleaned)
                raw_b = cosine_similarity(cat_query, self._cat_matrix)
                b_mat = np.nan_to_num(raw_b, nan=0.0).astype(np.float32) * 100
            except Exception:
                b_mat = np.zeros((N, M), dtype=np.float32)
        else:
            b_mat = np.zeros((N, M), dtype=np.float32)

        # Component C — vectorized Jaccard (N, M)
        asset_term_sets = [_term_set(t) for t in asset_texts]
        c_mat = self._compute_jaccard_batch(asset_term_sets)

        # Weighted hybrid — use per-column domain weights when set, else defaults
        if self._w_a is not None:
            hybrid = a_mat * self._w_a + b_mat * self._w_b + c_mat * self._w_c
        elif self._cat_matrix is not None:
            hybrid = 0.50 * a_mat + 0.25 * b_mat + 0.25 * c_mat
        else:
            hybrid = 0.625 * a_mat + 0.375 * c_mat

        return np.round(hybrid, 1), a_mat, b_mat, c_mat

    def compute_ml_hybrid(
        self,
        a_arr: np.ndarray,
        b_arr: np.ndarray,
        c_arr: np.ndarray,
        d_arr: np.ndarray,
    ) -> np.ndarray:
        """Combine A/B/C/D component arrays using ML-mode domain weights.

        Uses per-column domain weights when set via set_domain_weights(mode='ml'),
        otherwise falls back to the standard ML defaults (A=0.35, B=0.15, C=0.15, D=0.35).
        """
        if self._w_a is not None and self._w_d is not None:
            return a_arr * self._w_a + b_arr * self._w_b + c_arr * self._w_c + d_arr * self._w_d
        return 0.35 * a_arr + 0.15 * b_arr + 0.15 * c_arr + 0.35 * d_arr

    def compute_ml_hybrid_no_embed(
        self,
        a_arr: np.ndarray,
        b_arr: np.ndarray,
        c_arr: np.ndarray,
    ) -> np.ndarray:
        """ML-mode fallback when embeddings are unavailable (D=0)."""
        if self._w_a is not None:
            # Redistribute D weight proportionally to A when embeddings absent
            total = self._w_a + self._w_b + self._w_c
            safe_total = np.where(total > 0, total, 1.0)
            return (a_arr * self._w_a + b_arr * self._w_b + c_arr * self._w_c) / safe_total * 100 / 100
        return 0.50 * a_arr + 0.25 * b_arr + 0.25 * c_arr

    def set_domain_weights(
        self,
        classification_codes: List[str],
        code_to_domain: dict,
        domain_weights_cfg: dict,
        mode: str = "fast",
    ) -> None:
        """Pre-compute per-column weight vectors from config/domain_weights.yaml.

        Each of the M classification columns gets its own A/B/C(/D) weight based
        on the failure domain of that code, enabling domain-adaptive scoring.
        Codes not in code_to_domain, or domains not in the overrides block, use
        the global defaults from the YAML defaults section.

        Call after fit() and before score_batch() / compute_ml_hybrid().
        """
        defaults_cfg = domain_weights_cfg.get("defaults", {}).get(mode, {})
        if mode == "ml":
            w_a_def = float(defaults_cfg.get("A", 0.35))
            w_b_def = float(defaults_cfg.get("B", 0.15))
            w_c_def = float(defaults_cfg.get("C", 0.15))
            w_d_def = float(defaults_cfg.get("D", 0.35))
        else:
            w_a_def = float(defaults_cfg.get("A", 0.50))
            w_b_def = float(defaults_cfg.get("B", 0.25))
            w_c_def = float(defaults_cfg.get("C", 0.25))
            w_d_def = 0.0

        overrides = domain_weights_cfg.get("overrides", {})
        M = self._n_classes
        w_a = np.full(M, w_a_def, dtype=np.float32)
        w_b = np.full(M, w_b_def, dtype=np.float32)
        w_c = np.full(M, w_c_def, dtype=np.float32)
        w_d = np.full(M, w_d_def, dtype=np.float32)

        for j, code in enumerate(classification_codes[:M]):
            domain = code_to_domain.get(str(code))
            if not domain:
                continue
            ov = overrides.get(domain, {}).get(mode, {})
            if not ov:
                continue
            w_a[j] = float(ov.get("A", w_a_def))
            w_b[j] = float(ov.get("B", w_b_def))
            w_c[j] = float(ov.get("C", w_c_def))
            w_d[j] = float(ov.get("D", w_d_def))

        self._w_a = w_a
        self._w_b = w_b
        self._w_c = w_c
        self._w_d = w_d
