import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

# Minimal stop-word list: only removes truly generic connective words so that
# technical domain terms (road, bridge, tunnel, pipe…) keep their IDF weight.
_STOP_WORDS = frozenset(
    "a an the and or but if in on at to for of with from by is are was were be been "
    "being have has had do does did will would could should may might shall not no "
    "this that these those it its also about as up out all any into over such than "
    "when which who what how where".split()
)


def _clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_asset_text(row: pd.Series) -> str:
    name = str(row.get("asset_name", "")).strip()
    atype = str(row.get("asset_type", "")).strip()
    desc = str(row.get("description", "")).strip()
    specs = str(row.get("technical_specs", "")).strip()
    location = str(row.get("location", "")).strip()
    parts = []
    # Repeat name 3× and type 2× to amplify their discriminative TF-IDF weight
    if name:
        parts += [name] * 3
    if atype:
        parts += [atype] * 2
    if desc:
        parts.append(desc)
    if specs:
        parts.append(specs)
    if location:
        parts.append(location)
    return " ".join(parts)


def build_classification_text(row: pd.Series) -> str:
    parts = []
    for col in ["classification_name", "classification_description", "category", "subcategory", "classification_code"]:
        val = str(row.get(col, "")).strip()
        if val:
            parts.append(val)
    return " ".join(parts)


class SimilarityEngine:
    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            stop_words=list(_STOP_WORDS),
        )
        self._class_matrix = None
        self._n_classes = 0

    def fit(self, classification_texts: List[str]) -> None:
        self._n_classes = len(classification_texts)
        cleaned = [_clean(t) for t in classification_texts]
        self._class_matrix = self._vectorizer.fit_transform(cleaned)

    def score(self, asset_text: str, top_n: int = 3) -> List[Tuple[int, float]]:
        """Return (index, score_0_100) pairs for the top-N matching classifications."""
        n = min(top_n, self._n_classes)
        cleaned = _clean(asset_text)

        if not cleaned or self._class_matrix is None:
            return [(i, 0.0) for i in range(n)]

        query_vec = self._vectorizer.transform([cleaned])
        if query_vec.nnz == 0:
            return [(i, 0.0) for i in range(n)]

        raw = cosine_similarity(query_vec, self._class_matrix).flatten()
        scores = np.round(np.nan_to_num(raw, nan=0.0) * 100, 1)
        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_idx]
