"""Disk cache for fitted TF-IDF vectorisers and embedding matrices.

Cache files are keyed by a sha256 hash of the classification table content,
so any change to the table automatically invalidates the cache.

Usage::

    cache = FitCache(cache_dir)
    fitted = cache.load_tfidf(system_name, class_df)
    if fitted is None:
        # fit from scratch
        eng = SimilarityEngine(); eng.fit(texts)
        cache.save_tfidf(system_name, class_df, eng)
    else:
        eng = fitted
"""

from __future__ import annotations

import hashlib
import pathlib
import pickle
from typing import Optional

import numpy as np


def _table_hash(class_df) -> str:
    """SHA-256 of the CSV representation of a classification table."""
    csv_bytes = class_df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()[:16]


class FitCache:
    """Read/write disk cache for fitted SimilarityEngine and embedding matrices."""

    def __init__(self, cache_dir: Optional[str]) -> None:
        self._dir: Optional[pathlib.Path] = (
            pathlib.Path(cache_dir) if cache_dir else None
        )
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def active(self) -> bool:
        return self._dir is not None

    def _tfidf_path(self, system: str, h: str) -> pathlib.Path:
        return self._dir / f"{system}_{h}_tfidf.pkl"

    def _embed_path(self, system: str, h: str) -> pathlib.Path:
        return self._dir / f"{system}_{h}_embed.npy"

    def load_tfidf(self, system: str, class_df) -> Optional[object]:
        """Return a cached SimilarityEngine, or None if not found / stale."""
        if not self.active:
            return None
        h = _table_hash(class_df)
        p = self._tfidf_path(system, h)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None

    def save_tfidf(self, system: str, class_df, engine: object) -> None:
        if not self.active:
            return
        h = _table_hash(class_df)
        p = self._tfidf_path(system, h)
        try:
            with open(p, "wb") as fh:
                pickle.dump(engine, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass  # cache write failures are non-fatal

    def load_embed(self, system: str, class_df) -> Optional[np.ndarray]:
        """Return cached embedding matrix (N, D) or None."""
        if not self.active:
            return None
        h = _table_hash(class_df)
        p = self._embed_path(system, h)
        if not p.exists():
            return None
        try:
            return np.load(str(p))
        except Exception:
            return None

    def save_embed(self, system: str, class_df, matrix: np.ndarray) -> None:
        if not self.active:
            return
        h = _table_hash(class_df)
        p = self._embed_path(system, h)
        try:
            np.save(str(p), matrix)
        except Exception:
            pass
