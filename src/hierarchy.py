"""Multi-level classification hierarchy support.

HierarchyIndex enriches classification texts with ancestor vocabulary and
applies a leaf-score boost when the matched code is a specific leaf node
but the asset description only uses high-level vocabulary.

Expects optional columns in classification tables:
  parent_code      : code of the parent entry (empty string for root nodes)
  hierarchy_level  : integer depth (1 = root, 2 = child, 3 = grandchild, …)

If neither column is present, HierarchyIndex is a no-op — all methods return
unchanged inputs with zero performance overhead.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .similarity_engine import build_classification_text


_LEAF_BOOST_THRESHOLD = 65.0      # scores below this may benefit from parent re-check
_LEAF_BOOST_FACTOR    = 0.90      # parent_score × factor used for comparison


class HierarchyIndex:
    """Ancestor-enriched text builder and leaf-score booster.

    Usage::

        hi = HierarchyIndex()
        hi.build(class_df)                      # once per classification table
        texts = hi.enrich_texts(class_df)       # pass to SimilarityEngine.fit()
        # In Pass 2, per rank-1 result:
        score = hi.leaf_boost(score, code, calibrated_arr, class_df)
    """

    def __init__(self) -> None:
        self._ancestors: Dict[str, List[str]] = {}  # code → [parent, grandparent, …]
        self._code_to_row: Dict[str, int] = {}       # code → iloc index
        self._leaf_codes: set = set()
        self._active = False

    def build(self, class_df: pd.DataFrame) -> None:
        """Build ancestor index from class_df.

        Sets self._active = True only when parent_code column is present and
        at least one non-empty parent_code value exists.
        """
        self._ancestors = {}
        self._code_to_row = {}
        self._leaf_codes = set()
        self._active = False

        if "parent_code" not in class_df.columns:
            return

        codes = class_df["classification_code"].astype(str).tolist()
        parents = class_df["parent_code"].fillna("").astype(str).tolist()

        for i, code in enumerate(codes):
            self._code_to_row[code] = i

        if not any(p.strip() for p in parents):
            return

        self._active = True

        parent_map = {c: p.strip() for c, p in zip(codes, parents)}
        children_of: Dict[str, set] = {c: set() for c in codes}
        for code, parent in parent_map.items():
            if parent and parent in children_of:
                children_of[parent].add(code)

        self._leaf_codes = {c for c, children in children_of.items() if not children}

        for code in codes:
            chain: List[str] = []
            current = parent_map.get(code, "")
            visited: set = set()
            while current and current in self._code_to_row and current not in visited:
                chain.append(current)
                visited.add(current)
                current = parent_map.get(current, "")
            self._ancestors[code] = chain

    def ancestor_text(self, code: str, class_df: pd.DataFrame) -> str:
        """Return concatenated classification text of all ancestor rows."""
        if not self._active:
            return ""
        parts = []
        for anc_code in self._ancestors.get(code, []):
            idx = self._code_to_row.get(anc_code)
            if idx is not None:
                parts.append(build_classification_text(class_df.iloc[idx]))
        return " ".join(parts)

    def enrich_texts(self, class_df: pd.DataFrame) -> List[str]:
        """Return classification texts enriched with ancestor vocabulary.

        Child entries inherit the vocabulary of their parents, making specific
        leaf codes findable even when asset descriptions use only high-level terms.
        When hierarchy is absent, returns plain build_classification_text() output.
        """
        result = []
        for _, row in class_df.iterrows():
            base = build_classification_text(row)
            if self._active:
                code = str(row.get("classification_code", ""))
                anc_text = self.ancestor_text(code, class_df)
                if anc_text:
                    base = f"{base} {anc_text}"
            result.append(base)
        return result

    def is_leaf(self, code: str) -> bool:
        return self._active and code in self._leaf_codes

    def parent_row_index(self, code: str) -> Optional[int]:
        """Return the class_df iloc index of the direct parent, or None."""
        if not self._active:
            return None
        ancestors = self._ancestors.get(code, [])
        if not ancestors:
            return None
        return self._code_to_row.get(ancestors[0])

    def leaf_boost(
        self,
        score: float,
        code: str,
        calibrated: np.ndarray,
        class_df: pd.DataFrame,
    ) -> float:
        """Boost a low leaf-node score using its parent's calibrated score.

        If the matched code is a leaf AND score < threshold, compare against
        parent_score × factor.  Returns max(score, parent_score × factor).

        Returns the original score unchanged if:
          - hierarchy is absent (parent_code column not in class_df)
          - code is not a leaf node
          - score is already above _LEAF_BOOST_THRESHOLD
          - parent code not found in calibrated array
        """
        if not self._active or score >= _LEAF_BOOST_THRESHOLD:
            return score
        if not self.is_leaf(code):
            return score
        parent_idx = self.parent_row_index(code)
        if parent_idx is None or parent_idx >= len(calibrated):
            return score
        parent_score = float(calibrated[parent_idx]) * _LEAF_BOOST_FACTOR
        return max(score, parent_score)
