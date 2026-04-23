"""
ML Enhancement Engine for Asset Classification Tool.

Provides all machine learning primitives used in --mode ml:
  - EmbeddingEngine        : sentence-transformer wrapper with lazy loading
  - TwoStageClassifier     : category-level Stage 1 + code-level Stage 2 filtering
  - CrossRegisterIndex     : cross-register semantic equivalence lookup
  - FailureModeEngine      : ISO 14224-aligned failure mode vocabulary and scoring
  - TwoStageResult         : dataclass for two-stage classification results

No imports from classifier.py or classifier_ml.py (zero circular dependency).
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Failure mode vocabulary — ISO 14224 / RCM-aligned, 13 domains
# ---------------------------------------------------------------------------

_FAILURE_VOCAB: Dict[str, Dict[str, Any]] = {
    "bridge_structures": {
        "iso_14224_class": "Civil Structures – Load-bearing",
        "category_keywords": [
            "bridge", "viaduct", "superstructure", "substructure",
            "deck", "span", "girder", "abutment", "pier",
        ],
        "failure_modes": [
            "structural cracking", "spalling", "scour", "bearing failure",
            "chloride ingress", "delamination", "joint leakage", "fatigue",
            "settlement", "collision damage",
        ],
        "failure_mechanisms": [
            "corrosion", "carbonation", "alkali silica reaction", "overload",
            "differential settlement", "vibration", "freeze thaw", "impact",
        ],
        "inspection_terms": [
            "concrete condition", "steel corrosion", "deflection", "crack width",
            "cover depth", "load rating", "scour depth", "bearing inspection",
        ],
    },
    "pavement": {
        "iso_14224_class": "Civil Structures – Surface",
        "category_keywords": [
            "pavement", "road", "highway", "asphalt", "surfacing",
            "bitumen", "footpath", "pedestrian", "flexible", "rigid",
        ],
        "failure_modes": [
            "rutting", "fatigue cracking", "potholing", "ravelling",
            "surface deformation", "bleeding", "shoving", "stripping", "edge break",
        ],
        "failure_mechanisms": [
            "subgrade failure", "base failure", "moisture damage", "thermal cracking",
            "heavy vehicle loading", "aging", "oxidation", "frost heave",
        ],
        "inspection_terms": [
            "roughness index", "rutting depth", "cracking percentage", "skid resistance",
            "deflection testing", "pavement condition index", "surface texture",
        ],
    },
    "drainage": {
        "iso_14224_class": "Piping Systems",
        "category_keywords": [
            "drain", "culvert", "stormwater", "pipe", "channel",
            "kerb", "gutter", "waterway", "inlet", "open channel",
        ],
        "failure_modes": [
            "blockage", "structural collapse", "corrosion", "root intrusion",
            "sedimentation", "joint leakage", "overflow", "scour erosion",
        ],
        "failure_mechanisms": [
            "soil corrosion", "abrasion", "hydraulic pressure", "differential settlement",
            "vegetation growth", "construction defect", "aging", "erosion",
        ],
        "inspection_terms": [
            "CCTV inspection", "flow capacity", "joint condition", "invert level",
            "sediment depth", "structural condition", "headwall condition",
        ],
    },
    "pump_station": {
        "iso_14224_class": "Pumps and Compressors",
        "category_keywords": [
            "pump", "pump station", "wet well", "submersible",
            "rising main", "telemetry", "pumping",
        ],
        "failure_modes": [
            "bearing failure", "seal failure", "cavitation", "impeller wear",
            "motor failure", "blockage", "control failure",
        ],
        "failure_mechanisms": [
            "abrasive wear", "corrosion", "vibration fatigue", "overheating",
            "debris ingestion", "electrical fault", "aging insulation",
        ],
        "inspection_terms": [
            "vibration analysis", "flow rate", "discharge pressure", "motor current",
            "wet well level", "run hours", "seal condition", "bearing temperature",
        ],
    },
    "traffic_control": {
        "iso_14224_class": "Instrumentation and Control",
        "category_keywords": [
            "traffic signal", "controller", "detector", "intersection",
            "phase", "SCATS", "signal head", "signal control",
        ],
        "failure_modes": [
            "controller failure", "loop detector failure", "communication failure",
            "power supply failure", "signal timing fault", "phase conflict",
        ],
        "failure_mechanisms": [
            "software fault", "electrical surge", "moisture ingress", "aging components",
            "cable damage", "vandalism", "detector cut",
        ],
        "inspection_terms": [
            "phase timing", "detector sensitivity", "signal head condition",
            "controller diagnostics", "communication status", "power supply test",
        ],
    },
    "variable_message_sign": {
        "iso_14224_class": "Instrumentation and Control",
        "category_keywords": [
            "VMS", "variable message", "overhead sign", "gantry",
            "display", "LED sign", "dynamic sign", "motorway sign",
        ],
        "failure_modes": [
            "LED panel failure", "pixel dropout", "communication failure",
            "structural failure", "display corruption",
        ],
        "failure_mechanisms": [
            "LED aging", "moisture ingress", "power surge", "wind loading",
            "cable fault", "software error", "heat stress",
        ],
        "inspection_terms": [
            "pixel check", "display brightness", "structural inspection",
            "communication test", "power supply condition", "cabinet condition",
        ],
    },
    "lighting": {
        "iso_14224_class": "Electrical Systems",
        "category_keywords": [
            "lighting", "luminaire", "street light", "lamp", "pole",
            "HPS", "LED light", "column", "luminaire bracket",
        ],
        "failure_modes": [
            "lamp failure", "ballast failure", "pole corrosion", "cable fault",
            "photocell failure", "column impact damage",
        ],
        "failure_mechanisms": [
            "end of lamp life", "corrosion", "UV degradation", "vibration fatigue",
            "electrical fault", "moisture ingress", "vehicle impact",
        ],
        "inspection_terms": [
            "lamp condition", "column corrosion", "cable test", "photocell operation",
            "earthing test", "bracket condition", "base plate condition",
        ],
    },
    "tunnel": {
        "iso_14224_class": "Civil Structures – Underground",
        "category_keywords": [
            "tunnel", "tunnel lining", "underground", "ventilation fan",
            "jet fan", "tunnel drainage", "portal",
        ],
        "failure_modes": [
            "lining cracking", "water ingress", "joint leakage", "ventilation failure",
            "fire damage", "drainage failure",
        ],
        "failure_mechanisms": [
            "groundwater pressure", "geological movement", "construction joint defect",
            "aging concrete", "vibration", "seismic activity", "thermal stress",
        ],
        "inspection_terms": [
            "lining condition", "water ingress mapping", "drainage inspection",
            "ventilation test", "structural monitoring", "fire protection condition",
        ],
    },
    "its_cctv": {
        "iso_14224_class": "Instrumentation and Control",
        "category_keywords": [
            "CCTV", "camera", "PTZ", "incident detection", "surveillance",
            "monitoring", "ITS", "intelligent transport", "ANPR", "detection",
        ],
        "failure_modes": [
            "camera failure", "lens fouling", "communication failure",
            "pan-tilt mechanism failure", "power fault", "housing damage",
        ],
        "failure_mechanisms": [
            "aging electronics", "moisture ingress", "vandalism", "UV degradation",
            "cable damage", "power surge", "mechanical wear",
        ],
        "inspection_terms": [
            "image quality", "pan-tilt operation", "housing condition",
            "communication test", "cable condition", "mounting integrity",
        ],
    },
    "retaining_structures": {
        "iso_14224_class": "Civil Structures – Earth Retention",
        "category_keywords": [
            "retaining wall", "embankment", "soil retention", "cantilever",
            "MSE wall", "geotechnical", "earth retention",
        ],
        "failure_modes": [
            "sliding", "overturning", "bearing failure", "tie-back failure",
            "drainage failure", "wall cracking", "MSE strip corrosion",
        ],
        "failure_mechanisms": [
            "excess pore pressure", "soil saturation", "surcharge overload",
            "corrosion of reinforcement", "drainage blockage", "construction defect",
        ],
        "inspection_terms": [
            "wall deflection", "crack mapping", "drainage condition",
            "anchor load test", "facing condition", "vegetation inspection",
        ],
    },
    "water_supply": {
        "iso_14224_class": "Piping Systems – Pressure",
        "category_keywords": [
            "water main", "water pipe", "distribution pipe", "pressure main",
            "cast iron pipe", "DN", "water supply", "water distribution",
        ],
        "failure_modes": [
            "pipe burst", "corrosion perforation", "tuberculation", "valve failure",
            "joint failure", "water hammer",
        ],
        "failure_mechanisms": [
            "external corrosion", "internal corrosion", "pressure transient",
            "soil movement", "aging material", "electrolytic corrosion",
        ],
        "inspection_terms": [
            "pressure test", "pipe condition assessment", "CCTV survey",
            "corrosion survey", "leak detection", "break frequency",
        ],
    },
    "passenger_facilities": {
        "iso_14224_class": "Civil Structures – Amenity",
        "category_keywords": [
            "bus shelter", "passenger shelter", "waiting area", "shelter",
            "seating", "amenity", "passenger facility",
        ],
        "failure_modes": [
            "structural damage", "glazing failure", "vandalism",
            "display failure", "roof leakage",
        ],
        "failure_mechanisms": [
            "impact damage", "weathering", "aging", "graffiti", "vandalism",
            "UV degradation", "corrosion",
        ],
        "inspection_terms": [
            "structural condition", "glazing condition", "roof condition",
            "seating condition", "display operation", "graffiti assessment",
        ],
    },
    "road_safety": {
        "iso_14224_class": "Civil Structures – Safety",
        "category_keywords": [
            "guardrail", "safety barrier", "W-beam", "crash barrier",
            "road safety", "end treatment", "post", "barrier system",
        ],
        "failure_modes": [
            "vehicle impact damage", "post corrosion", "beam deformation",
            "end treatment failure", "cable tension loss", "splice failure",
        ],
        "failure_mechanisms": [
            "vehicle collision", "corrosion", "fatigue", "overload",
            "construction defect", "inadequate anchorage",
        ],
        "inspection_terms": [
            "barrier height", "post condition", "beam deflection",
            "end treatment condition", "splice condition", "corrosion rating",
        ],
    },
}

# ---------------------------------------------------------------------------
# Code-to-domain mapping — all 66 codes across UNICLASS, AUSTROADS, TfNSW
# ---------------------------------------------------------------------------

_CODE_TO_DOMAIN: Dict[str, str] = {
    # UNICLASS (22 codes)
    "Ss_25_10_30": "pavement",
    "Ss_25_10_37": "pavement",
    "Ss_25_10_38": "pavement",
    "Ss_25_10_50": "pavement",
    "Ss_25_13_15": "bridge_structures",
    "Ss_25_13_20": "drainage",
    "Ss_25_13_50": "bridge_structures",
    "Ss_25_13_75": "retaining_structures",
    "Ss_25_14_15": "traffic_control",
    "Ss_25_14_30": "variable_message_sign",
    "Ss_25_14_60": "road_safety",
    "Ss_25_14_70": "its_cctv",
    "Ss_25_16_40": "lighting",
    "Ss_25_16_50": "tunnel",
    "Ss_25_16_57": "its_cctv",
    "Ss_25_30_10": "drainage",
    "Ss_25_30_15": "drainage",
    "Ss_25_30_20": "pump_station",
    "Ss_25_50_15": "pavement",
    "Ss_25_60_25": "retaining_structures",
    "Ss_70_30_10": "passenger_facilities",
    "Ss_80_90_20": "water_supply",
    # AUSTROADS (22 codes)
    "ASTR-BRG-110": "bridge_structures",
    "ASTR-BRG-120": "drainage",
    "ASTR-BRG-130": "bridge_structures",
    "ASTR-PAV-210": "pavement",
    "ASTR-PAV-220": "pavement",
    "ASTR-PAV-230": "pavement",
    "ASTR-DRN-310": "drainage",
    "ASTR-DRN-320": "drainage",
    "ASTR-DRN-330": "pump_station",
    "ASTR-DRN-340": "drainage",
    "ASTR-TMG-410": "traffic_control",
    "ASTR-TMG-420": "variable_message_sign",
    "ASTR-TMG-430": "road_safety",
    "ASTR-TMG-440": "traffic_control",
    "ASTR-LGT-500": "lighting",
    "ASTR-STR-610": "retaining_structures",
    "ASTR-TUN-700": "tunnel",
    "ASTR-TUN-710": "tunnel",
    "ASTR-ITS-810": "its_cctv",
    "ASTR-ITS-820": "its_cctv",
    "ASTR-PSG-910": "passenger_facilities",
    "ASTR-UTL-960": "water_supply",
    # TFNSW (22 codes)
    "TfNSW-STR-001": "bridge_structures",
    "TfNSW-STR-002": "bridge_structures",
    "TfNSW-STR-003": "drainage",
    "TfNSW-STR-004": "retaining_structures",
    "TfNSW-STR-005": "tunnel",
    "TfNSW-PAV-001": "pavement",
    "TfNSW-PAV-002": "pavement",
    "TfNSW-PAV-003": "pavement",
    "TfNSW-DRN-001": "drainage",
    "TfNSW-DRN-002": "drainage",
    "TfNSW-DRN-003": "drainage",
    "TfNSW-DRN-004": "pump_station",
    "TfNSW-TRF-001": "traffic_control",
    "TfNSW-TRF-002": "variable_message_sign",
    "TfNSW-TRF-003": "road_safety",
    "TfNSW-LGT-001": "lighting",
    "TfNSW-TUN-001": "tunnel",
    "TfNSW-ITS-001": "its_cctv",
    "TfNSW-ITS-002": "its_cctv",
    "TfNSW-PST-001": "passenger_facilities",
    "TfNSW-UTL-001": "water_supply",
    "TfNSW-UTL-002": "its_cctv",
}


# ---------------------------------------------------------------------------
# EmbeddingEngine
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """Lazy-loading sentence-transformer wrapper.

    Uses all-MiniLM-L6-v2 (~80 MB, CPU-only).  If sentence-transformers is
    not installed, available=False and the tool degrades to fast-mode weights.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None
        self._available: Optional[bool] = None  # None = not yet attempted

    @property
    def available(self) -> bool:
        return self._available is True

    def _load(self) -> None:
        if self._available is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device="cpu")
            self._available = True
        except ImportError:
            print(
                "  [ML mode] sentence-transformers not installed — "
                "falling back to TF-IDF weights.\n"
                "  Install with: pip install sentence-transformers",
                file=sys.stderr,
            )
            self._available = False
        except Exception as exc:
            print(f"  [ML mode] Could not load embedding model: {exc}", file=sys.stderr)
            self._available = False

    def fit(self, texts: List[str]) -> np.ndarray:
        """Encode texts, L2-normalise, cache internally; return (N, D) matrix."""
        self._load()
        if not self.available:
            return np.zeros((len(texts), 1), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def transform(self, texts: List[str]) -> np.ndarray:
        """Encode and L2-normalise without caching; returns (N, D) matrix."""
        self._load()
        if not self.available:
            return np.zeros((len(texts), 1), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# TwoStageResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class TwoStageResult:
    top_categories: List[str]          # Stage-1 top-K category names
    category_scores: Dict[str, float]  # all-category scores (category → score×100)
    candidate_indices: List[int]       # class_df row indices in top categories
    category_mismatch_flag: bool       # set externally in _classify_one_ml
    stage1_available: bool             # False when embedding model absent


# ---------------------------------------------------------------------------
# TwoStageClassifier
# ---------------------------------------------------------------------------

class TwoStageClassifier:
    """Two-stage classification filter.

    Stage 1: embedding cosine on per-category aggregate texts → top-K categories.
    Stage 2: caller restricts code search to candidate_indices.
    """

    def __init__(self, embedding_engine: EmbeddingEngine) -> None:
        self._embedding_engine = embedding_engine
        self._cat_matrix: Optional[np.ndarray] = None
        self._cat_names: List[str] = []
        self._cat_to_indices: Dict[str, List[int]] = {}
        self._all_indices: List[int] = []

    def fit(self, class_df: pd.DataFrame, failure_engine: "FailureModeEngine") -> None:
        from .similarity_engine import build_classification_text

        # Group entries by category; build per-entry enriched text
        categories: Dict[str, Dict] = {}
        for i, (_, row) in enumerate(class_df.iterrows()):
            cat = str(row.get("category", "Unknown")).strip() or "Unknown"
            if cat not in categories:
                categories[cat] = {"texts": [], "indices": []}
            code = str(row.get("classification_code", ""))
            domain = failure_engine.get_domain_for_code(code)
            base = build_classification_text(row)
            enriched = failure_engine.enrich_classification_text(base, domain)
            categories[cat]["texts"].append(enriched)
            categories[cat]["indices"].append(i)

        self._cat_names = list(categories.keys())
        self._cat_to_indices = {cat: data["indices"] for cat, data in categories.items()}
        self._all_indices = list(range(len(class_df)))

        if not self._embedding_engine.available:
            self._cat_matrix = None
            return

        cat_agg_texts = [" ".join(data["texts"]) for data in categories.values()]
        self._cat_matrix = self._embedding_engine.transform(cat_agg_texts)

    def score_two_stage(
        self,
        asset_text: str,
        top_categories: int = 2,
    ) -> TwoStageResult:
        if self._cat_matrix is None or not self._embedding_engine.available:
            return TwoStageResult(
                top_categories=list(self._cat_names),
                category_scores={},
                candidate_indices=self._all_indices,
                category_mismatch_flag=False,
                stage1_available=False,
            )

        asset_emb = self._embedding_engine.transform([asset_text])  # (1, D)
        raw_scores = (asset_emb @ self._cat_matrix.T).flatten()      # (n_cats,)

        k = min(top_categories, len(self._cat_names))
        top_idx = np.argsort(raw_scores)[::-1][:k]
        top_cat_names = [self._cat_names[i] for i in top_idx]

        cat_score_dict = {
            self._cat_names[i]: round(float(raw_scores[i]) * 100, 1)
            for i in range(len(self._cat_names))
        }

        candidate_indices: List[int] = []
        for cat in top_cat_names:
            candidate_indices.extend(self._cat_to_indices.get(cat, []))

        return TwoStageResult(
            top_categories=top_cat_names,
            category_scores=cat_score_dict,
            candidate_indices=sorted(set(candidate_indices)),
            category_mismatch_flag=False,   # set by _classify_one_ml after ranking
            stage1_available=True,
        )


# ---------------------------------------------------------------------------
# CrossRegisterIndex
# ---------------------------------------------------------------------------

class CrossRegisterIndex:
    """Pre-computes semantic equivalence between codes across classification systems.

    Uses cosine similarity of sentence-transformer embeddings.  Pairs above
    ``threshold`` are stored as equivalents for quick lookup.
    """

    def __init__(self) -> None:
        self._equiv: Dict[str, List[Tuple[str, str, float]]] = {}

    def build(
        self,
        classification_tables: Dict[str, pd.DataFrame],
        system_matrices: Dict[str, np.ndarray],
        threshold: float = 0.65,
    ) -> None:
        systems = list(classification_tables.keys())
        code_lists: Dict[str, List[str]] = {
            sys: classification_tables[sys]["classification_code"].tolist()
            for sys in systems
        }

        for i, sys_a in enumerate(systems):
            mat_a = system_matrices.get(sys_a)
            if mat_a is None or mat_a.shape[1] == 1:
                continue
            for sys_b in systems:
                if sys_b == sys_a:
                    continue
                mat_b = system_matrices.get(sys_b)
                if mat_b is None or mat_b.shape[1] == 1:
                    continue
                sim = (mat_a @ mat_b.T)  # (n_a, n_b)
                for idx_a, code_a in enumerate(code_lists[sys_a]):
                    key = f"{sys_a}:{code_a}"
                    for idx_b, code_b in enumerate(code_lists[sys_b]):
                        score = float(sim[idx_a, idx_b])
                        if score >= threshold:
                            self._equiv.setdefault(key, []).append(
                                (sys_b, code_b, round(score, 3))
                            )

        # Sort each entry by score descending
        for key in self._equiv:
            self._equiv[key].sort(key=lambda t: -t[2])

    def lookup_equivalents(
        self,
        code: str,
        system: str,
        top_k: int = 2,
    ) -> List[Tuple[str, str, float]]:
        return self._equiv.get(f"{system}:{code}", [])[:top_k]

    def majority_domain_vote(
        self,
        top1_codes_by_system: Dict[str, str],
        failure_engine: "FailureModeEngine",
    ) -> Tuple[str, bool]:
        domain_votes: List[str] = []
        for code in top1_codes_by_system.values():
            if code:
                domain = failure_engine.get_domain_for_code(code)
                if domain:
                    domain_votes.append(domain)
        if not domain_votes:
            return "no_data", False
        most_common, count = Counter(domain_votes).most_common(1)[0]
        return most_common, count > len(domain_votes) / 2


# ---------------------------------------------------------------------------
# FailureModeEngine
# ---------------------------------------------------------------------------

class FailureModeEngine:
    """ISO 14224 / RCM-aligned failure mode vocabulary for classification enrichment.

    Enriches classification texts with reliability engineering vocabulary before
    TF-IDF / embedding fitting, and scores how well an asset description aligns
    with the expected failure modes of a matched classification domain.
    """

    def get_domain_for_code(self, code: str) -> Optional[str]:
        return _CODE_TO_DOMAIN.get(code)

    def get_vocab(self, domain: str) -> Dict[str, Any]:
        return _FAILURE_VOCAB.get(domain, {})

    def enrich_classification_text(
        self, base_text: str, domain: Optional[str]
    ) -> str:
        if not domain:
            return base_text
        vocab = _FAILURE_VOCAB.get(domain, {})
        if not vocab:
            return base_text
        extra = " ".join(
            vocab.get("failure_modes", [])
            + vocab.get("failure_mechanisms", [])
            + vocab.get("inspection_terms", [])
        )
        return f"{base_text} {extra}".strip()

    def compute_alignment_score(
        self,
        asset_text: str,
        domain: Optional[str],
        embedding_engine: EmbeddingEngine,
    ) -> float:
        if not domain:
            return 0.0
        vocab = _FAILURE_VOCAB.get(domain, {})
        if not vocab:
            return 0.0
        vocab_text = " ".join(
            vocab.get("failure_modes", [])
            + vocab.get("failure_mechanisms", [])
            + vocab.get("inspection_terms", [])
        )
        if embedding_engine.available:
            asset_emb = embedding_engine.transform([asset_text])   # (1, D)
            vocab_emb = embedding_engine.transform([vocab_text])   # (1, D)
            return round(float((asset_emb @ vocab_emb.T)[0, 0]) * 100, 1)
        # Fallback: Jaccard on cleaned token sets
        from .similarity_engine import _term_set
        asset_terms = _term_set(asset_text)
        vocab_terms = _term_set(vocab_text)
        union = asset_terms | vocab_terms
        if not union:
            return 0.0
        return round(len(asset_terms & vocab_terms) / len(union) * 100, 1)

    def top_failure_modes(self, domain: str, n: int = 5) -> str:
        vocab = _FAILURE_VOCAB.get(domain, {})
        modes = vocab.get("failure_modes", [])
        return ", ".join(modes[:n])
