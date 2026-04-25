# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [2.0.0] — 2026-04-25

### Added
- **Config externalisation** — YAML files in `config/` for acronyms, failure vocabulary,
  code-to-domain mapping, and domain weights. All parameters tunable without touching source code.
- **Batch vectorized scoring** — `SimilarityEngine.score_batch()` processes all assets in a single
  matrix multiply, replacing the per-asset scoring loop. 10–50× faster on large registers.
- **Multi-level classification hierarchy** — `HierarchyIndex` honours `parent_code` /
  `hierarchy_level` columns; enriches child texts with ancestor vocabulary; applies leaf-score
  boost for specific leaf codes matched via high-level descriptions.
- **Streaming Excel writer** — `openpyxl` write-only mode for datasets exceeding 50,000 rows,
  avoiding out-of-memory errors on large registers.
- **Disk cache** — `FitCache` saves fitted TF-IDF and embedding matrices keyed by SHA-256 hash
  of the classification table; `--cache-dir` CLI flag enables on repeated runs.
- **Composite confidence score** — `composite_confidence` column combines similarity score (60%),
  cross-system domain agreement (25%), and failure-mode alignment (15%).
- **Richer reasoning** — rank-1 reasoning now includes score gap to 2nd candidate.
- **Proper Python packaging** — `pyproject.toml` with entry point `asset-classifier`, optional
  `[ml]` dependency group, and `[dev]` group for testing.
- **Structured logging** — replaced all `print()` calls with `logging` module; `--verbose` /
  `--quiet` CLI flags; library callers configure log level independently.
- **Public library API** — `from src import classify_assets, classify_assets_ml`
  for programmatic use without the CLI.
- **pytest test suite** — full test suite under `tests/` with fixtures, parametrize, and coverage.
- **Docker support** — multi-stage `Dockerfile`; `.dockerignore` for minimal image size.
- **CI workflow** — `.github/workflows/ci.yml` runs tests and linting on every push and PR.

### Changed
- `classify_all()` and `classify_all_ml()` now accept optional `cache_dir` parameter.
- Progress output is now at `INFO` log level (suppressed with `--quiet`).
- All warnings from `data_loader.py` now use `logging.warning()` instead of `print()`.
- `src/__init__.py` exports the public API surface.

---

## [1.0.0] — 2026-04-20

### Added
- Initial production release.
- Hybrid 3-component scorer: TF-IDF full-text (A) + TF-IDF category (B) + Jaccard keywords (C).
- Per-system min-max score calibration to [5, 95].
- Two-pass architecture (score → calibrate → output).
- `--mode ml` with sentence-transformer embeddings (Component D), two-stage category filtering,
  cross-register equivalence index, and ISO 14224 failure-mode vocabulary injection.
- Colour-coded Excel output (green / amber / red by confidence flag).
- Cross-system domain consistency metric.
- Data quality reporting.
- 24-test edge-case suite covering empty inputs, duplicates, ordinals, Unicode, large files,
  Windows CRLF, Excel loading, and ML mode fallback.
- Domain acronym expansion (18 transport engineering acronyms).
