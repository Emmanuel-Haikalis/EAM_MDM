#!/usr/bin/env python3
"""
Asset Register Classification Integration Tool

Matches assets in a register against one or more standardised classification
systems (e.g. UNICLASS, AUSTROADS, TfNSW) using a hybrid TF-IDF / category /
keyword-Jaccard similarity engine with per-system score calibration.

Usage:
  python main.py --assets <path> --classifications NAME:PATH [NAME:PATH ...]
                 [--output DIR] [--top-n N]

  python main.py --generate-templates --output DIR

Examples:
  # Run classification
  python main.py \\
      --assets data/sample_asset_register.csv \\
      --classifications UNICLASS:data/uniclass_classification.csv \\
      --classifications AUSTROADS:data/austroads_classification.csv \\
      --classifications TFNSW:data/tfnsw_classification.csv \\
      --output output/

  # Generate blank CSV templates for filling in
  python main.py --generate-templates --output templates/

Asset register columns
  Required : asset_id, asset_name
  Recommended : asset_type, description, technical_specs
  Optional  : location, condition, existing_classification, notes, manufacturer_model

Classification table columns
  Required : classification_code, classification_name
  Recommended : classification_description, category, subcategory
  Optional  : keywords, parent_code
"""
import argparse
import logging
import sys

from src.logging_utils import configure_logging
from src.data_loader import load_asset_register, load_classification_table, write_templates
from src.classifier import classify_all
from src.report_generator import save_results, generate_summary, save_summary

logger = logging.getLogger(__name__)

SEP = "=" * 62


def _parse_classification_arg(arg: str) -> tuple[str, str]:
    if ":" not in arg:
        raise argparse.ArgumentTypeError(
            f"Must be in NAME:PATH format, got: {arg!r}"
        )
    name, path = arg.split(":", 1)
    return name.strip(), path.strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Asset Register Classification Integration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--assets", metavar="PATH",
                   help="Asset register file (CSV or Excel)")
    p.add_argument("--classifications", action="append", metavar="NAME:PATH",
                   help="Classification system as NAME:PATH (repeat for multiple systems)")
    p.add_argument("--output", default="output", metavar="DIR",
                   help="Output directory (default: output/)")
    p.add_argument("--top-n", type=int, default=3, dest="top_n",
                   help="Top N candidates per asset per system (default: 3)")
    p.add_argument("--generate-templates", action="store_true", dest="generate_templates",
                   help="Write blank CSV templates to --output dir and exit")
    p.add_argument(
        "--mode",
        choices=["fast", "ml"],
        default="fast",
        dest="mode",
        help=(
            "Scoring mode: 'fast' (TF-IDF + Jaccard, default) or 'ml' "
            "(adds sentence-transformer embeddings, two-stage recursive "
            "filtering, cross-register equivalence, and failure mode "
            "alignment — takes longer). "
            "ML mode requires: pip install sentence-transformers"
        ),
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        dest="cache_dir",
        help=(
            "Directory for caching fitted TF-IDF and embedding matrices. "
            "On first run matrices are saved here; on subsequent runs with "
            "the same classification tables they are loaded from cache, "
            "skipping the fit step. Cache is invalidated when tables change."
        ),
    )

    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging output.",
    )
    verbosity.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors.",
    )

    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.verbose:
        verbosity = 2
    elif args.quiet:
        verbosity = 0
    else:
        verbosity = 1
    configure_logging(verbosity)

    logger.info(SEP)
    logger.info("  Asset Classification Integration Tool")
    logger.info(SEP)

    # ── Template generation mode ─────────────────────────────────────────────
    if args.generate_templates:
        logger.info("Generating CSV templates → %s/", args.output)
        paths = write_templates(args.output)
        logger.info("  Asset register template  : %s", paths['asset_register'])
        logger.info("  Classification template  : %s", paths['classification_table'])
        logger.info("  Fill in the templates and run again with --assets and --classifications.")
        return 0

    # ── Validate required args for classification mode ────────────────────────
    if args.top_n < 1:
        logger.error("--top-n must be at least 1.")
        return 1
    if not args.assets:
        logger.error("--assets is required (or use --generate-templates to create blank templates).")
        return 1
    if not args.classifications:
        logger.error("--classifications is required (or use --generate-templates).")
        return 1

    # ── Load asset register ──────────────────────────────────────────────────
    logger.info("Loading asset register  : %s", args.assets)
    try:
        asset_df, data_quality = load_asset_register(args.assets)
    except Exception as exc:
        logger.error("Loading asset register failed: %s", exc)
        return 1
    logger.info("  %d asset(s) loaded", len(asset_df))

    # ── Load classification tables ───────────────────────────────────────────
    classification_tables = {}
    for raw_arg in args.classifications:
        try:
            system_name, cls_path = _parse_classification_arg(raw_arg)
        except argparse.ArgumentTypeError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("Loading classification  : %s  (%s)", system_name, cls_path)
        try:
            cdf = load_classification_table(cls_path, system_name)
        except Exception as exc:
            logger.error("Loading '%s' failed: %s", system_name, exc)
            return 1
        classification_tables[system_name] = cdf
        logger.info("  %d classification entries loaded", len(cdf))

    # ── Classify ─────────────────────────────────────────────────────────────
    logger.info(SEP)
    if args.mode == "ml":
        logger.info("  Running ML classification …")
        logger.info("  (embeddings + two-stage + cross-register + failure alignment)")
    else:
        logger.info("  Running classification …")
    logger.info(SEP)

    if args.mode == "ml":
        from src.classifier_ml import classify_all_ml
        results_df = classify_all_ml(asset_df, classification_tables, top_n=args.top_n)
    else:
        results_df = classify_all(asset_df, classification_tables, top_n=args.top_n,
                                  cache_dir=args.cache_dir)
    logger.info("  %d result records generated", len(results_df))

    # ── Save outputs ─────────────────────────────────────────────────────────
    result_paths = save_results(results_df, args.output)
    systems = list(classification_tables.keys())
    summary = generate_summary(results_df, systems, data_quality=data_quality)
    summary_paths = save_summary(summary, args.output)

    # ── Console summary ──────────────────────────────────────────────────────
    logger.info(SEP)
    logger.info("  SUMMARY")
    logger.info(SEP)
    logger.info("  Total assets processed       : %s", summary['total_assets_processed'])
    logger.info("  Total records generated      : %s", summary['total_classification_records'])
    logger.info("  Assets for manual review     : %s", summary['manual_review_count'])

    for system, stats in summary["systems"].items():
        logger.info("  %s:", system)
        logger.info("    Average top-1 score  : %s", stats['average_top_score'])
        logger.info("    High / Med / Low     : %s / %s / %s",
                    stats['high_confidence'], stats['medium_confidence'], stats['low_confidence'])

    if "cross_system_consistency" in summary:
        cs = summary["cross_system_consistency"]
        logger.info("  Cross-system consistency : %s%%  (%s consistent, %s inconsistent)",
                    cs['consistency_pct'], cs['consistent_count'], cs['inconsistent_count'])

    logger.info(SEP)
    logger.info("  Output files:")
    logger.info("    Results   CSV   : %s", result_paths['csv'])
    logger.info("    Results   Excel : %s", result_paths['excel'])
    logger.info("    Summary   JSON  : %s", summary_paths['json'])
    logger.info("    Summary   TXT   : %s", summary_paths['txt'])
    logger.info(SEP)

    return 0


if __name__ == "__main__":
    sys.exit(main())
