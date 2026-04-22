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
import sys

from src.data_loader import load_asset_register, load_classification_table, write_templates
from src.classifier import classify_all
from src.report_generator import save_results, generate_summary, save_summary

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
    return p


def main() -> int:
    args = build_parser().parse_args()

    print(f"\n{SEP}")
    print("  Asset Classification Integration Tool")
    print(SEP)

    # ── Template generation mode ─────────────────────────────────────────────
    if args.generate_templates:
        print(f"\nGenerating CSV templates → {args.output}/")
        paths = write_templates(args.output)
        print(f"  Asset register template  : {paths['asset_register']}")
        print(f"  Classification template  : {paths['classification_table']}")
        print(f"\n  Fill in the templates and run again with --assets and --classifications.")
        print(SEP + "\n")
        return 0

    # ── Validate required args for classification mode ────────────────────────
    if not args.assets:
        print("ERROR: --assets is required (or use --generate-templates to create blank templates).",
              file=sys.stderr)
        return 1
    if not args.classifications:
        print("ERROR: --classifications is required (or use --generate-templates).",
              file=sys.stderr)
        return 1

    # ── Load asset register ──────────────────────────────────────────────────
    print(f"\nLoading asset register  : {args.assets}")
    try:
        asset_df, data_quality = load_asset_register(args.assets)
    except Exception as exc:
        print(f"ERROR loading asset register: {exc}", file=sys.stderr)
        return 1
    print(f"  {len(asset_df)} asset(s) loaded")

    # ── Load classification tables ───────────────────────────────────────────
    classification_tables = {}
    for raw_arg in args.classifications:
        try:
            system_name, cls_path = _parse_classification_arg(raw_arg)
        except argparse.ArgumentTypeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"\nLoading classification  : {system_name}  ({cls_path})")
        try:
            cdf = load_classification_table(cls_path, system_name)
        except Exception as exc:
            print(f"ERROR loading '{system_name}': {exc}", file=sys.stderr)
            return 1
        classification_tables[system_name] = cdf
        print(f"  {len(cdf)} classification entries loaded")

    # ── Classify ─────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Running classification …")
    print(SEP)
    results_df = classify_all(asset_df, classification_tables, top_n=args.top_n)
    print(f"\n  {len(results_df)} result records generated")

    # ── Save outputs ─────────────────────────────────────────────────────────
    result_paths = save_results(results_df, args.output)
    systems = list(classification_tables.keys())
    summary = generate_summary(results_df, systems, data_quality=data_quality)
    summary_paths = save_summary(summary, args.output)

    # ── Console summary ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)
    print(f"  Total assets processed       : {summary['total_assets_processed']}")
    print(f"  Total records generated      : {summary['total_classification_records']}")
    print(f"  Assets for manual review     : {summary['manual_review_count']}")

    for system, stats in summary["systems"].items():
        print(f"\n  {system}:")
        print(f"    Average top-1 score  : {stats['average_top_score']}")
        print(f"    High / Med / Low     : {stats['high_confidence']} / {stats['medium_confidence']} / {stats['low_confidence']}")

    if "cross_system_consistency" in summary:
        cs = summary["cross_system_consistency"]
        print(f"\n  Cross-system consistency : {cs['consistency_pct']}%  "
              f"({cs['consistent_count']} consistent, {cs['inconsistent_count']} inconsistent)")

    print(f"\n{SEP}")
    print("  Output files:")
    print(f"    Results   CSV   : {result_paths['csv']}")
    print(f"    Results   Excel : {result_paths['excel']}")
    print(f"    Summary   JSON  : {summary_paths['json']}")
    print(f"    Summary   TXT   : {summary_paths['txt']}")
    print(SEP + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
