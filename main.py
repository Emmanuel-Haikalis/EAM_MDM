#!/usr/bin/env python3
"""
Asset Register Classification Integration Tool

Usage:
  python main.py --assets <path> --classifications NAME:PATH [NAME:PATH ...] [--output DIR] [--top-n N]

Example:
  python main.py \\
      --assets data/sample_asset_register.csv \\
      --classifications UNICLASS:data/uniclass_classification.csv \\
      --classifications AUSTROADS:data/austroads_classification.csv \\
      --classifications TFNSW:data/tfnsw_classification.csv \\
      --output output/
"""
import argparse
import sys

from src.data_loader import load_asset_register, load_classification_table
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
    p.add_argument("--assets", required=True, metavar="PATH",
                   help="Asset register file (CSV or Excel)")
    p.add_argument("--classifications", required=True, action="append",
                   metavar="NAME:PATH",
                   help="Classification system as NAME:PATH (repeat for multiple systems)")
    p.add_argument("--output", default="output", metavar="DIR",
                   help="Output directory (default: output/)")
    p.add_argument("--top-n", type=int, default=3, dest="top_n",
                   help="Top N candidates per asset per system (default: 3)")
    return p


def main() -> int:
    args = build_parser().parse_args()

    print(f"\n{SEP}")
    print("  Asset Classification Integration Tool")
    print(SEP)

    # ── Load asset register ──────────────────────────────────────────────────
    print(f"\nLoading asset register  : {args.assets}")
    try:
        asset_df = load_asset_register(args.assets)
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
    summary = generate_summary(results_df, systems)
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
