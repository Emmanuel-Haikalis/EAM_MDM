import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def save_results(results_df: pd.DataFrame, output_dir: str, base: str = "classification_results") -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / f"{base}.csv"
    xlsx_path = out / f"{base}.xlsx"

    results_df.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Classification Results", index=False)
        ws = writer.sheets["Classification Results"]
        for col_cells in ws.columns:
            max_len = max((len(str(cell.value or "")) for cell in col_cells), default=10)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 55)

    return {"csv": str(csv_path), "excel": str(xlsx_path)}


def generate_summary(results_df: pd.DataFrame, systems: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_assets_processed": int(results_df["asset_id"].nunique()),
        "total_classification_records": len(results_df),
        "systems": {},
    }

    manual_review: set = set()

    for system in systems:
        top1 = results_df[
            (results_df["classification_system"] == system)
            & (results_df["match_rank"] == "1st")
        ]
        if top1.empty:
            continue

        counts = top1["confidence_flag"].value_counts().to_dict()
        summary["systems"][system] = {
            "assets_classified": len(top1),
            "average_top_score": round(float(top1["similarity_score"].mean()), 1),
            "high_confidence": counts.get("high", 0),
            "medium_confidence": counts.get("medium", 0),
            "low_confidence": counts.get("low", 0),
            "flagged_for_review": counts.get("low", 0),
        }
        low_ids = top1[top1["confidence_flag"] == "low"]["asset_id"].tolist()
        manual_review.update(low_ids)

    summary["assets_requiring_manual_review"] = sorted(manual_review)
    summary["manual_review_count"] = len(manual_review)
    return summary


def save_summary(summary: Dict[str, Any], output_dir: str, base: str = "classification_summary") -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{base}.json"
    txt_path = out / f"{base}.txt"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "=" * 62,
        "  ASSET CLASSIFICATION SUMMARY REPORT",
        "=" * 62,
        f"  Total Assets Processed       : {summary['total_assets_processed']}",
        f"  Total Classification Records : {summary['total_classification_records']}",
        f"  Assets Requiring Review      : {summary['manual_review_count']}",
        "",
    ]

    for system, stats in summary["systems"].items():
        lines += [
            f"  [{system}]",
            f"    Assets Classified   : {stats['assets_classified']}",
            f"    Average Top Score   : {stats['average_top_score']}",
            f"    High Confidence     : {stats['high_confidence']}",
            f"    Medium Confidence   : {stats['medium_confidence']}",
            f"    Low Confidence      : {stats['low_confidence']}",
            f"    Flagged for Review  : {stats['flagged_for_review']}",
            "",
        ]

    if summary["assets_requiring_manual_review"]:
        lines.append("  Assets Requiring Manual Review:")
        for aid in summary["assets_requiring_manual_review"]:
            lines.append(f"    - {aid}")
        lines.append("")

    lines.append("=" * 62)

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return {"json": str(json_path), "txt": str(txt_path)}
