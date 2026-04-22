# Asset Register Classification Tool

Automatically matches your asset register against standardised classification systems (UNICLASS, AUSTROADS, TfNSW) and outputs a scored, colour-coded Excel report.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
python main.py \
  --assets data/sample_asset_register.csv \
  --classifications UNICLASS:data/uniclass_classification.csv \
  --classifications AUSTROADS:data/austroads_classification.csv \
  --classifications TFNSW:data/tfnsw_classification.csv \
  --output output/
```

Results are saved to the `output/` folder.

---

## Step-by-Step Guide

### Step 1 — Get blank templates

If you're starting from scratch, generate pre-formatted CSV templates:

```bash
python main.py --generate-templates --output templates/
```

This creates:
- `templates/template_asset_register.csv` — fill in your assets
- `templates/template_classification_table.csv` — fill in your classification codes

---

### Step 2 — Prepare your Asset Register

Open `template_asset_register.csv` and fill in your assets.

| Column | Required? | Description |
|---|---|---|
| `asset_id` | **Yes** | Unique ID (e.g. AST001) |
| `asset_name` | **Yes** | Full asset name |
| `asset_type` | Recommended | High-level type (e.g. Bridge Structure) |
| `description` | Recommended | What the asset is and does |
| `technical_specs` | Recommended | Materials, dimensions, standards |
| `location` | Optional | Where the asset is |
| `condition` | Optional | Good / Fair / Poor |
| `existing_classification` | Optional | Any legacy/current classification code — **boosts accuracy** |
| `notes` | Optional | Engineer remarks or field notes — **boosts accuracy** |
| `manufacturer_model` | Optional | Make and model (useful for ITS/mechanical assets) |

> **Tip:** The more fields you fill in, the better the match quality. `existing_classification` and `notes` are especially powerful if available.

---

### Step 3 — Prepare your Classification Tables

Each classification system (UNICLASS, AUSTROADS, TfNSW, or your own) needs its own CSV file.

| Column | Required? | Description |
|---|---|---|
| `classification_code` | **Yes** | The code (e.g. Ss_25_13_15) |
| `classification_name` | **Yes** | Short name |
| `classification_description` | Recommended | Full description of what the code covers |
| `category` | Recommended | High-level domain (e.g. Structures, Drainage) |
| `subcategory` | Recommended | Mid-level group (e.g. Bridges, Pump Stations) |
| `keywords` | Optional | Comma-separated synonyms or index terms |
| `parent_code` | Optional | Parent code for hierarchy reference |

> The sample classification files in `data/` are ready to use as-is.

---

### Step 4 — Run the tool

```bash
python main.py \
  --assets your_asset_register.csv \
  --classifications UNICLASS:uniclass.csv \
  --classifications AUSTROADS:austroads.csv \
  --output output/
```

You can include as many `--classifications` systems as you need.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--assets` | — | Path to your asset register (CSV or Excel) |
| `--classifications` | — | `NAME:PATH` pair, repeat for each system |
| `--output` | `output/` | Folder for all output files |
| `--top-n` | `3` | Number of candidate matches per asset per system |
| `--generate-templates` | — | Write blank templates and exit |

---

## Output Files

All outputs land in your `--output` folder.

| File | Contents |
|---|---|
| `classification_results.csv` | Full results table — one row per asset × system × match rank |
| `classification_results.xlsx` | Same data as Excel, colour-coded by confidence |
| `classification_summary.json` | Machine-readable summary with statistics |
| `classification_summary.txt` | Human-readable summary report |

### Excel Colour Coding

| Colour | Confidence | Meaning |
|---|---|---|
| 🟢 Green | High (score ≥ 70) | Reliable match — accept with spot-check |
| 🟡 Amber | Medium (score 45–69) | Good match — review before accepting |
| 🔴 Red | Low (score < 45) | Ambiguous — manual classification needed |

### Result Columns

| Column | Description |
|---|---|
| `asset_id` / `asset_name` | From your register |
| `classification_system` | UNICLASS / AUSTROADS / TFNSW etc. |
| `matched_classification_code` | Best-matching code |
| `matched_classification_name` | Name of that code |
| `matched_category` | Domain category of the match |
| `similarity_score` | 0–100 (calibrated per system) |
| `match_rank` | 1st / 2nd / 3rd best match |
| `confidence_flag` | high / medium / low |
| `reasoning` | Why this match was selected |
| `score_breakdown` | Component scores: `TF-IDF \| Category \| Jaccard` |

---

## Summary Report

After each run the console prints (and `classification_summary.txt` saves):

- Total assets processed and confidence distribution per system
- Which assets need manual review
- Cross-system consistency — whether all three systems agree on the asset's domain
- Data quality — which fields in your register are populated

---

## Tips for Better Results

1. **Fill in `description` and `technical_specs`** — these two fields have the most impact on match quality.
2. **Use `existing_classification`** if you have legacy codes — the tool treats these as a strong prior signal.
3. **Add `notes`** from field staff — informal remarks often contain specific technical terms that improve accuracy.
4. **Use the `keywords` column** in your classification tables to add synonyms and alternative terms.
5. **Low-confidence assets** (red rows) are flagged in the summary — focus your manual review effort there.

---

## Common Errors

| Error | Fix |
|---|---|
| `missing required columns: ['asset_id']` | Check your column names match the template — run `--generate-templates` to see the expected format |
| `File not found: path/to/file.csv` | Check the file path; use absolute paths if needed |
| `Unsupported format '.xlsx'` | Ensure `openpyxl` is installed: `pip install openpyxl` |
