# Asset Classification Tool

This tool takes your list of assets and automatically finds the best matching classification codes for each one. It produces a colour-coded Excel report showing how confident each match is, a plain-text summary, and a machine-readable JSON file.

---

## Before you start

You need Python 3.10 or later. Run this once to install the required packages:

```bash
pip install -r requirements.txt
```

For the advanced ML mode (optional):

```bash
pip install sentence-transformers
```

---

## How to use it — 4 steps

### Step 1 — Get the blank templates (first time only)

Run this to create pre-formatted spreadsheet templates:

```bash
python main.py --generate-templates --output templates/
```

Two files are created inside the `templates/` folder:
- `template_asset_register.csv` — your list of assets goes here
- `template_classification_table.csv` — the classification codes go here

---

### Step 2 — Fill in your asset register

Open `template_asset_register.csv` and add one row per asset.

**Required (the tool won't run without these):**
| Column | Example |
|---|---|
| `asset_id` | AST001 |
| `asset_name` | Main Street Bridge |

**Recommended (better results with these):**
| Column | Example |
|---|---|
| `asset_type` | Bridge Structure |
| `description` | Reinforced concrete bridge over Main Creek |
| `technical_specs` | Span: 45m; RC slab; Load: Class A |

**Optional (include if you have them):**
| Column | Example |
|---|---|
| `existing_classification` | BRG-RC-45 |
| `notes` | Last inspected 2023; minor spalling |
| `location` | Main Street over Main Creek |
| `condition` | Good |
| `manufacturer_model` | Siemens SEPAC |

> The more fields you fill in, the more accurate the results. Even a short `description` makes a significant difference.

---

### Step 3 — Prepare your classification tables

Each classification system you want to use (e.g. UNICLASS, AUSTROADS, TfNSW) needs its own CSV file.

**The sample files in the `data/` folder are ready to use as-is.** If you are using your own system, create a CSV with at minimum:
- `classification_code` — the code (e.g. Ss_25_13_15)
- `classification_name` — the name (e.g. Bridge Structures)

Adding `classification_description`, `category`, and `subcategory` improves accuracy. The `parent_code` column enables hierarchy-aware matching for multi-level classification trees.

---

### Step 4 — Run the tool

```bash
python main.py \
  --assets your_asset_register.csv \
  --classifications UNICLASS:data/uniclass_classification.csv \
  --classifications AUSTROADS:data/austroads_classification.csv \
  --classifications TFNSW:data/tfnsw_classification.csv \
  --output output/
```

Change `your_asset_register.csv` to the path of your file. Add or remove `--classifications` lines for each system you want to match against.

Results are saved to the `output/` folder.

---

## What you get

Four files are created in your output folder:

| File | What it contains |
|---|---|
| `classification_results.xlsx` | The main report — open this first |
| `classification_results.csv` | Same data as a plain CSV |
| `classification_summary.txt` | Plain-text summary of results |
| `classification_summary.json` | Machine-readable summary |

### Reading the Excel report

Each row is one potential match. Every asset gets up to 3 matches per classification system, ranked best to worst.

**Rows are colour-coded by confidence:**

| Colour | Meaning |
|---|---|
| Green | Strong match — safe to accept after a quick check |
| Amber | Decent match — review before accepting |
| Red | Weak match — needs manual classification |

### Key columns

| Column | What it tells you |
|---|---|
| `match_rank` | 1st is the best match, 2nd is next, and so on |
| `matched_classification_code` | The code that was matched |
| `matched_classification_name` | The name of that code |
| `similarity_score` | How closely this specific system's scoring found a match (0–100) |
| `composite_confidence` | **Use this to prioritise your review queue** — see below |
| `confidence_flag` | high / medium / low — the same signal as the row colour |
| `reasoning` | Plain-English explanation of why this match was chosen |
| `score_breakdown` | How each scoring component contributed |

### How to use `composite_confidence`

`composite_confidence` is the single most reliable signal for deciding which results to trust. It combines three independent signals into one score (0–100):

- **How well the asset text matched** (60% weight)
- **Whether multiple classification systems agreed** on the same domain (25% weight)
- **Whether the asset's failure modes align** with the matched class (15% weight — ML mode only)

**Decision rule:**

| `composite_confidence` | What to do |
|---|---|
| 70 or above | Accept. All signals agree — only a quick sanity check needed. |
| 45–69 | Review the `reasoning` column and the 2nd-ranked match before accepting. |
| Below 45 | Manual classification required. The tool has low confidence across all signals. |

> Use the `confidence_flag` column for colour-coding and quick filtering. Use `composite_confidence` for prioritising which rows to review first — sort by this column ascending to start with the hardest cases.

---

## Extra options

### Speed up repeated runs with caching

If you run the tool multiple times against the same classification tables, use `--cache-dir` to save the fitting step. Works in both fast and ML mode.

```bash
python main.py \
  --assets your_asset_register.csv \
  --classifications UNICLASS:data/uniclass_classification.csv \
  --output output/ \
  --cache-dir cache/
```

The first run saves fitted data to `cache/`. Every subsequent run loads from cache and starts faster. The cache resets automatically when a classification file changes.

### Get more or fewer match candidates

By default you get the top 3 matches per asset per system. Change this with `--top-n`:

```bash
python main.py --assets ... --classifications ... --top-n 5
```

### Control output verbosity

```bash
python main.py --assets ... --classifications ... --verbose   # full debug output
python main.py --assets ... --classifications ... --quiet     # errors only
```

### Use the higher-accuracy ML mode

ML mode adds sentence-transformer embeddings, two-stage category filtering, cross-register equivalence checking, and failure-mode alignment scoring. It takes longer but produces more reliable results on ambiguous assets.

```bash
pip install sentence-transformers

python main.py \
  --assets your_asset_register.csv \
  --classifications UNICLASS:data/uniclass_classification.csv \
  --output output/ \
  --mode ml \
  --cache-dir cache/
```

---

## Use as a Python library

You can call the tool directly from your own code:

```python
from src import classify_assets, load_asset_register, load_classification_table
import logging
logging.basicConfig(level=logging.INFO)

asset_df, quality = load_asset_register("assets.csv")
tables = {
    "UNICLASS": load_classification_table("uniclass.csv", "UNICLASS"),
}
results = classify_assets(asset_df, tables, top_n=3)

# results is a pandas DataFrame with all output columns
top1 = results[results["match_rank"] == "1st"]
review_queue = top1.sort_values("composite_confidence")
```

For ML mode:

```python
from src import classify_assets_ml
results = classify_assets_ml(asset_df, tables, top_n=3, cache_dir="cache/")
```

---

## Run with Docker

```bash
docker build -t asset-classifier .

docker run --rm \
  -v $(pwd)/data:/data/input \
  -v $(pwd)/output:/data/output \
  asset-classifier \
  --assets /data/input/sample_asset_register.csv \
  --classifications UNICLASS:/data/input/uniclass_classification.csv \
  --output /data/output/
```

---

## Adapting to your industry

The tool works with any structured asset register and any classification system. The domain vocabulary (acronym expansions, failure mode keywords) ships pre-loaded for infrastructure and transport assets and is fully configurable without touching code:

| File | What you can change |
|---|---|
| `config/acronyms.yaml` | Add abbreviations specific to your sector |
| `config/failure_vocab.yaml` | Replace or extend the 13 built-in failure domains |
| `config/code_to_domain.yaml` | Map your classification codes to failure domains |
| `config/domain_weights.yaml` | Tune scoring weights per domain |

---

## Something went wrong?

| Error message | What to do |
|---|---|
| `missing required columns: ['asset_id']` | Open your asset register and make sure the column is named exactly `asset_id` (lowercase, no spaces). Run `--generate-templates` to see the correct format. |
| `Classification table(s) are empty` | Your classification CSV loaded but had no rows. Check the file has data rows below the header. |
| `No such file or directory` | Check the file path in your command. Copy and paste the full path to be safe. |
| `ERROR: --assets is required` | You forgot to include `--assets your_file.csv` in the command. |
| `Unsupported file format` | The tool accepts `.csv` and `.xlsx` files only. |
| `sentence-transformers not installed` | Run `pip install sentence-transformers` before using `--mode ml`. |
