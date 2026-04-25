# Asset Classification Tool

This tool takes your list of assets and automatically finds the best matching classification codes for each one. It produces a colour-coded Excel report showing how confident each match is.

---

## Before you start

You need Python installed. Run this once to install the required packages:

```bash
pip install -r requirements.txt
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
- `asset_id` — a unique ID for each asset (e.g. AST001)
- `asset_name` — the full name of the asset (e.g. Main Street Bridge)

**Recommended (better results with these):**
- `asset_type` — what type of asset it is (e.g. Bridge, Pump Station)
- `description` — what the asset is and does
- `technical_specs` — materials, dimensions, standards

**Optional (include if you have them):**
- `existing_classification` — any old or current classification code you already have
- `notes` — any engineer comments or field observations
- `location` — where the asset is
- `condition` — current condition (Good / Fair / Poor)
- `manufacturer_model` — make and model (useful for cameras, signals, etc.)

> The more fields you fill in, the more accurate the results. Even a short description helps significantly.

---

### Step 3 — Prepare your classification tables

Each classification system you want to use (e.g. UNICLASS, AUSTROADS, TfNSW) needs its own CSV file.

**The sample files in the `data/` folder are ready to use as-is.** If you're using your own classification system, fill in the template with at minimum:
- `classification_code` — the code (e.g. Ss_25_13_15)
- `classification_name` — the name of the code (e.g. Bridge Structures)

Adding `classification_description`, `category`, and `subcategory` will improve accuracy.

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

Change `your_asset_register.csv` to the path of your file. Add or remove `--classifications` lines for each system you want to use.

Results are saved to the `output/` folder.

---

## What you get

Four files are created in your output folder:

| File | What it contains |
|---|---|
| `classification_results.xlsx` | The main report — open this first |
| `classification_results.csv` | Same data as a plain CSV |
| `classification_summary.txt` | A plain-text summary of results |
| `classification_summary.json` | Machine-readable summary |

### Reading the Excel report

Each row is one potential match. Every asset gets the top 3 matches per classification system, ranked best to worst.

**Rows are colour-coded by confidence:**

| Colour | Meaning |
|---|---|
| Green | Strong match — safe to accept after a quick check |
| Amber | Decent match — review before accepting |
| Red | Weak match — needs manual classification |

**Key columns to look at:**

| Column | What it tells you |
|---|---|
| `matched_classification_code` | The code that was matched |
| `matched_classification_name` | The name of that code |
| `similarity_score` | How strong the match is (0–100) |
| `composite_confidence` | Overall confidence combining score, cross-system agreement, and failure-mode alignment |
| `match_rank` | 1st is the best match, 2nd is next, and so on |
| `confidence_flag` | high / medium / low — same as the row colour |
| `reasoning` | Plain-English explanation of why this match was chosen |

---

## Extra options

### Speed up repeated runs with caching

If you run the tool multiple times with the same classification tables, use `--cache-dir` to save the fitting step:

```bash
python main.py \
  --assets your_asset_register.csv \
  --classifications UNICLASS:data/uniclass_classification.csv \
  --output output/ \
  --cache-dir cache/
```

The first run saves the fitted data to `cache/`. Every run after that loads from cache and starts faster. The cache resets automatically if you change a classification file.

### Get more or fewer match candidates

By default you get the top 3 matches per asset. Change this with `--top-n`:

```bash
python main.py --assets ... --classifications ... --top-n 5
```

---

## Something went wrong?

| Error message | What to do |
|---|---|
| `missing required columns: ['asset_id']` | Open your asset register and make sure the column is named exactly `asset_id` (lowercase, no spaces). Run `--generate-templates` to see the correct format. |
| `No such file or directory` | Check the file path in your command. Copy and paste the full path to be safe. |
| `ERROR: --assets is required` | You forgot to include `--assets your_file.csv` in the command. |
| `Unsupported file format` | The tool accepts `.csv` and `.xlsx` files only. |
