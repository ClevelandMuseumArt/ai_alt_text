# AI Alt Text Generator

Generates accessibility alt text for Cleveland Museum of Art artwork images using Google Gemini and CLIP embedding validation.

## How it works

Each image goes through a three-step pipeline:

1. **Classify** — Gemini analyzes the image and returns structured metadata: whether it contains people, is abstract, is 3D, contains text, and whether any people are iconic/recognizable figures.
2. **Caption** — A composable prompt is assembled from modular section files in `prompts/` based on the classification, then passed to Gemini with the image to generate an initial alt text.
3. **Validate** — CLIP embeddings compare the generated caption against the image using cosine similarity. If the score meets the threshold, the result is accepted. If not, the pipeline optionally retries using RAG-retrieved examples to refine the caption.

Output rows are tagged with `ALT_TEXT_MEETS_THRESHOLD`:
- `YES` — passed CLIP cosine threshold
- `NO` — failed after all retries
- `OVERWRITTEN` — manually edited in Piction (set externally, not by this script)

## Setup

```shell
pip install -r requirements.txt
```

Requires a Google Cloud service account JSON key file with Vertex AI access (`project: ai-alt-text-481516`).

Requires `cma_piction` (internal CMA package, installed via git in `requirements.txt`).

GPU is optional but recommended. Run `test_gpu.py` to verify CUDA availability before a large batch.

## Scripts

### test_gpu.py

Verifies CUDA availability and runs a basic GPU operation.

```shell
python -m test_gpu
```

---

### artwork_bulk_load.py

Downloads primary image data from the CO API into a CSV for use as `--bulk-data-path` input. Skips artworks that already have manually written alt text (`ALT_TEXT_MEETS_THRESHOLD = OVERWRITTEN` in Piction or `human_reviewed = true` in CO API).

Output CSV columns: `athena_id, accession_number, image_src, UMO_ID`

Output is written to `image_data/<timestamp>_co_api_data.csv`.

```shell
# Full collection
python -m artwork_bulk_load

# Test mode (500 random artworks)
python -m artwork_bulk_load --test 1

# Specific artworks by Athena ID
python -m artwork_bulk_load --art-ids 12345,67890

# With logging
python -m artwork_bulk_load --log-level DEBUG --log-file logs/bulk_load
```

| Flag | Default | Description |
|------|---------|-------------|
| `--test` | `0` | `1` = test mode (500 random artworks) |
| `--art-ids` | — | Comma-separated Athena IDs for targeted download |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--log-file` | — | Log file path prefix (timestamp appended) |

---

### generate_alt_text.py

Main generation script. Runs in two modes:

**Bulk mode** — reads a CSV (from `artwork_bulk_load.py`), generates alt text for all rows in parallel, writes results to CSV.

**Piction query mode** (default) — queries the Piction API for recently uploaded images, generates alt text, and posts results back to Piction.

```shell
# Bulk mode with RAG, storing metrics
python -m generate_alt_text \
  --bulk \
  --bulk-data-path image_data/my_data.csv \
  --gemini-credentials-file gemini-key.json \
  --classifier-model gemini-3-pro-preview \
  --captioner-model gemini-3-flash-preview \
  --refinement-model gemini-3-flash-preview \
  --rag-directory rag_examples \
  --with-rag \
  --store-metrics \
  --log-level DEBUG

# Piction query mode (daily CRON use)
python -m generate_alt_text \
  --gemini-credentials-file gemini-key.json \
  --piction-days-since-query 1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--bulk` | false | Enable bulk CSV processing mode |
| `--bulk-data-path` | — | Path to input CSV (required with `--bulk`) |
| `--gemini-credentials-file` | — | **Required.** Path to Google service account JSON |
| `--classifier-model` | `gemini-3-flash-preview` | Model for image classification |
| `--captioner-model` | `gemini-3-flash-preview` | Model for initial caption generation |
| `--refinement-model` | `gemini-3-flash-preview` | Model for RAG refinement pass |
| `--rag-directory` | — | Directory of RAG example `.txt` files |
| `--with-rag` | false | Always run RAG refinement pass and write a second output CSV |
| `--store-metrics` | false | Include `cosine_similarity` column in output |
| `--min-cosine` | `0.25` | CLIP cosine similarity threshold |
| `--max-retries` | `5` | Retry attempts per image |
| `--max-workers` | `8` | Parallel worker threads (bulk mode) |
| `--output-file` | auto | Output CSV path (default: timestamped) |
| `--piction-base-url` | `https://piction.clevelandart.org/cma/` | Piction base URL |
| `--piction-query` | *(internal endpoint)* | Piction query endpoint path |
| `--piction-update` | — | Piction update endpoint path |
| `--piction-days-since-query` | `1` | Days back to query Piction for uploads |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--log-file` | — | Log file path prefix (timestamp appended) |

**Output CSV columns:** `image_id, caption, ALT_TEXT_MEETS_THRESHOLD, attempts[, cosine_similarity]`

When `--with-rag` is set, two CSVs are written: the standard output and a `_rag_` prefixed file with RAG-refined captions.

---

### combine_results.py

Merges multiple result CSVs, deduplicating by `image_id`. Edit the `__main__` block to configure input glob pattern and output path, then run directly:

```shell
python combine_results.py
```

---

### analyze_results.py

Computes statistics (min, max, mean, median, top/bottom 10) on `cosine_similarity` and `inner_product` columns across one or more result CSVs. Writes a JSON report.

```shell
python -m analyze_results output.json results_1.csv results_2.csv
```

## Directory structure

```
prompts/              Modular prompt files assembled per image type
  base_rules.txt      Core alt text rules, always included
  classifier.txt      Classification prompt
  2d_section.txt      Rules for 2D works
  3d_section.txt      Rules for 3D/sculptural works
  abstract_section.txt
  people_rules.txt
  iconic_people_rules.txt
  text_section.txt
  examples_*.txt      Few-shot examples per category

rag_examples/         Reference alt texts used for RAG refinement
  example_1.txt
  example_2.txt
  ...

image_data/           Output of artwork_bulk_load.py (gitignored)
```
