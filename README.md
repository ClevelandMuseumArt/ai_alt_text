# AI Alt Text Generator

Generates accessibility alt text for Cleveland Museum of Art artwork images using Google Gemini and CLIP embedding validation.

## How it works

Each image goes through a three-step pipeline:

1. **Classify** — Gemini analyzes the image and returns structured metadata: whether it contains people, is abstract, is 3D, contains text, and whether any people are iconic/recognizable figures.
2. **Caption** — A composable prompt is assembled from modular section files in `prompts/` based on the classification, then passed to Gemini with the image to generate an initial alt text.
3. **Validate** — CLIP embeddings compare the generated caption against the image using cosine similarity. If the score meets the threshold, the result is accepted.

If the initial caption fails validation, the script can run a refinement pass using examples from `rag_examples/`. Those examples are retrieved by similarity to the generated caption text, then sent to Gemini to rewrite the caption in a style closer to the reference examples while keeping it grounded in the image.

Output rows are tagged with `ALT_TEXT_MEETS_THRESHOLD`:
- `YES` — passed CLIP cosine threshold
- `NO` — failed after all retries
- `OVERWRITTEN` — manually edited in Piction (set externally, not by this script)

## Setup

```shell
pip install -r requirements.txt
```

Copy `credentials.yml.example` to `credentials.yml` and fill in all values. This file is gitignored and must never be committed. It contains:

- Piction username, password, and endpoint URLs
- Google Cloud project ID and service account credentials
- Collection API base URL
- Gemini model names for each generation step

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

# Non-default credentials file
python -m artwork_bulk_load --config /path/to/credentials.yml
```

| Flag | Default | Description |
|------|---------|-------------|
| `--test` | `0` | `1` = test mode (500 random artworks) |
| `--art-ids` | — | Comma-separated Athena IDs for targeted download |
| `--config` | `credentials.yml` | Path to credentials YAML file |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--log-file` | — | Log file path prefix (timestamp appended) |

---

### cache_rag_embeddings.py

Pre-computes CLIP text embeddings for all `.txt` files in a RAG examples directory and saves them to `embeddings.npz` inside that directory. Run this once after any changes to `rag_examples/` — `generate_alt_text.py` will detect and load the cache automatically on startup, skipping per-example embedding at query time.

Without a cache, `generate_alt_text.py` still works but recomputes embeddings for every RAG example on every image processed, which adds significant overhead on small runs. On large bulk runs the overhead is amortized; on Piction query mode (a few images at a time) it can exceed the generation time itself.

```shell
# Generate cache in the default location (rag_examples/embeddings.npz)
python -m cache_rag_embeddings --rag-directory rag_examples

# Write cache to a custom path
python -m cache_rag_embeddings --rag-directory rag_examples --cache-file path/to/embeddings.npz

# Regenerate after adding or editing example files
python -m cache_rag_embeddings --rag-directory rag_examples --force
```

| Flag | Default | Description |
|------|---------|-------------|
| `--rag-directory` | — | Directory containing RAG example `.txt` files (required) |
| `--cache-file` | `<rag-directory>/embeddings.npz` | Output path for the `.npz` cache file |
| `--force` | false | Overwrite an existing cache file |
| `--device` | auto | Compute device: `cpu` or `cuda` |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

> **Keep the cache in sync.** Any time you add, edit, or remove a `.txt` file in `rag_examples/`, re-run this script with `--force`. If the cache is stale it will still be used — `generate_alt_text.py` does not detect changes to individual example files.

---

### generate_alt_text.py

Main generation script. Runs in two modes:

**Bulk mode** — reads a CSV (from `artwork_bulk_load.py`), generates alt text for all rows in parallel, writes results to CSV.

**Piction query mode** (default) — queries the Piction API for recently uploaded images, generates alt text, appends all results to a single JSON file, and posts results back to Piction. Without `--with-rag`, standard results are posted. With `--with-rag`, both standard and RAG results are generated and saved, but only the RAG results are posted to the DAM.

```shell
# Bulk mode with RAG, storing metrics
python -m generate_alt_text \
  --bulk \
  --bulk-data-path image_data/my_data.csv \
  --rag-directory rag_examples \
  --with-rag \
  --store-metrics \
  --log-level DEBUG

# Piction query mode (daily CRON use)
python -m generate_alt_text \
  --piction-days-since-query 1

# Piction query mode with RAG results posted to DAM
python -m generate_alt_text \
  --piction-days-since-query 1 \
  --rag-directory rag_examples \
  --with-rag

# Non-default credentials file
python -m generate_alt_text --config /path/to/credentials.yml
```

| Flag | Default | Description |
|------|---------|-------------|
| `--bulk` | false | Enable bulk CSV processing mode |
| `--bulk-data-path` | — | Path to input CSV (required with `--bulk`) |
| `--config` | `credentials.yml` | Path to credentials YAML file |
| `--rag-directory` | — | Directory of RAG example `.txt` files used during refinement |
| `--with-rag` | false | Also run a forced RAG refinement pass and write results to a second output file alongside the standard one. In Piction query mode, the RAG results are posted to the DAM instead of the standard results. |
| `--store-metrics` | false | Include `cosine_similarity` column/field in output |
| `--min-cosine` | `0.25` | CLIP cosine similarity threshold |
| `--max-retries` | `5` | Retry attempts per image |
| `--max-workers` | `8` | Parallel worker threads (bulk mode) |
| `--output-file` | auto | Output file path or directory. If a specific filename is provided, the RAG output (when `--with-rag` is set) is written to the same directory with `_rag` inserted before the extension (e.g. `results.csv` → `results_rag.csv`). If a directory path is provided, both files are auto-generated with timestamps inside that directory. If omitted, both files are auto-generated with timestamps in the working directory. Bulk mode outputs `.csv`; Piction query mode outputs `.json`. |
| `--piction-days-since-query` | `1` | Days back to query Piction for uploads |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--log-file` | — | Log file path prefix (timestamp appended) |

**Bulk mode output** — a single CSV with columns: `image_id, caption, ALT_TEXT_MEETS_THRESHOLD, attempts[, cosine_similarity]`

**Piction query mode output** — a single newline-delimited JSON file where each line is one result record with the same fields as above.

When `--with-rag` is set, a second output file is written alongside the standard one with `_rag_` in the filename, containing the RAG-refined captions.

---

### combine_results.py

Merges multiple result CSVs, deduplicating by `image_id`. Edit the `__main__` block to configure input glob pattern and output path, then run directly:

```shell
python combine_results.py
```

---

### analyze_results.py

Computes statistics (min, max, mean, median, top/bottom 10) on `cosine_similarity` across one or more result CSVs. Writes a JSON report.

```shell
python -m analyze_results output.json results_1.csv results_2.csv
```

## Directory structure

```
credentials.yml          Credentials and environment config (gitignored — never commit)
credentials.yml.example  Template with all required keys, safe to commit

prompts/                 Modular prompt files assembled per image type
  base_rules.txt         Core alt text rules, always included
  classifier.txt         Classification prompt
  2d_section.txt         Rules for 2D works
  3d_section.txt         Rules for 3D/sculptural works
  abstract_section.txt
  people_rules.txt
  iconic_people_rules.txt
  text_section.txt
  examples_*.txt         Few-shot examples per category

rag_examples/            Reference alt texts used for RAG refinement
  example_1.txt
  example_2.txt
  ...
  embeddings.npz         Pre-computed CLIP embeddings (generated by cache_rag_embeddings.py, gitignored)

image_data/              Output of artwork_bulk_load.py (gitignored)
```