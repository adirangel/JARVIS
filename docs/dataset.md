# Conversational Dataset

This repository organizes training data for dialogue modeling under the `data/`
folder. The structure separates raw transcripts, metadata, and curated
training splits:

```text
data/
├── raw/          # Source transcripts gathered from first-party and public corpora
├── metadata/     # Conversation-level annotations and provenance information
└── splits/       # Materialized train/validation/test partitions
```

## Record schema

Normalized transcripts are stored as JSON Lines (JSONL) files where each line is
an object with the following fields:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `id` | string | Unique identifier for the conversational turn (includes conversation ID and turn index). |
| `conversation_id` | string | Identifier derived from the source filename that groups turns belonging to the same conversation. |
| `speaker` | string | Speaker label inferred from the transcript (falls back to `"unknown"`). |
| `text` | string | Sanitized utterance text with personally-identifiable information redacted. |
| `source` | string | Human-readable name describing where the transcript originated. |
| `metadata` | object | Additional annotations merged from `data/metadata/` (e.g., topics, languages, acquisition dates). |

The schema is produced by `tools.collect_data`, which also applies basic PII
scrubbing (emails, phone numbers, SSNs, and credit card numbers are replaced
with redaction tokens).

## Tooling workflow

1. **Collect raw transcripts** with `python -m tools.collect_data`.
   ```bash
   python -m tools.collect_data data/raw/my_corpus data/raw/my_corpus.jsonl \
       --pattern "*.txt" \
       --source-name "my_corpus" \
       --metadata data/metadata/my_corpus.json \
       --id-prefix "myc_"
   ```
   The command discovers transcript files, parses turn-by-turn utterances, and
   emits normalized JSONL ready for aggregation.

2. **Build dataset splits** with `python -m tools.build_dataset`.
   ```bash
   python -m tools.build_dataset data/raw/*.jsonl \
       --output-dir data/splits \
       --format jsonl \
       --dedupe-on text id \
       --seed 13 \
       --train 0.8 --val 0.1 --test 0.1
   ```
   The script concatenates multiple normalized files, removes duplicates (based
   on the fields provided), shuffles with a reproducible seed, and exports the
   requested partitions as JSONL or Parquet files.

## Quality checks

* **Schema validation** – ensure each record contains all required fields and
  that redaction tokens (`<EMAIL>`, `<PHONE>`, `<SSN>`, `<CC>`) appear instead of
  raw PII patterns.
* **Deduplication review** – inspect the logs or resulting counts to confirm
  that near-duplicate utterances are removed without losing distinct
  conversations.
* **Metadata coverage** – verify that every conversation referenced by
  transcripts has a corresponding entry in `data/metadata/`, especially when
  topic labels or licensing information are required.
* **Split balance** – compare the number of records in each split to detect
  class or speaker imbalances introduced by randomization.

## Contribution guidelines

* Store newly collected transcripts under `data/raw/` and accompany them with
  metadata files in `data/metadata/` documenting provenance and licensing.
* Run `python -m tools.collect_data` with a meaningful `--source-name` and `--id-prefix`
  so downstream datasets can trace origins.
* Include regression tests or sample snippets when adding new PII sanitization
  patterns to `tools/dataset_utils.py`.
* When contributing new splits, describe the input sources, deduplication
  strategy, and random seed in your pull request to maintain reproducibility.
