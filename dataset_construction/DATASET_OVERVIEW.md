# MBTI Corpus: Dataset Creation and Cleaning Overview

This document is the entry point for the corpus side of the project: where the data comes
from, how it is shaped into one common corpus, how author self-references are scrubbed from
it, and which file documents which part. It is deliberately high level. Every section links
to the authoritative document or code for the details.

**Status (2026-07-22).** The data pipeline (download, manipulate, combine) and the cleaning
pipeline are built and validated. The full corpus is cleaned by `build_dataset.py` (§4;
resume-aware). The cleaning configuration in use is panel = Gemini + Anthropic + OpenAI, final
judge = DeepSeek (`noD__final-D`). The ablation that selects and justifies this config is not
in this repo yet — **TODO: bring in the ablation code, gold, and methodology.**

---

## 1. Purpose

The end goal of the project is type-conditioned text generation: training language models to
speak like users of specific MBTI profile types. Forum corpora labeled with MBTI types have a
built-in leak: authors frequently state their own type in the text ("I'm an INTP", "as a
Counselor...", or anaphorically, "I've come to love my type"). Any model trained or evaluated
on such text can read the label off the page instead of learning stylistic signal. The corpus
work therefore has two halves:

1. **Dataset creation**: ingest multiple sources into one uniform, provenance-tracked corpus.
2. **Cleaning**: remove author self-references to their own type, and only those, from that
   corpus before it is used downstream.

---

## 2. Sources

![Data pipeline](/assets/data_pipeline.svg)

| dataset | platform | rows contributed | license | citation |
|---|---|---|---|---|
| [`jingjietan/kaggle-mbti`](https://huggingface.co/datasets/jingjietan/kaggle-mbti) (Tan et al. 2025) | PersonalityCafe (via the datasnaek scrape) | 8,675 (one row per user, 50 posts joined with `\|\|\|`) | Apache 2.0, HF DOI 10.57967/hf/3955 | DOI 10.1038/s41598-025-22853-y |
| [`minhaozhang/mbti`](https://huggingface.co/datasets/minhaozhang/mbti) | Reddit | 221,692 chunks after per-author consolidation | not verified; confirm on the dataset page | no paper citation supplied |

Combined pre-clean corpus: **230,367 rows** in `processed_data/pre_clean_data.parquet`.
jingjietan is 3.77% of the corpus, minhaozhang 96.23%.

Provenance notes that matter:

- The two sources are from **different platforms** (PersonalityCafe vs Reddit), which is what
  supports the cross-source generalization claim. Any future source should be hash-intersected
  against existing text before ingest; several public MBTI sets are re-hosted derivatives of
  the same scrapes.
- **jingjietan is truncated**: posts cap at roughly 200 characters and about 50% end in
  "...", cut mid-clause before the `|||` join. The lost text is unrecoverable. minhaozhang is
  not truncated. This is a stated corpus limitation.
- **PANDORA** (TakeLab) access has been requested and would be a provenance upgrade if
  granted (native Big Five and MBTI labels, published paper). The original is not ingested;
  Tan's Big Five redistribution of it is downloaded but unused (§2.1).
- Datasets that were considered and dropped are logged with reasons in `EXCLUSIONS.md`
  (internal only, not for write-ups).

### 2.1 OCEAN sources (downloaded, not in the corpus)

Two Big Five sources are downloaded and sitting in `raw_data/`. They are **not** part of the
MBTI corpus: they are not registered in `orchestrator.py`, not reachable from
`build_dataset.py`, have no manipulator, and the `schema.py` contract does not apply to them.
They are raw holdings for future OCEAN work.

| importer | dataset | rows | raw file |
|---|---|---|---|
| `import_tan_ocean.py` | [`jingjietan/essays-big5`](https://huggingface.co/datasets/jingjietan/essays-big5) | 2,467 | `raw_data/tan_ocean.parquet` (4.8 MB) |
| `import_tan_pandora.py` | [`jingjietan/pandora-big5`](https://huggingface.co/datasets/jingjietan/pandora-big5) | 3,006,566 | `raw_data/tan_pandora.parquet` (480 MB) |

Both ship the same eight columns — `O`, `C`, `E`, `A`, `N`, `ptype`, `text`, and an upstream
`__index_level_0__` — but with different label encodings:

- **essays-big5**: OCEAN as binary strings (`"0"` / `"1"`), `ptype` a string.
- **pandora-big5**: OCEAN as float percentiles (0–100), `ptype` an int.

`ptype` in both is a 5-bit packing of the binarized traits,
`ptype = 16·O + 8·C + 4·E + 2·A + 1·N` (32 values), which is a *different* scheme from the
corpus's 4-bit MBTI `PTYPE = 8O + 4C + 2E + 1A` (§3). Both were verified empirically at zero
mismatches: essays directly from its binary columns, and pandora from its percentile columns
under a strict `> 50` threshold (0 mismatches across all 3,006,566 rows). So pandora's
upstream binning is a recoverable median split, and `ptype` can be treated as the source of
truth for a binary OCEAN label without re-deriving a threshold.

Both importers concatenate all three upstream splits (train / validation / test) into one
frame, deliberately: the two OCEAN sources are expected to be merged and possibly deduped
later, so split boundaries are not preserved at ingest.

The PANDORA training source should be cited as Gjurković et al. (2021),
[*PANDORA Talks: Personality and Demographics on Reddit*](https://aclanthology.org/2021.socialnlp-1.12/),
DOI `10.18653/v1/2021.socialnlp-1.12`. Full source and benchmark attribution is
collected in the repository-level `DATASET_ATTRIBUTION.md`.

### 2.2 OCEAN class structure and the author-proxy tiering

**All 32 OCEAN classes are retained and trained on all available data. No authors and no rows
are cut.** Class membership is authoritative from `ptype` (§2.1), so no re-binning is applied.

Row counts overstate trainable diversity: pandora is per-comment with per-author labels, so a
single author contributes thousands of rows all carrying their one label. The parquet has **no
author, user, or id column** — its only non-trait column beyond `text` is `__index_level_0__`,
which is fully distinct across all 3,006,566 rows and is therefore an upstream row index, not
an author key. The usable diversity metric is the count of distinct `(O, C, E, A, N)`
percentile tuples, one tuple per author, totaling **1,548** across the corpus (≈1,940 rows per
author, consistent with PANDORA's published author scale). This is a lower bound: two authors
sharing an identical five-trait tuple would collapse into one count.

Per-class counts and tiers live in `processed_data/ocean_class_counts.csv` (32 rows: `ptype`,
`pattern`, `decoded`, `rows`, `distinct_authors`, `tier`, `pct_of_total`). Tiers are cut at the
two largest ratio breaks in the author distribution — `14 → 8` (1.75×) and `26 → 19` (1.37×):

| tier | distinct authors | classes | rows | authors |
|---|---|---|---|---|
| `solid` | ≥ 26 | 24 | 2,862,658 | 1,460 |
| `asterisk` | 10–25 | 4 (ptype 21, 18, 3, 15) | 99,752 | 66 |
| `floor` | < 10 | 4 (ptype 22 at 8, 19 at 7, 23 at 5, 7 at 2) | 44,156 | 22 |

The tier column is **annotation only** — it drives no filtering, subsampling, or reordering.
`asterisk` classes may carry more individual voice than trait signal. `floor` classes are
trained like the rest but flagged as potentially closer to individual impersonation than trait
representation; ptype 7 is two people.

Ranking by rows and by authors diverge sharply, which is the reason the proxy matters: ptype 19
is 25th by rows but 30th by authors (28,097 rows from 7 people), and the largest class, ptype 9,
is 345,069 rows from 150 authors.

Whether the sparse classes can be reproduced via direction composition rather than their own
adapters is contingent on the coupling result, not assumed.

Downstream, these 32 classes each get their own LoRA adapter over `meta-llama/Llama-3.1-8B`,
trained on 3 row-balanced pods. That code is in `training/` and is independent of this corpus
pipeline — see `training/README.md`.

---

## 3. The data pipeline: download, manipulate, combine

The ingest is a strict ETL split under `data/`, driven by `orchestrator.py`.

- **`importers/`** download only. One `import_<source>.py` per source, writing the raw source
  to `data/raw_data/<source>.parquet` only if it is missing. Zero transformation.
- **`manipulators/`** do all shaping. One `Manipulator` subclass per source turns raw rows
  into the common schema: consolidate to one row per author, chunk, derive the MBTI columns,
  stamp provenance. `finalize()` enforces the schema exactly, so cross-dataset parity is
  structural rather than a manual backfill that could be missed.
- **`schema.py`** is the single source of truth for the column contract, including the
  MBTI derivation helpers shared by every manipulator so type encoding cannot drift.
- **`chunking.py`** packs an author's text into chunks under an 11,000 character cap without
  ever splitting a sentence, labeling each chunk `#` of `Tot`. Sources whose rows already fit
  (jingjietan) pass through as 1-of-1.
- **`orchestrator.py`** runs the whole flow per registered source and writes the single
  combined `processed_data/pre_clean_data.parquet`.

**Validation baseline.** `mbti_profiles.parquet` (the original jingjietan build) is kept, not
deleted: the new jingjietan flow must reproduce its content columns exactly. This is the
direct check that the refactored pipeline generates the same thing as before.

**Key schema facts.** Every row carries `dataset` and a time-sortable `id` for provenance,
`author_id` (hashed handle, or null for anonymous sources), and the chunk columns `#` / `Tot`.
The MBTI-to-axis mapping (O to N/S, C to J/P, E to E/I, A to F/T, with
`PTYPE = 8O + 4C + 2E + 1A`) was validated empirically at zero mismatches across all 8,675
jingjietan rows.

---

## 4. The cleaning pipeline

![Cleaning pipeline](/assets/cleaning_pipeline.svg)

The cleaner (in `cleaning/`, adapters in `judges/`) is a committee, not a single model, so no
one model's blind spot is decisive. Per row:

1. **Segment.** `segment()` splits on the source's hard delimiter, then sentence-tokenizes
   with syntok, producing a canonical sentence array with absolute character offsets. This
   array is the shared coordinate system for everything downstream.
2. **Panel detect.** Three judges each flag the canonical indices they believe contain an
   author self-reference (explicit or implicit), each flag carrying a verbatim quote.
3. **Standardize.** Quotes are validated against their claimed sentence; mis-indexed flags
   are dropped and logged. Votes are tallied per index, anonymized so the final judge is not
   biased by who flagged what.
4. **Gates.** A row gate (majority of overall verdicts) and a sentence gate (per-index votes
   at or above majority) decide what reaches the final judge. Sub-majority sentences get
   `no_action` and are never reviewed.
5. **Final judge.** On each majority sentence: `cut`, `rephrase`, or `veto` (overturn the
   panel). The same model instance also drives the transformer, so judge and rephraser are
   always bound.
6. **Transformer.** Executes edits by character-offset span; the text is byte-faithful
   everywhere except edited spans.
7. **Log.** Every row becomes one JSONL `CleaningRecord`; `report.py` renders the run as a
   readable Markdown report.

The action vocabulary distinction is deliberate: `no_action` (never reviewed) and `veto`
(reviewed and overturned) both leave text unchanged but are logged separately, so the panel
overturn rate is measurable.

**Running it.** `build_dataset.py` is the entry point: it ensures `pre_clean_data.parquet`
exists (running the ingest if it does not), then cleans the corpus through the harness. The
cleaned text is written to a new `validated_text` column; the original `text` is left
untouched and every other column is preserved. Rows are processed by a thread pool
(`WORKERS`, default 40) in batches of `BATCH_SIZE` (default 95,000 — under Gemini's daily
request cap); after each batch it pauses for `[C]ontinue` / `[Q]uit`. It is resume-aware —
each cleaned row is checkpointed as it completes (with per-row backoff on transient errors),
so a re-run (or `--limit N` for a single partial pass) skips finished rows and continues. The
final `cleaned_data.parquet` is written only once every row is done.

**Cost — TODO before the full run.** The full-corpus run at the current config (3-model
panel, `gpt-4o` + `gemini-2.5-flash`) is expensive; the config is tuned for accuracy, not
cost. Cheaper path, to settle before running all 230k: disable Gemini thinking / use
`flash-lite`, move OpenAI to a mini tier, gate out no-anchor rows via the `draw_sample` regex
proxy (~60% of the corpus has no code/nickname), and use the providers' batch APIs (~50% off).

---

## 5. Judges and the configuration in use

![Judge roles](/assets/judge_roles.svg)

All four adapters (Gemini `gemini-2.5-flash`, Anthropic `claude-haiku-4-5`, OpenAI `gpt-4o`,
DeepSeek `deepseek-chat`) are role-agnostic: each can be a panelist, the final judge, or the
transformer, and each owns its three role prompts so tuning one model never forces retesting
the others.

The configuration in use, set in `cleaning/harness.py`:

- **Panel:** Gemini + Anthropic + OpenAI
- **Final judge (and transformer):** DeepSeek

This is `noD__final-D`, one of the four valid independent-final-judge configs (a final judge
must not sit on the panel it arbitrates). The v1 ablation found the four eligible configs
statistically tied at n=20 (end-to-end F1 0.731 to 0.760), with one durable finding: DeepSeek
is a weak panelist but the most conservative, clean-text-preserving final judge. The roles
reward opposite dispositions, which is exactly what the role-agnostic design exploits.

**Ablation status — TODO.** The ablation that justifies this config (v1: 20-row model-built
gold; v2: 100-row human-labeled exhaustive gold, designed but on hold) is not in this repo
yet — it lives in the working sandbox. TODO: promote the ablation code, gold, and methodology
here so this claim is reproducible from the repo.

---

## 6. Known limitations (carried forward)

- **jingjietan truncation** (about 200-char post cap, roughly half the posts cut mid-clause).
- **No-op rephrases**: an implicit reference anchored in a different sentence cannot be
  removed by rephrasing the flagged sentence alone. Unresolved; partly a segmentation
  artifact that the ellipsis fix may reduce.
- **Type-name introduction policy**: rephrasing "us" can name the type. Whether the target is
  author self-references only, or all type mentions, is an open definitional choice.
- **Model-built v1 gold**: the reason v2 exists; a stated threat to validity until then.
- **minhaozhang provenance**: no published citation or collection statement; license
  unverified.

---

## 7. Document and file map

| file / location | what it covers |
|---|---|
| `DATASET_OVERVIEW.md` (this file) | Entry point; the whole corpus story at a glance |
| `build_dataset.py` | Entry point: ingest if needed, then clean the corpus; resume-aware (§4) |
| `data/` | Ingest: importers (download), manipulators (shape), `schema.py`, `chunking.py`, `orchestrator.py` |
| `judges/` | Role-agnostic model adapters and their role prompts |
| `cleaning/` | segment, standardize, gates, final judge, transformer, harness, report |
| `utilities/` | id scheme and run log |
| `training/` (repo root, not under `dataset_construction/`) | OCEAN LoRA adapter training, independent of this pipeline (§2.2) — see `training/README.md` |

**TODO — docs not yet in this repo.** The design-rationale and ablation docs (ingest design,
schema parity, ablation methodology, labeling definition, exclusions log) still live in the
working sandbox. Promote the relevant ones here so this overview's references resolve.

Figures live in `assets/`: `data_pipeline.svg`, `cleaning_pipeline.svg`, `judge_roles.svg`.
