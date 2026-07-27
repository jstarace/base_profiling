# OCEAN LoRA Adapter Training

Trains **32 LoRA adapters, one per OCEAN ptype class (0–31)**, on `jingjietan/pandora-big5`.
Each adapter sees only the rows of its own class. Class membership comes from the `ptype`
column, which is authoritative and verified (see `dataset_construction/DATASET_OVERVIEW.md`
§2.1–2.2).

This is OCEAN training code. It is independent of the MBTI corpus pipeline — it does not use
`schema.py`, the manipulators, the orchestrator, or the cleaning/judges stack, and nothing in
those paths calls into here.

## Setup

| | |
|---|---|
| Base model | `meta-llama/Llama-3.1-8B` (base, **not** Instruct) |
| Adapters | 32, one per ptype, trained independently |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| Target modules | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |
| Precision | bf16, gradient checkpointing on |
| Pods | 3, row-balanced, running in parallel |
| Persistence | shared RunPod network volume at `$ADAPTER_OUT_DIR` |

`meta-llama/Llama-3.1-8B` is a **gated** repo (`gated: manual` on the Hub). The account behind
`HF_TOKEN` must have been granted access before the pod runs, or the model pull 401s. The
legacy id `meta-llama/Meta-Llama-3.1-8B` redirects here and should not be used.

### Training defaults

| flag | default | note |
|---|---|---|
| `--lr` | `2e-4` | conventional LoRA LR; cosine schedule, 3% warmup |
| `--epochs` | `1.0` | see *Compute imbalance* below |
| `--batch-size` | `8` | per device |
| `--grad-accum` | `4` | effective batch 32 |
| `--max-seq-len` | `512` | covers ~99% of rows uncut (median comment ≈ 103 chars, p99 ≈ 1,810) |
| `--max-steps` | `-1` | `-1` means use `--epochs` |
| `--seed` | `42` | |

Padding is dynamic (`DataCollatorForLanguageModeling`), so short comments do not pad to 512.

**Compute imbalance.** Classes span 486 to 345,069 rows, a 710× range, so a fixed epoch count
gives wildly unequal training per adapter — 1 epoch on ptype 9 is ~10,800 optimizer steps, on
ptype 23 it is ~15. The default is 1 epoch; `--max-steps N` caps every class at the same step
count instead, which is the option to use if adapters need to be comparable across classes.
This is an open decision, not a settled one.

## Partition

`partition_classes.py` greedy bin-packs the 32 classes into 3 groups **balanced by row count,
not class count**: classes sorted by rows descending, each assigned to whichever group has
fewest rows so far. The three largest classes (9, 25, 10) therefore land in different groups.

The result is committed to `ocean_partition.json` so a launch is reproducible. Regenerate with:

```bash
python training/partition_classes.py            # --dry-run to preview
```

| group | classes | ptypes (training order, largest first) | rows | share |
|---|---|---|---|---|
| 0 | 10 | 9, 0, 24, 30, 26, 14, 20, 3, 21, 18 | 1,003,286 | 33.37% |
| 1 | 12 | 25, 1, 17, 12, 4, 6, 13, 27, 19, 15, 7, 23 | 998,090 | 33.20% |
| 2 | 10 | 10, 8, 11, 31, 29, 28, 16, 2, 5, 22 | 1,005,190 | 33.43% |

Spread is 7,100 rows, 0.24% of the corpus. All 32 classes appear exactly once.

## Running a pod

```bash
export HF_TOKEN=hf_...                 # needs access to the gated Llama repo
export ADAPTER_OUT_DIR=/workspace/adapters   # network volume mount

pip install -r requirements.txt
python training/train_ocean_adapters.py --group 0
```

One pod per group, `--group 0|1|2`, all three writing to the same network volume. The mount
path is read from `ADAPTER_OUT_DIR` and created if missing; it is never hardcoded. It defaults
to `/workspace/adapters` if unset.

Each adapter is saved to `${ADAPTER_OUT_DIR}/ptype_<N>/`. Since the three groups are disjoint,
the pods never write the same directory.

**Data is regenerated on each pod** — `load_dataset("jingjietan/pandora-big5")`, all splits
concatenated, then filtered per class. No parquet is shipped, and none is committed.

**Resume-safe.** A ptype whose `adapter_model.safetensors` already exists on the volume is
skipped at startup, so a killed or preempted pod can be relaunched with the same command and
picks up where it stopped.

**Logging.** Start and end wall-clock time plus elapsed duration are printed per adapter, along
with its row count, so per-class cost is visible from the pod log.

Useful flags for a smoke test before committing a GPU to the full run:

```bash
python training/train_ocean_adapters.py --group 0 --dry-run          # resolve group + row counts only
python training/train_ocean_adapters.py --group 0 --limit 5000 --max-steps 5
```

## Hardware

bf16 LoRA on an 8B needs roughly 16 GB for weights plus activations and optimizer state —
a 48 GB card (L40S, A6000) or better is comfortable. `--load-4bit` switches to QLoRA via
bitsandbytes for smaller cards, at some quality cost.

## Files

| file | role |
|---|---|
| `partition_classes.py` | greedy row-balanced split of the 32 classes into pod groups |
| `ocean_partition.json` | the committed partition; read by the trainer via `--group` |
| `train_ocean_adapters.py` | trains and saves every adapter in one group |
