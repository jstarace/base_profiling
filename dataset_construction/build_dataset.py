"""Build the cleaned corpus: import if needed, then run the harness over it."""
import os
import json
import argparse

import paths  # noqa: F401
import pandas as pd

from dataset_construction.data import orchestrator
from dataset_construction.cleaning.harness import Harness

PROCESSED_DIR = paths.ROOT / "dataset_construction" / "data" / "processed_data"
PRE_CLEAN = orchestrator.PROCESSED                       # data/processed_data/pre_clean_data.parquet
CLEANED = PROCESSED_DIR / "cleaned_data.parquet"
CHECKPOINT = PROCESSED_DIR / "cleaned_data.checkpoint.jsonl"


def main():
    ap = argparse.ArgumentParser(description="Build the cleaned corpus (import if needed, then clean).")
    ap.add_argument("--limit", type=int, default=None,
                    help="clean only the first N not-yet-done rows, then stop (partial run, e.g. a spend estimate)")
    args = ap.parse_args()

    ensure_pre_clean()
    df = pd.read_parquet(PRE_CLEAN)
    clean_corpus(df, limit=args.limit)


def ensure_pre_clean():
    """Import + transform the whole corpus only if pre_clean_data.parquet is missing."""
    if PRE_CLEAN.exists():
        print(f"[build] pre_clean present -> {PRE_CLEAN} (skipping import)")
    else:
        print("[build] pre_clean missing -> running import + transform (orchestrator)")
        orchestrator.run()


def clean_corpus(df, limit=None):
    """Clean each not-yet-done row through the harness, checkpointing after every row so a crash
    (or a --limit run) resumes exactly where it left off. A per-row error is logged and skipped —
    the row is not checkpointed, so it is retried on the next run rather than killing the whole run.
    The final parquet is written ONLY when every row is done: the original corpus with an added
    `validated_text` column (the cleaned text; the original `text` column is left untouched)."""
    done = _completed()                                  # id -> validated_text (already cleaned)
    todo = df[~df["id"].isin(done)]
    if limit is not None:
        todo = todo.head(limit)
    print(f"[build] {len(done)} already done | cleaning {len(todo)} of {len(df)} rows (serial)")

    harness = Harness()
    with open(CHECKPOINT, "a", encoding="utf-8") as ckpt:
        for n, (_, row) in enumerate(todo.iterrows(), 1):
            try:
                validated = harness.clean_row(row)["text"]
            except Exception as e:
                print(f"[build] ERROR row {row['id']}: {type(e).__name__}: {e} — skipping (retries on resume)")
                continue
            ckpt.write(json.dumps({"id": row["id"], "validated_text": validated}, ensure_ascii=False) + "\n")
            ckpt.flush()
            os.fsync(ckpt.fileno())
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}")

    finalize_if_complete(df)


def finalize_if_complete(df):
    """Write the final cleaned_data.parquet iff every row has been cleaned; otherwise report progress."""
    done = _completed()
    remaining = len(df) - len(done)
    if remaining > 0:
        print(f"[build] partial: {len(done)}/{len(df)} cleaned, {remaining} remaining. "
              f"Run again with no --limit to resume and complete.")
        return
    out = df.copy()
    out["validated_text"] = out["id"].map(done)
    out.to_parquet(CLEANED, index=False)
    print(f"[build] COMPLETE: {len(out)} rows -> {CLEANED}  (original 'text' kept; cleaned text in 'validated_text')")


def _completed():
    """id -> validated_text for every row already in the checkpoint."""
    done = {}
    if os.path.exists(CHECKPOINT):
        for line in open(CHECKPOINT, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    done[rec["id"]] = rec["validated_text"]
                except Exception:
                    pass
    return done


if __name__ == "__main__":
    main()
