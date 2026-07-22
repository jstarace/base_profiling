"""Build the cleaned corpus: import if needed, then run the harness over it."""
import os
import sys
import time
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import paths  # noqa: F401
import pandas as pd

from dataset_construction.data import orchestrator
from dataset_construction.cleaning.harness import Harness

PROCESSED_DIR = paths.ROOT / "dataset_construction" / "data" / "processed_data"
PRE_CLEAN = orchestrator.PROCESSED                       # data/processed_data/pre_clean_data.parquet
CLEANED = PROCESSED_DIR / "cleaned_data.parquet"
CHECKPOINT = PROCESSED_DIR / "cleaned_data.checkpoint.jsonl"

WORKERS = 40                    # concurrent rows in flight; panel-provider limits gate this
BATCH_SIZE = 95_000             # rows per batch, then pause (stays under Gemini's 100k/day cap)
RETRIES = 4                     # per-row backoff attempts on a transient error


def main():
    ap = argparse.ArgumentParser(description="Build the cleaned corpus (import if needed, then clean).")
    ap.add_argument("--limit", type=int, default=None,
                    help="clean only the first N not-yet-done rows in a single pass, no batch pause (tests/estimates)")
    args = ap.parse_args()

    ensure_pre_clean()
    df = pd.read_parquet(PRE_CLEAN)
    if args.limit is not None:
        run_batch(df, todo_rows(df).head(args.limit))
        finalize_if_complete(df)
    else:
        run_batched(df)


def ensure_pre_clean():
    """Import + transform the whole corpus only if pre_clean_data.parquet is missing."""
    if PRE_CLEAN.exists():
        print(f"[build] pre_clean present -> {PRE_CLEAN} (skipping import)")
    else:
        print("[build] pre_clean missing -> running import + transform (orchestrator)")
        orchestrator.run()


def run_batched(df):
    """Clean the corpus in batches of BATCH_SIZE (WORKERS threads each), pausing for
    [C]ontinue / [Q]uit between batches. Resume-aware; the final parquet is written only once
    every row is done."""
    while True:
        todo = todo_rows(df)
        if len(todo) == 0:
            break
        batch = todo.head(BATCH_SIZE)
        done_before = len(df) - len(todo)
        print(f"[build] batch: cleaning {len(batch)} rows "
              f"({done_before}/{len(df)} already done) with {WORKERS} workers")
        completed = run_batch(df, batch)
        remaining = len(todo_rows(df))
        if remaining == 0:
            break
        if completed == 0:
            print(f"[build] no progress this batch — {remaining} rows keep failing. Stopping; investigate.")
            return
        if not prompt_continue(len(df) - remaining, len(df)):
            print(f"[build] quit — {len(df) - remaining}/{len(df)} done, {remaining} remaining. Re-run to resume.")
            return
    finalize_if_complete(df)


def run_batch(df, batch):
    """Clean `batch` concurrently; checkpoint each row as it completes. Checkpoint writes happen
    in this (single) thread as futures land, so they never interleave. Returns the number of
    rows completed (0 => the whole batch failed)."""
    harness = Harness()
    total = len(batch)
    done_n = 0
    with open(CHECKPOINT, "a", encoding="utf-8") as ckpt, \
            ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_clean_with_retry, harness, row): row["id"] for _, row in batch.iterrows()}
        for fut in as_completed(futs):
            validated = fut.result()                     # None if the row ultimately failed
            if validated is None:
                continue                                 # not checkpointed -> retried on resume
            ckpt.write(json.dumps({"id": futs[fut], "validated_text": validated}, ensure_ascii=False) + "\n")
            ckpt.flush()
            os.fsync(ckpt.fileno())
            done_n += 1
            if done_n % 100 == 0 or done_n == total:
                print(f"  {done_n}/{total}")
    return done_n


def _clean_with_retry(harness, row):
    """`clean_row` with exponential backoff on transient errors. Returns the validated text, or
    None if every attempt failed (skipped -> retried on the next run)."""
    delay = 2
    for attempt in range(RETRIES):
        try:
            return harness.clean_row(row)["text"]
        except Exception as e:
            if attempt == RETRIES - 1:
                print(f"[build] ERROR row {row['id']}: {type(e).__name__}: {e} — skipping (retries on resume)")
                return None
            time.sleep(delay)
            delay *= 2
    return None


def prompt_continue(done, total):
    """[C]ontinue to the next batch or [Q]uit. Returns True to continue."""
    print(f"\n[build] {done}/{total} rows done. [C]ontinue to next {BATCH_SIZE:,}, or [Q]uit? ", end="", flush=True)
    key = _getch().lower()
    print(key)
    return key == "c"


def finalize_if_complete(df):
    """Write the final cleaned_data.parquet iff every row is cleaned; otherwise report progress."""
    done = _completed()
    remaining = len(df) - len(done)
    if remaining > 0:
        print(f"[build] partial: {len(done)}/{len(df)} cleaned, {remaining} remaining. Re-run to resume.")
        return
    out = df.copy()
    out["validated_text"] = out["id"].map(done)
    out.to_parquet(CLEANED, index=False)
    print(f"[build] COMPLETE: {len(out)} rows -> {CLEANED}  (original 'text' kept; cleaned text in 'validated_text')")


def todo_rows(df):
    """Rows not yet in the checkpoint, in file order."""
    return df[~df["id"].isin(_completed())]


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


def _getch():
    try:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch
    except Exception:
        line = sys.stdin.readline()
        return line[:1] if line else "q"


if __name__ == "__main__":
    main()
