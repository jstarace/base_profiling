"""Split the 32 OCEAN ptype classes into row-balanced groups, one per training pod.

Greedy bin-packing on row count: largest class first, each class to the lightest group.
Writes a committed partition file so a pod launch is reproducible.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import paths
import pandas as pd

COUNTS = paths.ROOT / "dataset_construction" / "data" / "processed_data" / "ocean_class_counts.csv"
PARTITION = paths.ROOT / "training" / "ocean_partition.json"
N_GROUPS = 3


def main():
    args = parse_args()
    df = pd.read_csv(args.counts)
    groups = pack(df, args.n_groups)

    total = int(df["rows"].sum())
    for g in groups:
        print(f"group {g['group']}: {g['n_classes']} classes, {g['rows']:,} rows "
              f"({100 * g['rows'] / total:.2f}%)")
        print(f"  ptypes: {g['ptypes']}")
    spread = max(g["rows"] for g in groups) - min(g["rows"] for g in groups)
    print(f"\ntotal {total:,} rows | spread {spread:,} rows "
          f"({100 * spread / total:.2f}% of total)")

    if args.dry_run:
        print("\n[dry-run] partition not written")
        return groups

    payload = {
        "source": str(args.counts.relative_to(paths.ROOT)),
        "base_model": "meta-llama/Llama-3.1-8B",
        "n_groups": args.n_groups,
        "total_rows": total,
        "groups": groups,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return groups


def pack(df, n_groups):
    """Greedy: classes largest-first, each to whichever group has the fewest rows so far."""
    bins = [{"group": i, "ptypes": [], "rows": 0} for i in range(n_groups)]
    for row in df.sort_values("rows", ascending=False).itertuples():
        target = min(bins, key=lambda b: b["rows"])
        target["ptypes"].append(int(row.ptype))
        target["rows"] += int(row.rows)
    for b in bins:
        b["n_classes"] = len(b["ptypes"])
    return bins


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--counts", type=Path, default=COUNTS)
    p.add_argument("--out", type=Path, default=PARTITION)
    p.add_argument("--n-groups", type=int, default=N_GROUPS)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
