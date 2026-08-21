"""Build the frozen all-32 ptype catalog from verified legacy evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from full_profile_drift.io import atomic_json

TRAITS = tuple("OCEAN")
NAMES = {"O":"Openness","C":"Conscientiousness","E":"Extraversion","A":"Agreeableness","N":"Neuroticism"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_bits(ptype: int) -> dict[str, int]:
    return {trait: (ptype // weight) % 2 for trait, weight in zip(TRAITS, (16, 8, 4, 2, 1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    targets = pd.read_csv(args.targets).set_index("ptype")
    verification = json.loads(args.verification.read_text())
    adapters = {int(row["ptype"]): row for row in verification["adapters"]}
    rows = []
    for ptype in range(32):
        bits = profile_bits(ptype)
        verified = adapters[ptype]
        updates = verified["structural"]["effective_updates"]
        frobenius = float(np.sqrt(sum(float(item["frobenius_norm"]) ** 2 for item in updates)))
        spectral = float(np.sqrt(sum(float(item["spectral_norm"]) ** 2 for item in updates)))
        row = {
            "ptype": ptype,
            "adapter": f"ptype_{ptype}",
            "ocean_bit_string": "".join(str(bits[t]) for t in TRAITS),
            "human_readable_profile": "; ".join(f"{NAMES[t]} {'high' if bits[t] else 'low'}" for t in TRAITS),
            **{f"{t}_high": bits[t] for t in TRAITS},
            "row_count": int(targets.loc[ptype, "row_count"]),
            "unique_trait_tuple_proxy_count": int(targets.loc[ptype, "unique_trait_tuple_count"]),
            "nominal_optimizer_steps": int(targets.loc[ptype, "nominal_optimizer_steps"]),
            "effective_update_frobenius_norm": frobenius,
            "effective_update_spectral_aggregate_norm": spectral,
            "adapter_sha256": sha256(args.adapter_root / f"ptype_{ptype}" / "adapter_model.safetensors"),
        }
        for trait in TRAITS:
            row[f"row_weighted_{trait}_centroid"] = targets.loc[ptype, f"row_weighted_{trait}_mean"]
            row[f"unique_tuple_weighted_{trait}_centroid"] = targets.loc[ptype, f"unique_trait_tuple_weighted_{trait}_mean"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_root / "ptype_catalog.csv", index=False)
    lines = ["# Ptype catalog", "", "Encoding: `16O + 8C + 4E + 2A + N`; factorial coding uses low=-1, high=+1.", "",
             "| adapter | profile | rows | unique tuples | nominal steps | update F-norm | hash |", "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['adapter']} ({row['ocean_bit_string']}) | {row['human_readable_profile']} | {row['row_count']:,} | {row['unique_trait_tuple_proxy_count']} | {row['nominal_optimizer_steps']} | {row['effective_update_frobenius_norm']:.6f} | `{row['adapter_sha256']}` |")
    (args.output_root / "ptype_catalog.md").write_text("\n".join(lines) + "\n")
    atomic_json(args.output_root / "progress.json", {"project":"full_profile_drift_study_v1","stage":"catalog_created","completed_adapters":[],"integrity_failure":None})


if __name__ == "__main__":
    main()
