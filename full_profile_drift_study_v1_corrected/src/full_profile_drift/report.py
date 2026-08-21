"""Render the corrected decision from frozen, analysis-only remediation tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .remediation import write_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    analysis = output / "analysis_outputs"
    required = {
        "main effects": analysis / "main_effect_permutation_summary.csv",
        "fixed subsets": analysis / "fixed_subset_permutation_tests.csv",
        "exposure": output / "tables" / "exposure_relationships_by_representation.csv",
    }
    missing = [label for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Corrected reporting requires the remediation tables first; missing: " + ", ".join(missing)
        )
    write_decision(
        output,
        pd.read_csv(required["main effects"]),
        pd.read_csv(required["fixed subsets"]),
        pd.read_csv(required["exposure"]),
    )


if __name__ == "__main__":
    main()
