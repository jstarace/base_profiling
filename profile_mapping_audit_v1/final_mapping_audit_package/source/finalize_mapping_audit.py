"""Validate existing audit outputs and assemble the compact terminal package.

This script does not recompute distributions, mappings, permutations, or any
model result. It only validates already-generated audit files, copies the
minimal audit record, and hashes the package.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = Path(__file__).resolve().parent
PACKAGE = AUDIT / "final_mapping_audit_package"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    mapping = json.loads((AUDIT / "mapping_audit.json").read_text())
    continuous = json.loads((AUDIT / "continuous_target_distribution_audit.json").read_text())
    terminal = json.loads((ROOT / "temp_explore/activation_analysis_v1/analysis_outputs/activation_decision.json").read_text())
    threshold = pd.read_csv(AUDIT / "threshold_audit.csv")
    targets = pd.read_csv(AUDIT / "target_recomputation_all_32.csv")
    distributions = pd.read_csv(AUDIT / "continuous_target_distributions_all_32.csv")
    representatives = pd.read_csv(AUDIT / "representative_training_rows.csv")
    adapter_verification = json.loads((ROOT / "adapter_verification/adapter_verification.json").read_text())
    behavioral_manifest = json.loads((ROOT / "temp_explore/phase_two_benchmark_remediation/benchmark_manifest.json").read_text())
    activation_manifest = json.loads((ROOT / "temp_explore/activation_analysis_v1/study_manifest.json").read_text())

    assert mapping["full_dataset_ptype_mismatches"] == 0
    assert mapping["threshold_rule"] == "raw_trait > 50"
    assert mapping["raw_column_order"] == ["O", "C", "E", "A", "N", "ptype"]
    assert mapping["intended_formula"] == "ptype = 16*O_high + 8*C_high + 4*E_high + 2*A_high + N_high"
    assert mapping["pilot_decode_OCEAN"] == {
        "ptype_0": "00000", "ptype_31": "11111", "ptype_9": "01001", "ptype_23": "10111"
    }
    assert threshold.strict_gt_50_mismatches.eq(0).all()
    assert len(targets) == 32 and targets.exact_class_and_bits_match.all()
    assert targets.maximum_centroid_stat_abs_diff.max() < 1e-12
    assert len(distributions) == 160 and distributions.groupby("ptype").trait.nunique().eq(5).all()
    assert distributions.row_centroid_recompute_abs_diff.max() < 1e-12
    assert distributions.unique_centroid_recompute_abs_diff.max() < 1e-12
    assert representatives.match_status.eq("MATCH").all()
    assert "std(ddof=0)" in (AUDIT / "audit_profile_mapping.py").read_text()
    assert "std(ddof=0)" in (AUDIT / "continuous_target_distribution_audit.py").read_text()
    assert adapter_verification["discovered_ptypes"] == list(range(32))
    assert adapter_verification["expected_ptypes"] == list(range(32))
    assert adapter_verification["missing_ptypes"] == [] and adapter_verification["extra_ptypes"] == []
    assert behavioral_manifest["adapter_weight_sha256"] == activation_manifest["adapter_weight_sha256"]
    assert terminal["category"] == "B"
    assert mapping["decision"] == continuous["decision"] == "RETAIN_CATEGORY_B"

    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    (PACKAGE / "source").mkdir(parents=True)
    (PACKAGE / "outputs").mkdir()

    sources = {
        AUDIT / "audit_profile_mapping.py": "source/audit_profile_mapping.py",
        AUDIT / "continuous_target_distribution_audit.py": "source/continuous_target_distribution_audit.py",
        AUDIT / "finalize_mapping_audit.py": "source/finalize_mapping_audit.py",
        ROOT / "dataset_construction/data/importers/import_tan_pandora.py": "source/import_tan_pandora.py",
        ROOT / "training/train_ocean_adapters.py": "source/train_ocean_adapters.py",
        ROOT / "training/ocean_partition.json": "source/ocean_partition.json",
        ROOT / "temp_explore/phase_two_benchmark_remediation/scripts/analyze_interface_pilot.py": "source/analyze_interface_pilot.py",
        ROOT / "temp_explore/activation_analysis_v1/scripts/analyze_activations.py": "source/analyze_activations.py",
    }
    outputs = {
        AUDIT / "FINAL_CONCLUSION.md": "FINAL_CONCLUSION.md",
        AUDIT / "mapping_audit.md": "outputs/mapping_audit.md",
        AUDIT / "mapping_audit.json": "outputs/mapping_audit.json",
        AUDIT / "continuous_target_distribution_audit.md": "outputs/continuous_target_distribution_audit.md",
        AUDIT / "continuous_target_distribution_audit.json": "outputs/continuous_target_distribution_audit.json",
        AUDIT / "threshold_audit.csv": "outputs/threshold_audit.csv",
        AUDIT / "representative_training_rows.csv": "outputs/representative_training_rows.csv",
        AUDIT / "target_recomputation_all_32.csv": "outputs/target_recomputation_all_32.csv",
        AUDIT / "continuous_target_distributions_all_32.csv": "outputs/continuous_target_distributions_all_32.csv",
        AUDIT / "continuous_target_distributions_pilot.csv": "outputs/continuous_target_distributions_pilot.csv",
        AUDIT / "ptype_0_vs_ptype_31_training_separation.csv": "outputs/ptype_0_vs_ptype_31_training_separation.csv",
        AUDIT / "within_between_class_variance.csv": "outputs/within_between_class_variance.csv",
        AUDIT / "pilot_pairwise_continuous_comparisons.csv": "outputs/pilot_pairwise_continuous_comparisons.csv",
        AUDIT / "pilot_pairwise_distance_correlations.csv": "outputs/pilot_pairwise_distance_correlations.csv",
        ROOT / "temp_explore/phase_two_benchmark_remediation/ocean_profile_targets.csv": "outputs/frozen_ocean_profile_targets.csv",
        ROOT / "temp_explore/activation_analysis_v1/analysis_outputs/activation_decision.json": "outputs/terminal_activation_decision.json",
    }
    for source, relative in (sources | outputs).items():
        destination = PACKAGE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    pilot_hashes = behavioral_manifest["adapter_weight_sha256"]
    manifest = {
        "package": "final_mapping_audit_package",
        "audit_only_standard_deviation_ddof": 0,
        "raw_rows": 3_006_566,
        "threshold_mismatches": 0,
        "threshold": "strict >50",
        "trait_order": ["O", "C", "E", "A", "N"],
        "ptype_formula": mapping["intended_formula"],
        "all_32_class_target_correspondence": True,
        "adapter_directories": [f"/workspace/adapters/ptype_{ptype}" for ptype in range(32)],
        "pilot_adapter_weight_sha256": pilot_hashes,
        "pilot_mapping": mapping["pilot_decode_OCEAN"],
        "maximum_target_statistic_abs_difference": float(targets.maximum_centroid_stat_abs_diff.max()),
        "continuous_distribution_rows": len(distributions),
        "ptype_0_vs_ptype_31": continuous["ptype_0_vs_ptype_31"],
        "terminal_activation_decision": {
            "category": "B",
            "label": "adapter-specific but non-OCEAN-aligned structure",
            "changed_by_mapping_audit": False,
        },
        "operations_rerun": {
            "model_inference": False,
            "activation_capture": False,
            "questionnaires": False,
            "permutation_analyses": False,
            "adapter_training": False,
        },
    }
    (PACKAGE / "final_mapping_audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    files = sorted(path for path in PACKAGE.rglob("*") if path.is_file() and path.name != "PACKAGE_CHECKSUMS.sha256")
    (PACKAGE / "PACKAGE_CHECKSUMS.sha256").write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(PACKAGE).as_posix()}" for path in files) + "\n"
    )
    print(json.dumps({"package": str(PACKAGE), "files": len(files) + 1, "status": "PASS"}))


if __name__ == "__main__":
    main()
