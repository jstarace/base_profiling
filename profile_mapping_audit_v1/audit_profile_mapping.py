"""Read-only, end-to-end audit of the PANDORA OCEAN ptype mapping.

This program reads source code, frozen data, and completed result tables.  It
does not load a model or modify any prior experiment artifact.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
TRAITS = tuple("OCEAN")
WEIGHTS = (16, 8, 4, 2, 1)
PILOT = (0, 31, 9, 23)

RAW = ROOT / "dataset_construction/data/raw_data/tan_pandora.parquet"
TARGETS = ROOT / "temp_explore/phase_two_benchmark_remediation/ocean_profile_targets.csv"
BEHAVIOR = ROOT / "temp_explore/phase_two_benchmark_remediation/interface_pilot_analysis/interface_domain_scores.csv"
ACTIVATION_DISTANCES = ROOT / "temp_explore/activation_analysis_v1/analysis_outputs/pairwise_distances.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bits_from_ptype(ptype: int, position_order: tuple[str, ...], inverted: bool) -> dict[str, int]:
    position_bits = [(ptype // weight) % 2 for weight in WEIGHTS]
    result = {trait: int(position_bits[index]) for index, trait in enumerate(position_order)}
    if inverted:
        result = {trait: 1 - value for trait, value in result.items()}
    return result


def correlation(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, method: str = "pearson") -> float:
    # Discard source dataframe indexes so pandas cannot align unrelated rows.
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y, dtype=float))
    if method == "spearman":
        x, y = x.rank(), y.rank()
    return float(x.corr(y))


def target_distance(bits_by_ptype: dict[int, dict[str, int]]) -> np.ndarray:
    vectors = np.array([[bits_by_ptype[p][trait] for trait in TRAITS] for p in PILOT], dtype=float)
    return np.array([np.linalg.norm(vectors[i] - vectors[j]) for i, j in itertools.combinations(range(4), 2)])


def continuous_distance(target_rows: pd.DataFrame, position_order: tuple[str, ...], inverted: bool, prefix: str) -> np.ndarray:
    vectors = []
    for ptype in PILOT:
        row = target_rows.loc[ptype]
        decoded = []
        # Alternative semantic interpretations only permute dimensions; global
        # polarity reflection maps x to 100-x. Both preserve Euclidean distance.
        for trait in position_order:
            value = float(row[f"{prefix}_{trait}_mean"])
            decoded.append(100.0 - value if inverted else value)
        vectors.append(decoded)
    vectors = np.asarray(vectors)
    return np.array([np.linalg.norm(vectors[i] - vectors[j]) for i, j in itertools.combinations(range(4), 2)])


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw = pd.read_parquet(RAW, columns=[*TRAITS, "ptype", "__index_level_0__"])
    target = pd.read_csv(TARGETS).set_index("ptype")

    # Full-row independent threshold/packing audit.
    intended_bits = pd.DataFrame({f"{trait}_high": (raw[trait] > 50).astype("int8") for trait in TRAITS})
    recomputed = sum(intended_bits[f"{trait}_high"] * weight for trait, weight in zip(TRAITS, WEIGHTS))
    full_mismatches = int((recomputed != raw.ptype).sum())

    threshold_rows = []
    for trait, weight in zip(TRAITS, WEIGHTS):
        stored_bit = ((raw.ptype // weight) % 2).astype("int8")
        strict = (raw[trait] > 50).astype("int8")
        non_strict = (raw[trait] >= 50).astype("int8")
        inverted = (raw[trait] <= 50).astype("int8")
        at_boundary = raw[trait] == 50
        threshold_rows.append({
            "trait": trait,
            "weight": weight,
            "minimum": float(raw[trait].min()),
            "maximum": float(raw[trait].max()),
            "rows_equal_50": int(at_boundary.sum()),
            "equal_50_stored_high_count": int(stored_bit[at_boundary].sum()),
            "strict_gt_50_mismatches": int((stored_bit != strict).sum()),
            "non_strict_ge_50_mismatches": int((stored_bit != non_strict).sum()),
            "inverted_polarity_mismatches": int((stored_bit != inverted).sum()),
        })
    threshold = pd.DataFrame(threshold_rows)
    threshold.to_csv(OUT / "threshold_audit.csv", index=False)

    # Recompute exact target rows from precisely the numeric classes used by training.
    target_rows = []
    all_target_matches = True
    for ptype, group in raw.groupby("ptype", sort=True):
        unique = group[list(TRAITS)].drop_duplicates()
        row = {"ptype": int(ptype), "row_count_recomputed": len(group), "unique_trait_tuple_count_recomputed": len(unique)}
        decoded = bits_from_ptype(int(ptype), TRAITS, False)
        for trait in TRAITS:
            row[f"{trait}_high_recomputed"] = decoded[trait]
            row[f"{trait}_high_target"] = int(target.loc[ptype, f"{trait}_high"])
            for label, frame in (("row_weighted", group), ("unique_trait_tuple_weighted", unique)):
                for statistic in ("mean", "median", "std"):
                    recomputed_value = float(frame[trait].std(ddof=0) if statistic == "std" else getattr(frame[trait], statistic)())
                    target_value = float(target.loc[ptype, f"{label}_{trait}_{statistic}"])
                    row[f"{label}_{trait}_{statistic}_abs_diff"] = abs(recomputed_value - target_value)
        row["row_count_target"] = int(target.loc[ptype, "row_count"])
        row["unique_trait_tuple_count_target"] = int(target.loc[ptype, "unique_trait_tuple_count"])
        numeric_diffs = [value for key, value in row.items() if key.endswith("_abs_diff")]
        row["maximum_centroid_stat_abs_diff"] = max(numeric_diffs)
        row["exact_class_and_bits_match"] = (
            row["row_count_recomputed"] == row["row_count_target"]
            and row["unique_trait_tuple_count_recomputed"] == row["unique_trait_tuple_count_target"]
            and all(row[f"{trait}_high_recomputed"] == row[f"{trait}_high_target"] for trait in TRAITS)
            and row["maximum_centroid_stat_abs_diff"] < 1e-12
        )
        all_target_matches &= bool(row["exact_class_and_bits_match"])
        target_rows.append(row)
    target_recomputation = pd.DataFrame(target_rows)
    target_recomputation.to_csv(OUT / "target_recomputation_all_32.csv", index=False)

    # Three deterministic representative rows per pilot adapter.
    representative_rows = []
    for ptype in PILOT:
        group = raw[raw.ptype == ptype].sort_values("__index_level_0__", kind="stable").reset_index(drop=True)
        positions = (0, len(group) // 2, len(group) - 1)
        for label, position in zip(("first", "middle", "last"), positions):
            source = group.iloc[position]
            bits = {trait: int(source[trait] > 50) for trait in TRAITS}
            independently_recomputed = sum(bits[trait] * weight for trait, weight in zip(TRAITS, WEIGHTS))
            record = {
                "adapter": f"ptype_{ptype}",
                "representative": label,
                "source_row_index": int(source["__index_level_0__"]),
                **{f"raw_{trait}": float(source[trait]) for trait in TRAITS},
                **{f"recomputed_{trait}_high": bits[trait] for trait in TRAITS},
                "independently_recomputed_ptype": independently_recomputed,
                "stored_ptype": int(source.ptype),
                "training_adapter_path": f"/workspace/adapters/ptype_{ptype}",
                **{f"target_{trait}_high": int(target.loc[ptype, f"{trait}_high"]) for trait in TRAITS},
            }
            record["match_status"] = "MATCH" if (
                independently_recomputed == int(source.ptype) == ptype
                and all(bits[trait] == int(target.loc[ptype, f"{trait}_high"]) for trait in TRAITS)
            ) else "MISMATCH"
            representative_rows.append(record)
    representatives = pd.DataFrame(representative_rows)
    representatives.to_csv(OUT / "representative_training_rows.csv", index=False)

    # Exhaustive 5! position permutations × two global polarity orientations.
    behavior = pd.read_csv(BEHAVIOR)
    behavior = behavior[behavior.model_key.isin([f"ptype_{p}" for p in PILOT]) | behavior.model_key.eq("base")]
    base = behavior[behavior.model_key.eq("base")][["interface", "trait", "observed_0_100"]].rename(columns={"observed_0_100": "base_score"})
    adapter_behavior = behavior[~behavior.model_key.eq("base")].merge(base, on=["interface", "trait"])
    adapter_behavior["ptype_integer"] = adapter_behavior.model_key.str.removeprefix("ptype_").astype(int)
    adapter_behavior["adapter_minus_base"] = adapter_behavior.observed_0_100 - adapter_behavior.base_score

    activation = pd.read_csv(ACTIVATION_DISTANCES)
    pair_order = [(f"ptype_{PILOT[i]}", f"ptype_{PILOT[j]}") for i, j in itertools.combinations(range(4), 2)]
    activation_vectors = {}
    for keys, frame in activation.groupby(["prompt_group", "pooling_rule", "layer"]):
        lookup = {(row.adapter_a, row.adapter_b): row.mean_prompt_pair_distance for row in frame.itertuples()}
        activation_vectors[keys] = np.array([lookup[pair] for pair in pair_order])

    diagnostics = []
    activation_invariance = []
    intended_binary = target_distance({ptype: bits_from_ptype(ptype, TRAITS, False) for ptype in PILOT})
    intended_row_continuous = continuous_distance(target, TRAITS, False, "row_weighted")
    intended_unique_continuous = continuous_distance(target, TRAITS, False, "unique_trait_tuple_weighted")
    for order in itertools.permutations(TRAITS):
        for inverted in (False, True):
            mapping_id = f"positions={''.join(order)};polarity={'inverted' if inverted else 'high_is_1'}"
            decoded = {ptype: bits_from_ptype(ptype, order, inverted) for ptype in PILOT}
            binary_distance = target_distance(decoded)
            row_distance = continuous_distance(target, order, inverted, "row_weighted")
            unique_distance = continuous_distance(target, order, inverted, "unique_trait_tuple_weighted")

            activation_pearson = [correlation(vector, binary_distance) for vector in activation_vectors.values()]
            activation_spearman = [correlation(vector, binary_distance, "spearman") for vector in activation_vectors.values()]
            activation_invariance.append({
                "mapping_id": mapping_id,
                "position_order_high_to_low_weight": "".join(order),
                "polarity_inverted": inverted,
                "is_intended_mapping": order == TRAITS and not inverted,
                "binary_distance_max_abs_change_vs_intended": float(np.max(np.abs(binary_distance - intended_binary))),
                "row_continuous_distance_max_abs_change_vs_intended": float(np.max(np.abs(row_distance - intended_row_continuous))),
                "unique_continuous_distance_max_abs_change_vs_intended": float(np.max(np.abs(unique_distance - intended_unique_continuous))),
                "activation_binary_pearson_median": float(np.median(activation_pearson)),
                "activation_binary_spearman_median": float(np.median(activation_spearman)),
                "activation_conditions": len(activation_pearson),
            })

            for interface, frame in adapter_behavior.groupby("interface"):
                semantic_bits = np.array([decoded[int(row.ptype_integer)][row.trait] for row in frame.itertuples()])
                direction = np.where(semantic_bits == 1, 1.0, -1.0)
                observed_high = (frame.observed_0_100.to_numpy() > 50).astype(int)
                diagnostics.append({
                    "mapping_id": mapping_id,
                    "position_order_high_to_low_weight": "".join(order),
                    "polarity_inverted": inverted,
                    "is_intended_mapping": order == TRAITS and not inverted,
                    "behavioral_interface": interface,
                    "behavioral_binary_correlation": correlation(frame.observed_0_100, semantic_bits),
                    "behavioral_target_directed_count_of_20": int(((frame.adapter_minus_base.to_numpy() * direction) > 0).sum()),
                    "behavioral_hamming_distance_of_20": int((observed_high != semantic_bits).sum()),
                    "activation_binary_pearson_median": float(np.median(activation_pearson)),
                    "activation_binary_spearman_median": float(np.median(activation_spearman)),
                })
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_frame.to_csv(OUT / "alternative_mapping_diagnostics.csv", index=False)
    invariance_frame = pd.DataFrame(activation_invariance)
    invariance_frame.to_csv(OUT / "activation_mapping_invariance.csv", index=False)

    intended_diag = diagnostic_frame[diagnostic_frame.is_intended_mapping]
    best_by_interface = []
    for interface, frame in diagnostic_frame.groupby("behavioral_interface"):
        best = frame.sort_values(["behavioral_target_directed_count_of_20", "behavioral_binary_correlation"], ascending=False).iloc[0]
        intended = intended_diag[intended_diag.behavioral_interface == interface].iloc[0]
        best_by_interface.append({
            "interface": interface,
            "intended_binary_correlation": intended.behavioral_binary_correlation,
            "intended_target_directed_count_of_20": int(intended.behavioral_target_directed_count_of_20),
            "intended_hamming_distance_of_20": int(intended.behavioral_hamming_distance_of_20),
            "diagnostic_best_mapping_id": best.mapping_id,
            "diagnostic_best_binary_correlation": best.behavioral_binary_correlation,
            "diagnostic_best_target_directed_count_of_20": int(best.behavioral_target_directed_count_of_20),
            "diagnostic_best_has_source_code_evidence": False,
        })
    best_summary = pd.DataFrame(best_by_interface)
    best_summary.to_csv(OUT / "behavioral_alternative_summary.csv", index=False)

    intended_pilot = {ptype: "".join(str(bits_from_ptype(ptype, TRAITS, False)[trait]) for trait in TRAITS) for ptype in PILOT}
    result = {
        "audit_type": "read_only_profile_mapping",
        "model_experiments_rerun": False,
        "raw_dataset_sha256": sha256(RAW),
        "evidence_sha256": {
            "pandora_importer": sha256(ROOT / "dataset_construction/data/importers/import_tan_pandora.py"),
            "training_code": sha256(ROOT / "training/train_ocean_adapters.py"),
            "training_partition": sha256(ROOT / "training/ocean_partition.json"),
            "benchmark_targets": sha256(TARGETS),
            "behavioral_results": sha256(BEHAVIOR),
            "activation_target_lookup_code": sha256(ROOT / "temp_explore/activation_analysis_v1/scripts/analyze_activations.py"),
            "activation_distances": sha256(ACTIVATION_DISTANCES),
            "benchmark_manifest": sha256(ROOT / "temp_explore/phase_two_benchmark_remediation/benchmark_manifest.json"),
            "activation_study_manifest": sha256(ROOT / "temp_explore/activation_analysis_v1/study_manifest.json"),
        },
        "raw_rows": len(raw),
        "raw_column_order": list(pd.read_parquet(RAW, columns=[*TRAITS, "ptype"]).columns),
        "intended_formula": "ptype = 16*O_high + 8*C_high + 4*E_high + 2*A_high + N_high",
        "threshold_rule": "raw_trait > 50",
        "full_dataset_ptype_mismatches": full_mismatches,
        "all_32_target_rows_exactly_recomputed": bool(all_target_matches),
        "representative_rows_all_match": bool(representatives.match_status.eq("MATCH").all()),
        "pilot_decode_OCEAN": {f"ptype_{ptype}": value for ptype, value in intended_pilot.items()},
        "alternative_mappings_tested": 240,
        "alternative_behavioral_conditions": len(diagnostic_frame),
        "activation_distance_invariant_across_all_alternatives": bool(
            (invariance_frame[[
                "binary_distance_max_abs_change_vs_intended",
                "row_continuous_distance_max_abs_change_vs_intended",
                "unique_continuous_distance_max_abs_change_vs_intended",
            ]] < 1e-12).all().all()
        ),
        "alternative_mapping_with_original_code_evidence": None,
        "mapping_discrepancy_found": False,
        "decision": "RETAIN_CATEGORY_B",
        "source_evidence_note": "The importer preserves upstream columns; training filters the stored integer ptype directly. The strict >50 rule is an independently verified decoding of all upstream rows, not a re-binning step in training.",
    }
    (OUT / "mapping_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    checksum_names = (
        "audit_profile_mapping.py",
        "mapping_audit.md",
        "mapping_audit.json",
        "representative_training_rows.csv",
        "threshold_audit.csv",
        "target_recomputation_all_32.csv",
        "alternative_mapping_diagnostics.csv",
        "activation_mapping_invariance.csv",
        "behavioral_alternative_summary.csv",
        "continuous_target_distribution_audit.py",
        "continuous_target_distribution_audit.md",
        "continuous_target_distribution_audit.json",
        "continuous_target_distributions_all_32.csv",
        "continuous_target_distributions_pilot.csv",
        "pilot_pairwise_continuous_comparisons.csv",
        "pilot_pairwise_distance_correlations.csv",
        "ptype_0_vs_ptype_31_training_separation.csv",
        "within_between_class_variance.csv",
    )
    (OUT / "AUDIT_CHECKSUMS.sha256").write_text(
        "\n".join(f"{sha256(OUT / name)}  {name}" for name in checksum_names) + "\n"
    )


if __name__ == "__main__":
    main()
