"""Read-only continuous-target distribution and geometry audit."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
TRAITS = tuple("OCEAN")
PILOT = (0, 31, 9, 23)

RAW = ROOT / "dataset_construction/data/raw_data/tan_pandora.parquet"
TARGETS = ROOT / "temp_explore/phase_two_benchmark_remediation/ocean_profile_targets.csv"
BEHAVIOR = ROOT / "temp_explore/phase_two_benchmark_remediation/interface_pilot_analysis/interface_domain_scores.csv"
ACTIVATION_PAIR = ROOT / "temp_explore/activation_analysis_v1/analysis_outputs/pairwise_distances.csv"
EFFECT = ROOT / "temp_explore/activation_analysis_v1/analysis_outputs/effect_magnitude.csv"


def corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, rank: bool = False) -> float:
    a = pd.Series(np.asarray(x, dtype=float))
    b = pd.Series(np.asarray(y, dtype=float))
    if rank:
        a, b = a.rank(), b.rank()
    return float(a.corr(b))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def main() -> None:
    raw = pd.read_parquet(RAW, columns=[*TRAITS, "ptype"])
    targets = pd.read_csv(TARGETS).set_index("ptype")

    distribution_rows: list[dict[str, object]] = []
    centroid_vectors: dict[int, np.ndarray] = {}
    unique_vectors: dict[int, np.ndarray] = {}
    within_rms: dict[int, float] = {}
    for ptype, group in raw.groupby("ptype", sort=True):
        unique = group[list(TRAITS)].drop_duplicates()
        centroid_vectors[int(ptype)] = group[list(TRAITS)].mean().to_numpy(dtype=float)
        unique_vectors[int(ptype)] = unique[list(TRAITS)].mean().to_numpy(dtype=float)
        within_rms[int(ptype)] = float(np.sqrt(sum(group[trait].var(ddof=0) for trait in TRAITS)))
        for trait in TRAITS:
            values = group[trait]
            quantiles = values.quantile([0.05, 0.25, 0.75, 0.95])
            distribution_rows.append({
                "ptype": int(ptype),
                "adapter": f"ptype_{int(ptype)}",
                "pilot_adapter": int(ptype) in PILOT,
                "trait": trait,
                "training_row_count": len(group),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "standard_deviation_population": float(values.std(ddof=0)),
                "percentile_05": float(quantiles.loc[0.05]),
                "percentile_25": float(quantiles.loc[0.25]),
                "percentile_75": float(quantiles.loc[0.75]),
                "percentile_95": float(quantiles.loc[0.95]),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "row_weighted_centroid": float(targets.loc[ptype, f"row_weighted_{trait}_mean"]),
                "unique_trait_tuple_weighted_centroid": float(targets.loc[ptype, f"unique_trait_tuple_weighted_{trait}_mean"]),
                "row_centroid_recompute_abs_diff": abs(float(values.mean()) - float(targets.loc[ptype, f"row_weighted_{trait}_mean"])),
                "unique_centroid_recompute_abs_diff": abs(float(unique[trait].mean()) - float(targets.loc[ptype, f"unique_trait_tuple_weighted_{trait}_mean"])),
            })
    distributions = pd.DataFrame(distribution_rows)
    distributions.to_csv(OUT / "continuous_target_distributions_all_32.csv", index=False)
    distributions[distributions.pilot_adapter].to_csv(OUT / "continuous_target_distributions_pilot.csv", index=False)

    # All-class within/between audit.
    between_rows = []
    all_ptypes = sorted(centroid_vectors)
    for ptype in all_ptypes:
        distances = [euclidean(centroid_vectors[ptype], centroid_vectors[other]) for other in all_ptypes if other != ptype]
        nearest = min(distances)
        between_rows.append({
            "ptype": ptype,
            "adapter": f"ptype_{ptype}",
            "pilot_adapter": ptype in PILOT,
            "training_row_count": int((raw.ptype == ptype).sum()),
            "within_class_rms_distance_from_row_centroid": within_rms[ptype],
            "nearest_other_row_centroid_distance": nearest,
            "mean_other_row_centroid_distance": float(np.mean(distances)),
            "within_rms_over_nearest_between": within_rms[ptype] / nearest,
            "within_rms_exceeds_nearest_between": within_rms[ptype] > nearest,
        })
    within_between = pd.DataFrame(between_rows)
    within_between.to_csv(OUT / "within_between_class_variance.csv", index=False)

    activation = pd.read_csv(ACTIVATION_PAIR)
    effect = pd.read_csv(EFFECT)
    behavior = pd.read_csv(BEHAVIOR)
    behavior_vectors = {
        (interface, int(str(model).removeprefix("ptype_"))): frame.set_index("trait").loc[list(TRAITS), "observed_0_100"].to_numpy(dtype=float)
        for (interface, model), frame in behavior[behavior.model_key.str.startswith("ptype_")].groupby(["interface", "model_key"])
    }
    interfaces = sorted(behavior.interface.unique())

    pair_rows: list[dict[str, object]] = []
    for ptype_a, ptype_b in itertools.combinations(PILOT, 2):
        adapter_a, adapter_b = f"ptype_{ptype_a}", f"ptype_{ptype_b}"
        pair_activation = activation[(activation.adapter_a == adapter_a) & (activation.adapter_b == adapter_b)]
        if pair_activation.empty:
            pair_activation = activation[(activation.adapter_a == adapter_b) & (activation.adapter_b == adapter_a)]
        effect_a = effect[effect.adapter == adapter_a]
        effect_b = effect[effect.adapter == adapter_b]
        middle_a = effect_a[effect_a.layer.between(8, 24)]
        middle_b = effect_b[effect_b.layer.between(8, 24)]
        binary_a = np.array([(ptype_a // weight) % 2 for weight in (16, 8, 4, 2, 1)], dtype=float) * 100
        binary_b = np.array([(ptype_b // weight) % 2 for weight in (16, 8, 4, 2, 1)], dtype=float) * 100
        base_record = {
            "adapter_a": adapter_a,
            "adapter_b": adapter_b,
            "row_weighted_training_centroid_distance": euclidean(centroid_vectors[ptype_a], centroid_vectors[ptype_b]),
            "unique_tuple_training_centroid_distance": euclidean(unique_vectors[ptype_a], unique_vectors[ptype_b]),
            "binary_0_100_corner_distance_for_context_only": euclidean(binary_a, binary_b),
            "activation_centroid_distance_mean_all_conditions_layers": float(pair_activation.centroid_distance.mean()),
            "activation_centroid_distance_median_all_conditions_layers": float(pair_activation.centroid_distance.median()),
            "activation_prompt_pair_distance_mean_all_conditions_layers": float(pair_activation.mean_prompt_pair_distance.mean()),
            "activation_prompt_pair_distance_median_middle_layers_8_24": float(pair_activation[pair_activation.layer.between(8, 24)].mean_prompt_pair_distance.median()),
            "adapter_a_effect_mean_delta_l2_all_conditions_layers": float(effect_a.mean_delta_l2.mean()),
            "adapter_b_effect_mean_delta_l2_all_conditions_layers": float(effect_b.mean_delta_l2.mean()),
            "adapter_effect_magnitude_absolute_difference": abs(float(effect_a.mean_delta_l2.mean()) - float(effect_b.mean_delta_l2.mean())),
            "adapter_effect_magnitude_rss": float(np.hypot(effect_a.mean_delta_l2.mean(), effect_b.mean_delta_l2.mean())),
            "adapter_a_effect_median_middle_layers_8_24": float(middle_a.mean_delta_l2.median()),
            "adapter_b_effect_median_middle_layers_8_24": float(middle_b.mean_delta_l2.median()),
            "pooled_within_class_rms": float(np.sqrt((within_rms[ptype_a] ** 2 + within_rms[ptype_b] ** 2) / 2)),
        }
        base_record["pooled_within_rms_over_row_centroid_separation"] = base_record["pooled_within_class_rms"] / base_record["row_weighted_training_centroid_distance"]
        for interface in interfaces:
            record = dict(base_record)
            record["behavioral_interface"] = interface
            record["behavioral_ocean_score_distance"] = euclidean(behavior_vectors[(interface, ptype_a)], behavior_vectors[(interface, ptype_b)])
            pair_rows.append(record)
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(OUT / "pilot_pairwise_continuous_comparisons.csv", index=False)

    # Six-pair correlations are descriptive and severely underpowered.
    correlation_rows = []
    invariant_columns = {
        "activation_centroid": "activation_centroid_distance_median_all_conditions_layers",
        "activation_prompt_middle": "activation_prompt_pair_distance_median_middle_layers_8_24",
        "effect_absolute_difference": "adapter_effect_magnitude_absolute_difference",
        "effect_rss": "adapter_effect_magnitude_rss",
    }
    one_per_pair = pairwise.drop_duplicates(["adapter_a", "adapter_b"])
    for target_name, target_column in (
        ("row_weighted", "row_weighted_training_centroid_distance"),
        ("unique_tuple_weighted", "unique_tuple_training_centroid_distance"),
    ):
        for comparison_name, comparison_column in invariant_columns.items():
            correlation_rows.append({
                "training_target": target_name,
                "comparison": comparison_name,
                "behavioral_interface": None,
                "pair_count": 6,
                "pearson_r": corr(one_per_pair[target_column], one_per_pair[comparison_column]),
                "spearman_r": corr(one_per_pair[target_column], one_per_pair[comparison_column], rank=True),
            })
        for interface, frame in pairwise.groupby("behavioral_interface"):
            correlation_rows.append({
                "training_target": target_name,
                "comparison": "behavioral_ocean_score_distance",
                "behavioral_interface": interface,
                "pair_count": 6,
                "pearson_r": corr(frame[target_column], frame.behavioral_ocean_score_distance),
                "spearman_r": corr(frame[target_column], frame.behavioral_ocean_score_distance, rank=True),
            })
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(OUT / "pilot_pairwise_distance_correlations.csv", index=False)

    endpoint_trait_rows = []
    for index, trait in enumerate(TRAITS):
        endpoint_trait_rows.append({
            "trait": trait,
            "ptype_0_row_weighted_centroid": centroid_vectors[0][index],
            "ptype_31_row_weighted_centroid": centroid_vectors[31][index],
            "ptype_31_minus_ptype_0": centroid_vectors[31][index] - centroid_vectors[0][index],
            "ptype_0_standard_deviation": float(raw[raw.ptype == 0][trait].std(ddof=0)),
            "ptype_31_standard_deviation": float(raw[raw.ptype == 31][trait].std(ddof=0)),
        })
    endpoint_traits = pd.DataFrame(endpoint_trait_rows)
    endpoint_traits.to_csv(OUT / "ptype_0_vs_ptype_31_training_separation.csv", index=False)

    endpoint_pair = pairwise[(pairwise.adapter_a == "ptype_0") & (pairwise.adapter_b == "ptype_31")].iloc[0]
    target_use = {
        "behavioral": {
            "continuous_centroids_loaded": True,
            "continuous_targets": ["row_weighted", "unique_tuple_weighted"],
            "authorization_continuous_primary_criterion": "positive row-weighted continuous-target correlation",
            "binary_role": "Binary bits were also used for target-direction signs, endpoint sign expectations, and Hamming diagnostics; therefore not every behavioral alignment check was continuous.",
            "binary_encoded_as_0_100_for_primary_test": False,
            "verdict": "MIXED: continuous row-weighted correlation was the named target-correlation gate, but binary direction tests were co-primary authorization conditions.",
        },
        "activation": {
            "continuous_centroids_loaded": True,
            "continuous_targets": ["row_weighted", "unique_tuple_weighted"],
            "binary_target_also_reported": True,
            "category_a_decision_filter": "row_weighted and unique_tuple_weighted only",
            "binary_encoded_as_0_100_for_primary_test": False,
            "verdict": "PASS: the Category A target-alignment decision explicitly excluded the binary target and required corrected continuous-target support.",
        },
    }
    summary = {
        "audit_type": "continuous_target_distribution_read_only",
        "model_experiments_rerun": False,
        "all_ptypes_reported": len(distributions.ptype.unique()),
        "distribution_rows": len(distributions),
        "pilot_pair_interface_rows": len(pairwise),
        "target_use_verification": target_use,
        "ptype_0_vs_ptype_31": {
            "row_weighted_centroid_distance": float(endpoint_pair.row_weighted_training_centroid_distance),
            "unique_tuple_weighted_centroid_distance": float(endpoint_pair.unique_tuple_training_centroid_distance),
            "binary_0_100_corner_distance": float(endpoint_pair.binary_0_100_corner_distance_for_context_only),
            "pooled_within_class_rms": float(endpoint_pair.pooled_within_class_rms),
            "pooled_within_rms_over_row_centroid_separation": float(endpoint_pair.pooled_within_rms_over_row_centroid_separation),
        },
        "within_class_limitation": {
            "classes_within_rms_exceeds_nearest_centroid_separation": int(within_between.within_rms_exceeds_nearest_between.sum()),
            "total_classes": len(within_between),
            "pilot_classes_within_rms_exceeds_nearest_centroid_separation": int(within_between[within_between.pilot_adapter].within_rms_exceeds_nearest_between.sum()),
            "pilot_class_count": len(PILOT),
            "interpretation": "When within-class RMS exceeds nearest between-class centroid separation, thresholded classes overlap substantially in continuous trait space; binary class identity is not an extreme-point guarantee.",
        },
        "decision": "RETAIN_CATEGORY_B",
    }
    (OUT / "continuous_target_distribution_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
