"""Analysis-only correction of the completed full-profile drift study.

This module consumes only compact, saved centroids, pairwise distances, exposure
tables, and previously generated summaries.  It never imports a model runtime or
opens an adapter artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260803
TRAITS = tuple("OCEAN")
WEIGHTS = (16, 8, 4, 2, 1)
MODULES = ("down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj")
TERMS = tuple(
    subset
    for order in range(6)
    for subset in itertools.combinations(range(5), order)
)
TERM_NAMES = {
    subset: "intercept" if not subset else "x".join(TRAITS[i] for i in subset)
    for subset in TERMS
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with stable tie handling, equivalent to scipy rankdata."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray, *, ranked: bool = False) -> float:
    x = rankdata(x) if ranked else np.asarray(x, dtype=np.float64)
    y = rankdata(y) if ranked else np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denominator) if denominator else float("nan")


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.clip(adjusted, 0.0, 1.0)


def design_matrix() -> np.ndarray:
    bits = np.array(
        [[1.0 if ptype & weight else -1.0 for weight in WEIGHTS] for ptype in range(32)]
    )
    return np.column_stack(
        [np.ones(32) if not subset else np.prod(bits[:, subset], axis=1) for subset in TERMS]
    )


def representation_grams(project: Path) -> dict[str, np.ndarray]:
    analysis = project / "analysis_outputs"
    activation = []
    for path in sorted(analysis.glob("activation_centroids_*.npz")):
        centroids = np.load(path)["centroids"].astype(np.float64)
        activation.extend(centroids[:, layer] @ centroids[:, layer].T for layer in range(32))
    logits = []
    for path in sorted(analysis.glob("logit_centroids_*.npz")):
        centroids = np.load(path)["centroids"].astype(np.float64)
        logits.append(centroids @ centroids.T)
    weights = []
    for layer in range(32):
        weights.append(
            sum(
                np.load(project / "weight_geometry" / f"layer_{layer:02d}_{module}_geometry.npz")[
                    "gram"
                ]
                for module in MODULES
            )
        )
    result = {
        "activation": np.stack(activation),
        "logits": np.stack(logits),
        "weights": np.stack(weights),
    }
    assert {key: len(value) for key, value in result.items()} == {
        "activation": 192,
        "logits": 3,
        "weights": 32,
    }
    return result


def main_effect_permutation(project: Path, permutations: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce the independent shuffled-label Walsh audit from saved Gram matrices."""
    matrix = design_matrix()
    grams = representation_grams(project)
    rng = np.random.default_rng(SEED)
    label_permutations = np.stack([rng.permutation(32) for _ in range(permutations)])
    summary_rows: list[dict] = []
    term_rows: list[dict] = []
    order_rows: list[dict] = []
    term_counts = {order: sum(len(term) == order for term in TERMS) for order in range(1, 6)}

    for representation, gram in grams.items():
        denominator = np.trace(gram, axis1=1, axis2=2) / 32.0 - gram.sum(axis=(1, 2)) / 1024.0
        observed = np.einsum("pi,cpq,qi->ci", matrix, gram, matrix) / 1024.0
        observed_share = observed[:, 1:] / denominator[:, None]
        # Keep the five-column contraction used by the independent audit.  The
        # mathematically equivalent slice of the 32-column contraction differs
        # at a few ulps because BLAS chooses a different summation path.
        main_design = matrix[:, 1:6]
        observed_main = np.einsum("pi,cpq,qi->c", main_design, gram, main_design) / 1024.0 / denominator
        null_main = []
        null_terms = []
        for start in range(0, permutations, 100):
            current_permutations = label_permutations[start : start + 100]
            permuted_design = matrix[current_permutations]
            permuted_energy = (
                np.einsum("bpi,cpq,bqi->bci", permuted_design, gram, permuted_design) / 1024.0
            )
            permuted_share = permuted_energy[:, :, 1:] / denominator[None, :, None]
            permuted_main = main_design[current_permutations]
            null_main.extend(
                np.median(
                    np.einsum("bpi,cpq,bqi->bc", permuted_main, gram, permuted_main)
                    / 1024.0
                    / denominator[None, :],
                    axis=1,
                )
            )
            null_terms.extend(np.median(permuted_share, axis=1))
        null_main = np.asarray(null_main)
        null_terms = np.asarray(null_terms)
        observed_median = float(np.median(observed_main))
        summary_rows.append(
            {
                "representation": representation,
                "conditions": len(gram),
                "observed_median": observed_median,
                "null_median_mean": float(null_main.mean()),
                "null_median_sd": float(null_main.std(ddof=0)),
                "null_median_q025": float(np.quantile(null_main, 0.025)),
                "null_median_q975": float(np.quantile(null_main, 0.975)),
                "p_upper": (1 + int(np.sum(null_main >= observed_median))) / (permutations + 1),
                "z": float((observed_median - null_main.mean()) / null_main.std(ddof=0)),
            }
        )
        pvalues = []
        for index, subset in enumerate(TERMS[1:]):
            value = float(np.median(observed_share[:, index]))
            null = null_terms[:, index]
            pvalue = (1 + int(np.sum(null >= value))) / (permutations + 1)
            pvalues.append(pvalue)
            term_rows.append(
                {
                    "representation": representation,
                    "term": TERM_NAMES[subset],
                    "interaction_order": len(subset),
                    "observed_median_energy_share": value,
                    "null_median_mean": float(null.mean()),
                    "null_median_sd": float(null.std(ddof=0)),
                    "p_upper": pvalue,
                }
            )
        adjusted = bh_adjust(np.asarray(pvalues))
        for row, qvalue in zip(term_rows[-31:], adjusted):
            row["q_bh_within_representation_31"] = float(qvalue)
        for order in range(1, 6):
            mask = np.array([len(term) == order for term in TERMS[1:]])
            share = observed_share[:, mask].sum(axis=1)
            order_rows.append(
                {
                    "representation": representation,
                    "interaction_order": order,
                    "term_count": term_counts[order],
                    "combinatorial_expected_share": term_counts[order] / 31.0,
                    "observed_median_total_share": float(np.median(share)),
                    "observed_mean_energy_share_per_term": float(np.mean(observed_share[:, mask])),
                }
            )
    terms_frame = pd.DataFrame(term_rows)
    terms_frame["q_bh_all_93"] = bh_adjust(terms_frame.p_upper.to_numpy())
    return pd.DataFrame(summary_rows), terms_frame, pd.DataFrame(order_rows)


def averaged_distance_matrices(project: Path) -> dict[str, np.ndarray]:
    distances = pd.read_parquet(project / "analysis_outputs" / "all_representation_pairwise_distances.parquet")
    averaged = distances.groupby(["representation", "ptype_a", "ptype_b"], as_index=False).distance.mean()
    result = {}
    for representation, frame in averaged.groupby("representation"):
        matrix = np.zeros((32, 32), dtype=np.float64)
        matrix[frame.ptype_a.astype(int), frame.ptype_b.astype(int)] = frame.distance
        result[representation] = matrix + matrix.T
    return result


def subset_definitions(project: Path) -> tuple[dict[str, set[int]], np.ndarray]:
    metrics = pd.read_csv(project / "analysis_outputs" / "profile_exposure_effect_metrics.csv").sort_values("ptype").reset_index(drop=True)
    quartile = pd.qcut(metrics.row_count, 4, labels=False, duplicates="drop")
    definitions = {
        "all_32": set(range(32)),
        "rows_ge_1000": set(metrics.index[metrics.row_count >= 1000]),
        "rows_ge_10000": set(metrics.index[metrics.row_count >= 10000]),
        "rows_ge_50000": set(metrics.index[metrics.row_count >= 50000]),
        "exposure_quartile_1": set(metrics.index[quartile == 0]),
        "exposure_quartile_4": set(metrics.index[quartile == 3]),
    }
    rows = metrics.row_count.to_numpy(dtype=np.float64)
    ratios = np.maximum.outer(rows, rows) / np.minimum.outer(rows, rows)
    return definitions, ratios


def restricted_subset_permutation(project: Path, permutations: int = 3000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact fixed-subset Mantel-style audit, with one 24-test BH family."""
    matrices = averaged_distance_matrices(project)
    catalog = pd.read_csv(project / "ptype_catalog.csv").sort_values("ptype")
    targets = catalog[[f"row_weighted_{trait}_centroid" for trait in TRAITS]].to_numpy(float)
    target_matrix = np.sqrt(np.sum((targets[:, None] - targets[None, :]) ** 2, axis=2))
    definitions, ratios = subset_definitions(project)
    rng = np.random.default_rng(SEED)
    label_permutations = np.stack([rng.permutation(32) for _ in range(permutations)])
    rows = []
    membership_rows = []
    names = list(definitions) + ["pair_row_ratio_le_1_5", "pair_row_ratio_le_2_0"]
    pair_indices = {}
    for name in names:
        if name in definitions:
            members = np.array(sorted(definitions[name]), dtype=int)
            local_a, local_b = np.triu_indices(len(members), 1)
            pair_a, pair_b = members[local_a], members[local_b]
            membership_rows.extend(
                {"subset": name, "ptype": int(ptype), "included": True} for ptype in members
            )
        else:
            limit = 1.5 if name.endswith("1_5") else 2.0
            pair_a, pair_b = np.where(np.triu(ratios <= limit, 1))
            members = np.unique(np.r_[pair_a, pair_b])
            membership_rows.extend(
                {"subset": name, "ptype": int(ptype), "included": True} for ptype in members
            )
        pair_indices[name] = (pair_a, pair_b)

    for representation in ("activation", "logits", "weights"):
        for name in names:
            pair_a, pair_b = pair_indices[name]
            observed_values = matrices[representation][pair_a, pair_b]
            target_values = target_matrix[pair_a, pair_b]
            observed_spearman = correlation(observed_values, target_values, ranked=True)
            observed_pearson = correlation(observed_values, target_values)
            null = np.asarray(
                [
                    correlation(
                        observed_values,
                        target_matrix[np.ix_(permutation, permutation)][pair_a, pair_b],
                        ranked=True,
                    )
                    for permutation in label_permutations
                ]
            )
            rows.append(
                {
                    "representation": representation,
                    "subset": name,
                    "profiles": len(np.unique(np.r_[pair_a, pair_b])),
                    "pairs": len(pair_a),
                    "spearman_rho": observed_spearman,
                    "pearson_r": observed_pearson,
                    "null_mean": float(null.mean()),
                    "null_sd": float(null.std(ddof=0)),
                    "p_two_sided": (1 + int(np.sum(np.abs(null) >= abs(observed_spearman)))) / (permutations + 1),
                    "p_positive": (1 + int(np.sum(null >= observed_spearman))) / (permutations + 1),
                    "null_q025": float(np.quantile(null, 0.025)),
                    "null_q975": float(np.quantile(null, 0.975)),
                }
            )
    frame = pd.DataFrame(rows)
    frame["q_bh_all24"] = bh_adjust(frame.p_two_sided.to_numpy())
    return frame, pd.DataFrame(membership_rows).drop_duplicates().sort_values(["subset", "ptype"])


def exposure_summary(project: Path) -> pd.DataFrame:
    correlations = pd.read_csv(project / "analysis_outputs" / "exposure_effect_correlations.csv")
    wanted = {
        "update_frobenius_norm": "parameter_update_magnitude",
        "activation_effect_magnitude": "activation_effect_magnitude",
        "logit_effect_magnitude": "logit_effect_magnitude",
    }
    rows = []
    for variable, label in wanted.items():
        selected = correlations[
            ((correlations.variable_a == "retained_tokens") & (correlations.variable_b == variable))
            | ((correlations.variable_b == "retained_tokens") & (correlations.variable_a == variable))
        ].iloc[0]
        rows.append(
            {
                "exposure_measure": "retained_tokens",
                "effect_measure": label,
                "spearman_rho": selected.spearman_rho,
                "spearman_p": selected.spearman_p,
                "spearman_ci_low": selected.spearman_ci_low,
                "spearman_ci_high": selected.spearman_ci_high,
                "interpretation": (
                    "very strong parameter-space scaling"
                    if variable == "update_frobenius_norm"
                    else "moderate activation association"
                    if variable == "activation_effect_magnitude"
                    else "weak logit association"
                ),
            }
        )
    return pd.DataFrame(rows)


def render_figures(project: Path, main_effects: pd.DataFrame, subsets: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    figure_root = project / "figures"
    data_root = figure_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    plot = main_effects.copy()
    plot.to_csv(data_root / "figure_19_main_effect_permutation_enrichment.csv", index=False)
    x = np.arange(len(plot))
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.bar(x - 0.18, plot.observed_median, width=0.36, label="Observed median")
    axis.bar(x + 0.18, plot.null_median_mean, width=0.36, label="Label-permutation null")
    axis.errorbar(
        x + 0.18,
        plot.null_median_mean,
        yerr=np.vstack(
            [plot.null_median_mean - plot.null_median_q025, plot.null_median_q975 - plot.null_median_mean]
        ),
        fmt="none",
        capsize=4,
    )
    axis.set_xticks(x, [value.capitalize() for value in plot.representation])
    axis.set_ylabel("Main-effect share of non-intercept energy")
    axis.set_title("OCEAN main-effect energy versus shuffled-label baseline")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_root / "figure_19_main_effect_permutation_enrichment.png", dpi=220)
    fig.savefig(figure_root / "figure_19_main_effect_permutation_enrichment.svg")
    plt.close(fig)

    selected_names = ["all_32", "rows_ge_50000", "pair_row_ratio_le_1_5", "pair_row_ratio_le_2_0"]
    selected = subsets[
        subsets.subset.isin(selected_names) & subsets.representation.isin(["activation", "logits"])
    ].copy()
    subset_labels = {
        "all_32": "all 32",
        "rows_ge_50000": "rows ≥ 50,000",
        "pair_row_ratio_le_1_5": "row ratio ≤ 1.5",
        "pair_row_ratio_le_2_0": "row ratio ≤ 2.0",
    }
    selected["label"] = selected.apply(
        lambda row: f"{row.representation.capitalize()}: {subset_labels[row.subset]}", axis=1
    )
    selected = selected.sort_values("spearman_rho")
    selected.to_csv(data_root / "figure_20_exposure_matched_alignment.csv", index=False)
    fig, axis = plt.subplots(figsize=(11, 7))
    axis.barh(selected.label, selected.spearman_rho)
    for position, row in enumerate(selected.itertuples()):
        axis.text(row.spearman_rho + 0.006, position, f"ρ={row.spearman_rho:.3f}, q={row.q_bh_all24:.3f}", va="center")
    axis.axvline(0, color="0.4", linewidth=0.8)
    axis.set_xlabel("Spearman correlation with row-weighted OCEAN distance")
    axis.set_title("OCEAN alignment strengthens in fixed exposure-matched comparisons")
    fig.tight_layout()
    fig.savefig(figure_root / "figure_20_exposure_matched_alignment.png", dpi=220)
    fig.savefig(figure_root / "figure_20_exposure_matched_alignment.svg")
    plt.close(fig)
    atomic_json(
        figure_root / "figure_manifest.json",
        {
            "complete": True,
            "figures": 20,
            "formats": ["png", "svg"],
            "underlying_csv_directory": "figures/data",
            "corrected_figures": {
                "19": "shuffled-label Walsh main-effect enrichment",
                "20": "fixed exposure-matched continuous-target alignment",
            },
        },
    )


def write_decision(project: Path, main_effects: pd.DataFrame, subsets: pd.DataFrame, exposure: pd.DataFrame) -> None:
    main_lookup = main_effects.set_index("representation")
    subset_lookup = subsets.set_index(["representation", "subset"])
    exposure_lookup = exposure.set_index("effect_measure")
    decision = {
        "study": "full_profile_drift_study_v1_corrected",
        "complete": True,
        "correction_scope": "analysis-only; no model experiment, adapter load, capture, generation, or training rerun",
        "supported_interpretation": (
            "All 32 adapters learn strong, reproducible identities. Their parameter magnitudes are heavily shaped "
            "by unequal training exposure. Most geometry is idiosyncratic rather than five stable trait vectors. "
            "Nevertheless, activation and logit representations contain a weak aggregate OCEAN component, and "
            "continuous alignment is modest but statistically supported in fixed exposure-matched comparisons."
        ),
        "confirmatory_findings": [
            "adapter uniqueness",
            "cross-representation preregistered all-32 analyses",
        ],
        "exploratory_sensitivity_findings": [
            "fixed exposure-restricted subsets",
            "label-permutation enrichment",
            "residualized analyses",
        ],
        "results": {
            "main_effect_enrichment": {
                representation: {
                    "observed_median": float(main_lookup.loc[representation].observed_median),
                    "permutation_null_median_mean": float(main_lookup.loc[representation].null_median_mean),
                    "p_upper": float(main_lookup.loc[representation].p_upper),
                }
                for representation in ("activation", "logits", "weights")
            },
            "all_32_row_weighted_alignment": {
                representation: {
                    "spearman_rho": float(subset_lookup.loc[(representation, "all_32")].spearman_rho),
                    "q_bh_all24": float(subset_lookup.loc[(representation, "all_32")].q_bh_all24),
                }
                for representation in ("activation", "logits", "weights")
            },
            "fixed_exposure_matched_alignment": {
                representation: {
                    limit: {
                        "spearman_rho": float(subset_lookup.loc[(representation, limit)].spearman_rho),
                        "q_bh_all24": float(subset_lookup.loc[(representation, limit)].q_bh_all24),
                    }
                    for limit in ("pair_row_ratio_le_1_5", "pair_row_ratio_le_2_0")
                }
                for representation in ("activation", "logits", "weights")
            },
            "retained_token_effect_associations": {
                effect: float(exposure_lookup.loc[effect].spearman_rho)
                for effect in exposure_lookup.index
            },
        },
        "rejected_interpretation": (
            "The uncalibrated 79.9% interaction-energy share is not evidence that interactions dominate: "
            "26 of 31 non-intercept Walsh terms are interactions, so the relevant null is label permutation."
        ),
        "evidential_boundary": (
            "These results establish reproducible adapter identities and weak relational OCEAN-associated geometry; "
            "they do not establish human personality or validate any adapter as a human personality profile."
        ),
    }
    atomic_json(project / "full_profile_drift_decision.json", decision)
    lines = [
        "# Corrected full profile drift decision",
        "",
        decision["supported_interpretation"],
        "",
        "## Primary all-32 findings",
        "",
        "- All 32 adapters have strong, reproducible internal identities; this remains the central confirmatory result.",
        f"- Activation main-effect share is {main_lookup.loc['activation'].observed_median:.3f} versus a shuffled-label expectation of {main_lookup.loc['activation'].null_median_mean:.3f} (p={main_lookup.loc['activation'].p_upper:.4f}).",
        f"- Logit main-effect share is {main_lookup.loc['logits'].observed_median:.3f} versus {main_lookup.loc['logits'].null_median_mean:.3f} (p={main_lookup.loc['logits'].p_upper:.4f}).",
        f"- Weight main-effect share is {main_lookup.loc['weights'].observed_median:.3f} versus {main_lookup.loc['weights'].null_median_mean:.3f}; the absolute enrichment is substantively tiny despite detectability.",
        f"- All-32 continuous alignment is weak: activation rho={subset_lookup.loc[('activation','all_32')].spearman_rho:.3f}, q={subset_lookup.loc[('activation','all_32')].q_bh_all24:.3f}; logits rho={subset_lookup.loc[('logits','all_32')].spearman_rho:.3f}, q={subset_lookup.loc[('logits','all_32')].q_bh_all24:.3f}.",
        "",
        "## Exploratory fixed exposure sensitivity",
        "",
        f"- Row-count ratio <=2.0: activation rho={subset_lookup.loc[('activation','pair_row_ratio_le_2_0')].spearman_rho:.3f}, q={subset_lookup.loc[('activation','pair_row_ratio_le_2_0')].q_bh_all24:.3f}; logits rho={subset_lookup.loc[('logits','pair_row_ratio_le_2_0')].spearman_rho:.3f}, q={subset_lookup.loc[('logits','pair_row_ratio_le_2_0')].q_bh_all24:.3f}.",
        f"- Row-count ratio <=1.5: activation rho={subset_lookup.loc[('activation','pair_row_ratio_le_1_5')].spearman_rho:.3f}, q={subset_lookup.loc[('activation','pair_row_ratio_le_1_5')].q_bh_all24:.3f}; logits rho={subset_lookup.loc[('logits','pair_row_ratio_le_1_5')].spearman_rho:.3f}, q={subset_lookup.loc[('logits','pair_row_ratio_le_1_5')].q_bh_all24:.3f}.",
        "- Weight-space distance does not show the same exposure-matched alignment.",
        "",
        "## Exposure interpretation",
        "",
        f"Retained-token exposure is associated very strongly with update magnitude (rho={exposure_lookup.loc['parameter_update_magnitude'].spearman_rho:.3f}), moderately with activation magnitude (rho={exposure_lookup.loc['activation_effect_magnitude'].spearman_rho:.3f}), and weakly with logit magnitude (rho={exposure_lookup.loc['logit_effect_magnitude'].spearman_rho:.3f}). The corrected conclusion therefore does not call all learned drift uniformly exposure-dominated.",
        "",
        "## Multiplicity and evidential boundary",
        "",
        "No individual named Walsh term is treated as confirmed unless it survives the complete termwise multiplicity family. Fixed exposure subsets, permutation enrichment, and residualized analyses are exploratory sensitivity analyses. Adapter distinctiveness and weak OCEAN-associated relational structure do not establish human personality validity.",
    ]
    (project / "full_profile_drift_decision.md").write_text("\n".join(lines) + "\n")


def write_audit_report(project: Path, main_effects: pd.DataFrame, term_tests: pd.DataFrame, subsets: pd.DataFrame, exposure: pd.DataFrame) -> None:
    main = main_effects.set_index("representation")
    fixed = subsets.set_index(["representation", "subset"])
    effect = exposure.set_index("effect_measure")
    minimum_term_q = float(term_tests.q_bh_within_representation_31.min())
    lines = [
        "# Analysis-only remediation audit",
        "",
        "## Scope and integrity",
        "",
        "This correction used only the existing compact outputs. It did not load a base model or adapter, run inference, capture activations, generate text, or retrain anything. The 297 authoritative capture shards, 33 conditions, 1,080-prompt manifest, 3,960 stored continuations, and original deterministic/base-restoration audit remain unchanged.",
        "",
        "The remediation independently rebuilt representation Gram matrices from the six activation-centroid files, three logit-centroid files, and 32 exact weight-geometry layers. It reproduced all rows of both supplied independent audit CSVs before writing corrected conclusions.",
        "",
        "## Shuffled-label Walsh audit",
        "",
        "| Representation | Conditions | Observed median main share | Null median | Upper p |",
        "|---|---:|---:|---:|---:|",
    ]
    for representation in ("activation", "logits", "weights"):
        row = main.loc[representation]
        lines.append(f"| {representation.capitalize()} | {int(row.conditions)} | {row.observed_median:.6f} | {row.null_median_mean:.6f} | {row.p_upper:.6f} |")
    lines += [
        "",
        "The uncalibrated 79.9% interaction share is not interpreted as interaction dominance. Of 31 non-intercept Walsh terms, five are main effects and 26 are interactions; isotropic/idiosyncratic energy therefore has an expected main share of 5/31 (0.1613). Activation and especially logit representations show weak aggregate main-effect enrichment above shuffled labels. Weight-space enrichment is substantively tiny.",
        "",
        f"No individual named term survives BH correction across its complete 31-term representation family (minimum q={minimum_term_q:.3f}). Named traits are therefore not promoted from exploratory hints to confirmed trait vectors.",
        "",
        "## All-32 primary versus fixed exposure sensitivities",
        "",
        "| Representation / subset | Pairs | Spearman rho | BH q (24 tests) |",
        "|---|---:|---:|---:|",
    ]
    for representation, subset in (
        ("activation", "all_32"),
        ("logits", "all_32"),
        ("activation", "pair_row_ratio_le_1_5"),
        ("activation", "pair_row_ratio_le_2_0"),
        ("logits", "pair_row_ratio_le_1_5"),
        ("logits", "pair_row_ratio_le_2_0"),
        ("logits", "rows_ge_50000"),
        ("weights", "pair_row_ratio_le_2_0"),
    ):
        row = fixed.loc[(representation, subset)]
        lines.append(f"| {representation.capitalize()} / {subset} | {int(row.pairs)} | {row.spearman_rho:.6f} | {row.q_bh_all24:.6f} |")
    lines += [
        "",
        "The all-32 continuous alignment results are primary and weak. Fixed exposure subsets are exploratory sensitivity analyses. They show modest, statistically supported row-weighted continuous-target alignment for activation and logit distances, but not for weight-space distance. No favorable subset, profile, layer, prompt group, or trait was selected after inspection.",
        "",
        "## Exposure is representation-dependent",
        "",
        "| Effect measure | Retained-token Spearman rho | Interpretation |",
        "|---|---:|---|",
    ]
    for name in ("parameter_update_magnitude", "activation_effect_magnitude", "logit_effect_magnitude"):
        row = effect.loc[name]
        lines.append(f"| {name} | {row.spearman_rho:.6f} | {row.interpretation} |")
    lines += [
        "",
        "Unequal exposure nearly determines LoRA update magnitude, is moderately related to activation-effect magnitude, and is only weakly related to logit-effect magnitude. The corrected report therefore avoids calling every representation uniformly exposure-dominated.",
        "",
        "## Corrected conclusion",
        "",
        "All 32 adapters learn strong, reproducible identities. Their parameter magnitudes are heavily shaped by unequal training exposure. Most geometry is idiosyncratic rather than five stable trait vectors. Nevertheless, activation and logit representations contain a weak aggregate OCEAN component, and continuous alignment is modest but statistically supported in fixed exposure-matched comparisons.",
        "",
        "This is evidence about adapter geometry and OCEAN-associated response structure, not proof of human personality or intended profile validity.",
    ]
    (project / "audit" / "remediation_audit.md").write_text("\n".join(lines) + "\n")


def verify_against_supplied(project: Path, generated: pd.DataFrame, filename: str, columns: list[str]) -> None:
    supplied = pd.read_csv(project / "audit" / "remediation_inputs" / filename)
    merged = generated.merge(supplied, on=columns, suffixes=("_generated", "_supplied"))
    if len(merged) != len(supplied) or len(generated) != len(supplied):
        raise AssertionError(f"row mismatch against {filename}")
    for column in supplied.columns:
        if column in columns:
            continue
        left = merged[f"{column}_generated"]
        right = merged[f"{column}_supplied"]
        if pd.api.types.is_numeric_dtype(right):
            # The supplied z scores were computed from unrounded null moments
            # not retained in the CSV.  All raw shares/p-values match much more
            # tightly; 1e-5 relative tolerance covers that documented z-only
            # round-trip without weakening row/count or p-value comparisons.
            if not np.allclose(left, right, rtol=1e-5, atol=1e-8, equal_nan=True):
                raise AssertionError(f"numeric mismatch in {filename}:{column}")
        elif not left.equals(right):
            raise AssertionError(f"text mismatch in {filename}:{column}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    analysis = project / "analysis_outputs"
    tables = project / "tables"
    tables.mkdir(exist_ok=True)

    main_effects, term_tests, order_summary = main_effect_permutation(project)
    subsets, membership = restricted_subset_permutation(project)
    exposure = exposure_summary(project)

    verify_against_supplied(
        project,
        main_effects,
        "full_profile_drift_main_effect_permutation_summary.csv",
        ["representation"],
    )
    verify_against_supplied(
        project,
        subsets,
        "full_profile_drift_subset_permutation_audit_bh.csv",
        ["representation", "subset"],
    )

    main_effects.to_csv(analysis / "main_effect_permutation_summary.csv", index=False)
    term_tests.to_csv(analysis / "walsh_term_permutation_tests.csv", index=False)
    order_summary.to_csv(analysis / "walsh_order_null_context.csv", index=False)
    subsets.to_csv(analysis / "fixed_subset_permutation_tests.csv", index=False)
    membership.to_csv(tables / "fixed_subset_membership.csv", index=False)
    exposure.to_csv(tables / "exposure_relationships_by_representation.csv", index=False)
    if not args.skip_figures:
        render_figures(project, main_effects, subsets)
    write_decision(project, main_effects, subsets, exposure)

    summary_path = analysis / "analysis_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "corrected_analysis_only": True,
            "model_experiments_rerun": False,
            "confirmatory_primary": [
                "adapter uniqueness",
                "cross-representation preregistered all-32 analyses",
            ],
            "exploratory_sensitivity": [
                "fixed exposure-restricted subsets",
                "label-permutation enrichment",
                "residualized analyses",
            ],
            "main_effect_permutation_seed": SEED,
            "main_effect_permutations": 2000,
            "fixed_subset_permutations": 3000,
            "fixed_subset_bh_family": 24,
        }
    )
    atomic_json(summary_path, summary)
    compact_path = analysis / "compact_audit_summary.json"
    compact = json.loads(compact_path.read_text())
    compact.update(
        {
            "corrected": True,
            "model_experiments_rerun": False,
            "supported_descriptions": [
                "strong reproducible adapter identities",
                "strong exposure scaling in parameter space",
                "weak aggregate OCEAN-related activation/logit structure",
                "modest exposure-matched continuous alignment",
                "predominantly idiosyncratic residual geometry",
            ],
            "withdrawn_description": "interaction-dominated profile geometry based on uncalibrated aggregate energy share",
            "confirmatory_findings": ["adapter uniqueness", "cross-representation preregistered all-32 analyses"],
            "exploratory_sensitivity_findings": ["fixed exposure subsets", "label-permutation enrichment", "residualized analyses"],
        }
    )
    compact.pop("criteria", None)
    atomic_json(compact_path, compact)
    write_audit_report(project, main_effects, term_tests, subsets, exposure)
    atomic_json(
        analysis / "remediation_analysis_summary.json",
        {
            "complete": True,
            "input_scope": "existing compact outputs only",
            "model_experiments_rerun": False,
            "supplied_audit_statistics_reproduced": True,
            "main_effect_rows": len(main_effects),
            "termwise_rows": len(term_tests),
            "fixed_subset_rows": len(subsets),
            "main_effect_permutations": 2000,
            "fixed_subset_permutations": 3000,
            "seed": SEED,
            "source_files": {
                path.relative_to(project).as_posix(): sha256(path)
                for path in sorted((project / "audit" / "remediation_inputs").glob("*"))
            },
        },
    )
    atomic_json(
        project / "progress.json",
        {
            "project": "full_profile_drift_study_v1_corrected",
            "stage": "analysis_only_remediation_complete",
            "complete": True,
            "model_experiments_rerun": False,
            "integrity_failure": None,
        },
    )
    environment_path = project / "environment_manifest.json"
    atomic_json(
        project / "audit" / "remediation_execution.json",
        {
            "scope": "analysis-only over existing compact outputs",
            "model_experiments_rerun": False,
            "command": "PYTHONPATH=src python -m full_profile_drift.remediation --project .",
            "test_command": "PYTHONPATH=src python -m pytest -q",
            "integrity_command": "sha256sum -c ARTIFACT_CHECKSUMS.sha256",
            "seed": SEED,
            "main_effect_label_permutations": 2000,
            "fixed_subset_label_permutations": 3000,
            "bh_family_size": 24,
            "source_hashes": {
                path.relative_to(project).as_posix(): sha256(path)
                for path in sorted((project / "src").rglob("*.py"))
            },
            "environment_manifest": {
                "path": "environment_manifest.json",
                "sha256": sha256(environment_path),
            },
            "requirements_lock": {
                "path": "requirements.lock.txt",
                "sha256": sha256(project / "requirements.lock.txt"),
            },
        },
    )


if __name__ == "__main__":
    main()
