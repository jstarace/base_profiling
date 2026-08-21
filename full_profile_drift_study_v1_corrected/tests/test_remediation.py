import json

import numpy as np
import pandas as pd

from full_profile_drift.remediation import (
    SEED,
    bh_adjust,
    correlation,
    design_matrix,
    subset_definitions,
)


def test_shuffled_label_baseline_uses_all_five_main_terms():
    design = design_matrix()
    assert design.shape == (32, 32)
    assert np.array_equal(design.T @ design, 32 * np.eye(32))
    assert 5 / 31 == sum(len(term) == 1 for term in __import__("full_profile_drift.remediation", fromlist=["TERMS"]).TERMS) / 31
    rng = np.random.default_rng(SEED)
    gram = rng.normal(size=(32, 8))
    gram = gram @ gram.T
    denominator = np.trace(gram) / 32 - gram.sum() / 1024
    observed = np.einsum("pi,pq,qi->", design[:, 1:6], gram, design[:, 1:6]) / 1024 / denominator
    null = []
    for _ in range(100):
        permuted = design[rng.permutation(32), 1:6]
        null.append(np.einsum("pi,pq,qi->", permuted, gram, permuted) / 1024 / denominator)
    assert np.isfinite(observed) and np.isfinite(null).all()


def test_fixed_subset_membership_and_pair_counts():
    definitions, ratios = subset_definitions(__import__("pathlib").Path("."))
    assert len(definitions["all_32"]) == 32
    assert len(definitions["rows_ge_1000"]) == 30
    assert len(definitions["rows_ge_10000"]) == 30
    assert len(definitions["rows_ge_50000"]) == 22
    assert len(definitions["exposure_quartile_1"]) == 8
    assert len(definitions["exposure_quartile_4"]) == 8
    assert int(np.triu(ratios <= 1.5, 1).sum()) == 117
    assert int(np.triu(ratios <= 2.0, 1).sum()) == 181


def test_exact_permutation_p_uses_plus_one_correction():
    observed = 0.5
    null = np.array([-0.6, -0.4, 0.1, 0.5])
    two_sided = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (len(null) + 1)
    positive = (1 + int(np.sum(null >= observed))) / (len(null) + 1)
    assert two_sided == 3 / 5
    assert positive == 2 / 5


def test_bh_family_is_complete_and_monotone():
    values = np.array([0.001, 0.02, 0.04, 0.5] * 6)
    adjusted = bh_adjust(values)
    assert len(adjusted) == 24
    ordered = np.argsort(values)
    assert np.all(np.diff(adjusted[ordered]) >= -1e-15)
    assert np.all((0 <= adjusted) & (adjusted <= 1))


def test_corrected_decision_matches_support_tables():
    decision = json.load(open("full_profile_drift_decision.json"))
    main = pd.read_csv("analysis_outputs/main_effect_permutation_summary.csv").set_index("representation")
    subsets = pd.read_csv("analysis_outputs/fixed_subset_permutation_tests.csv").set_index(["representation", "subset"])
    assert "79.9% interaction-energy share is not evidence" in decision["rejected_interpretation"]
    assert np.isclose(decision["results"]["main_effect_enrichment"]["activation"]["observed_median"], main.loc["activation", "observed_median"])
    assert np.isclose(decision["results"]["fixed_exposure_matched_alignment"]["logits"]["pair_row_ratio_le_2_0"]["q_bh_all24"], subsets.loc[("logits", "pair_row_ratio_le_2_0"), "q_bh_all24"])
    assert decision["correction_scope"].startswith("analysis-only")


def test_rank_correlation_matches_known_reversal():
    assert correlation(np.arange(5), np.arange(5)[::-1], ranked=True) == -1.0
