# Full profile drift study v1 — corrected analysis package

Analysis-only correction of the completed, exploratory all-32 study of the existing OCEAN-profile LoRA adapters over
the frozen `meta-llama/Llama-3.1-8B` revision
`d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.

This project is isolated from the completed BlueDot behavioral, activation,
and mapping-audit packages. Those packages are read-only evidence inputs.
No model experiment was rerun for this correction. The correction reads only
the compact study's saved centroids, pairwise distances, exposure tables, and
analysis summaries. No adapter was loaded, merged, normalized, edited, or
retrained.

The verified encoding is:

```text
ptype = 16*O_high + 8*C_high + 4*E_high + 2*A_high + N_high
```

Progress is recorded atomically in `progress.json`. Engineering-integrity
failures are terminal; negative or unexpected scientific results are not.

## Study contents

- `ptype_catalog.*`: all 32 profiles, continuous centroids, exposure proxies,
  effective-update norms, and immutable adapter hashes.
- `training_exposure/`: streaming tokenizer exposure over all 3,006,566 rows.
- `weight_geometry/`: exact low-rank per-block kernels, distances, cosine
  matrices, effective ranks, PCA coordinates, clustering, and norm tables.
- `prompt_manifest/`: the frozen 1,080-prompt corpus with the exact 360-prompt
  legacy core marked, stable splits, token IDs, hashes, and fingerprint.
- `activation_outputs/` and `logit_outputs/`: raw capture shards on RunPod;
  the compact package contains a complete checksum index rather than copying
  tens of gigabytes.
- `analysis_outputs/`: all-layer factorial, matched-flip, uniqueness,
  predictive, exposure, continuous-target, sensitivity, and
  cross-representation results.
- `continuation_outputs/`: exact deterministic texts and non-personality
  signature analyses.
- `figures/`: the original 18 figures plus two corrected audit figures in PNG
  and SVG, each with source CSV data.
- `tables/`: fixed early/middle/late summaries and per-adapter audit tables.
- `analysis_outputs/main_effect_permutation_summary.csv`: observed aggregate
  Walsh main-effect energy versus a 2,000-permutation shuffled-label null.
- `analysis_outputs/walsh_term_permutation_tests.csv`: complete termwise tests
  with within-representation and all-term multiplicity control.
- `analysis_outputs/fixed_subset_permutation_tests.csv`: fixed exposure subset
  and pair-restriction tests with one predeclared 24-test BH family.
- `tables/fixed_subset_membership.csv`: exact subset membership and pair-count
  audit support.
- `tables/exposure_relationships_by_representation.csv`: separate exposure
  relationships for parameter, activation, and logit magnitudes.
- `full_profile_drift_decision.*`: the corrected supported interpretation, with
  confirmatory all-32 results separated from exploratory sensitivity analyses.

## Corrected interpretation

All 32 adapters learn strong, reproducible identities. Their parameter
magnitudes are heavily shaped by unequal training exposure. Most geometry is
idiosyncratic rather than five stable trait vectors. Nevertheless, activation
and logit representations contain a weak aggregate OCEAN component, and
continuous alignment is modest but statistically supported in fixed
exposure-matched comparisons.

The former statement that 79.9% interaction energy establishes
"interaction-dominated" geometry is withdrawn. There are 26 interaction terms
but only five main-effect terms among the 31 non-intercept Walsh terms. Against
a shuffled-label null, activation and logit main-effect energy is enriched;
weight-space enrichment is substantively tiny. No individual named Walsh term
survives the complete termwise multiplicity family.

The preregistered all-32 adapter-uniqueness and cross-representation analyses
are confirmatory. Fixed exposure subsets, label-permutation enrichment, and
residualized results are explicitly exploratory sensitivity analyses.

## Reproduce the correction

From this directory, using the pinned environment:

```bash
PYTHONPATH=src python -m full_profile_drift.remediation --project .
PYTHONPATH=src python -m pytest
sha256sum -c ARTIFACT_CHECKSUMS.sha256
```

The remediation command verifies its principal statistics against the supplied
independent audit tables before writing corrected outputs. It does not import a
model runtime or access adapter weights.

## Storage remediation

RunPod's network filesystem reported 310 TB free but enforced a hidden quota
at 18 GB of new study output during ptype_12. All completed and partial files
were preserved. Base plus ptype_0–11 remain under
`/workspace/full_profile_drift_study_v1`; authoritative ptype_12–31 shards and
downstream outputs are under `/root/full_profile_drift_spill`. The pod overlay
had 52 GB free at remediation. `audit/capture_shard_index.csv` records the
authoritative location and checksum of every raw shard.

## Evidential boundary

This exploratory study establishes systematic adapter geometry, exposure
relationships, reproducible continuation signatures, and weak aggregate
OCEAN-associated relational structure. It does not establish that an adapter
has a human personality or that a profile label is valid.
