# Independent audit of `full_profile_drift_study_v1`

## Integrity

- The compact package contains 705 checksum-covered files; `sha256sum -c ARTIFACT_CHECKSUMS.sha256` passes with zero failures.
- The capture index contains 297 authoritative shards: 33 conditions (base plus 32 adapters) × three prompt groups × two activation pooling rules plus logits.
- Every model condition has exactly nine indexed shards.
- The capture integrity summary reports no missing or extra authoritative shards, no checksum failures, and no base-restoration failures.
- The hidden RunPod quota incident was handled by storage routing only. No computational condition changed.
- The packaged tests pass (`6 passed`) when run with the documented `PYTHONPATH=src`. A bare `pytest` invocation fails to import the local package, which is only a minor packaging ergonomics issue.

## What the package clearly establishes

1. **All 32 adapters produce unique internal signatures.** Regularized logistic and shrinkage-LDA classification are approximately 0.97–1.00 balanced accuracy in the fixed summary layers and conditions. Nearest-centroid performance is more context/pooling dependent, especially for final-token activations, but remains far above 1/32 chance in aggregate.
2. **Activation and logit geometries are strongly related.** Their averaged pairwise-distance Spearman correlation is 0.861.
3. **Training exposure strongly structures the learned effects.** Token-exposure distance correlates 0.678 with activation distance and 0.613 with logit distance. Retained tokens correlate 0.944 with LoRA Frobenius norm, but only 0.487 with activation-effect rank and 0.174 with logit-effect rank. Exposure therefore dominates weight magnitude, moderately affects activations, and only weakly predicts logit magnitude.
4. **Generated continuations also contain adapter signatures.** A frozen TF–IDF logistic classifier identifies 32 adapters at 21.6% balanced accuracy versus 3.125% chance. This is distinctiveness, not personality validity.
5. **There is no clean global trait-vector geometry.** Matched high-minus-low directions have very low median pairwise cosines: O 0.0068, C 0.0216, E 0.0036, A 0.0337, N 0.0124. Main-effect factorial models have negative held-out explained variance.

## Important correction to the Codex decision

The label **“interaction-dominated profile geometry”** is not justified by the reported 79.9% interaction-energy share alone.

There are 5 main-effect Walsh terms but 26 interaction terms. For an isotropic/idiosyncratic mapping, the expected non-intercept energy shares are:

- main effects: 5/31 = 16.13%
- two-way: 10/31 = 32.26%
- three-way: 10/31 = 32.26%
- four-way: 5/31 = 16.13%
- five-way: 1/31 = 3.23%

The weight-space decomposition is almost exactly this combinatorial baseline. Consequently, summing all 26 interactions and comparing them with five main effects mostly measures the number of terms, not meaningful interaction structure.

The package's own Figure 13 implicitly exposes this: mean energy **per main-effect term** is larger than mean energy per interaction term.

### Independent label-permutation audit

Using the included all-32 centroids and a fixed 2,000-permutation label null:

| Representation | Observed median main-effect share | Permutation-null median | Result |
|---|---:|---:|---:|
| Activations | 0.2007 | 0.1605 | above null, p≈0.0005 |
| Logits | 0.2653 | 0.1601 | above null, p≈0.0005 |
| Weights | 0.1620 | 0.1613 | statistically detectable but substantively tiny |

Thus the better interpretation is:

> Most profile-dependent variance is idiosyncratic/higher-order, but activation and especially logit representations contain a small aggregate OCEAN main-effect enrichment above shuffled-label expectation.

No individual named Walsh term survives BH correction across all 31 terms. Nominally, Agreeableness and Conscientiousness are the strongest activation main effects, and Conscientiousness is the strongest logit main effect. These are exploratory hints, not confirmed trait vectors.

## The user's exposure-matching hypothesis is supported

The original sensitivity table suggested stronger OCEAN alignment among similarly trained profiles. I independently tested the averaged all-layer/all-context distance matrices with 3,000 label permutations and BH correction across 24 representation/subset tests.

| Representation and fixed subset | Spearman ρ with row-weighted OCEAN distance | BH q |
|---|---:|---:|
| Activation, row-count ratio ≤2.0 | 0.266 | 0.016 |
| Activation, row-count ratio ≤1.5 | 0.256 | 0.048 |
| Logits, row-count ratio ≤2.0 | 0.294 | 0.016 |
| Logits, row-count ratio ≤1.5 | 0.262 | 0.034 |
| Logits, profiles with ≥50,000 rows | 0.201 | 0.034 |

For comparison, the all-32 correlations are activation ρ=0.097 and logits ρ=0.111 and do not survive the same 24-test correction.

This is the clearest “method to the madness” result:

> When comparisons are restricted to adapters with similar training exposure, activation and logit distances show modest but statistically supported correspondence with continuous OCEAN distance.

The result does not establish persona validity. It shows that severe exposure imbalance masks a weak relational structure in the learned functional effects.

Weight-space distances do not show the same matched-exposure alignment. This implies that raw LoRA parameter distance is not a direct proxy for functional similarity.

## Recommended revised conclusion

The current decision should be revised from:

- interaction-dominated profile geometry;
- exposure-dominated drift;
- idiosyncratic identities.

To:

1. **Strong adapter identity:** all 32 adapters are internally and behaviorally distinguishable above chance.
2. **Strong exposure scaling in parameter space:** training exposure nearly determines update magnitude and contributes to a common activation drift.
3. **Weak OCEAN-related functional structure:** activation/logit main-effect energy exceeds a label-permutation baseline, and continuous-target alignment becomes clearer among exposure-matched adapters.
4. **No clean independent trait vectors:** matched trait-flip directions are weak and low-order factorial models do not generalize to held-out profiles.
5. **Predominantly idiosyncratic residual geometry:** most variation remains profile-specific after accounting for the weak aggregate trait structure.

## Required analysis-only remediation

No model capture or generation needs to be rerun. Codex should:

- add a shuffled-profile-label null for Walsh energy;
- report order energy relative to the combinatorial term-count baseline and report mean energy per term;
- remove or qualify “interaction-dominated” unless particular interactions exceed permutation baselines;
- add restricted Mantel-style permutation tests for the preregistered exposure-matched subsets;
- distinguish exposure relationships with update norm, activation magnitude, and logit magnitude rather than selecting the largest of the three;
- add the matched-exposure result to the decision and figures;
- preserve the conclusion that no clean global OCEAN trait-vector geometry was found;
- render every OCEAN bit string as five digits with leading zeros.
