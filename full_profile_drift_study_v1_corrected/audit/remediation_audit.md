# Analysis-only remediation audit

## Scope and integrity

This correction used only the existing compact outputs. It did not load a base model or adapter, run inference, capture activations, generate text, or retrain anything. The 297 authoritative capture shards, 33 conditions, 1,080-prompt manifest, 3,960 stored continuations, and original deterministic/base-restoration audit remain unchanged.

The remediation independently rebuilt representation Gram matrices from the six activation-centroid files, three logit-centroid files, and 32 exact weight-geometry layers. It reproduced all rows of both supplied independent audit CSVs before writing corrected conclusions.

## Shuffled-label Walsh audit

| Representation | Conditions | Observed median main share | Null median | Upper p |
|---|---:|---:|---:|---:|
| Activation | 192 | 0.200745 | 0.160488 | 0.000500 |
| Logits | 3 | 0.265315 | 0.160145 | 0.000500 |
| Weights | 32 | 0.161993 | 0.161280 | 0.001999 |

The uncalibrated 79.9% interaction share is not interpreted as interaction dominance. Of 31 non-intercept Walsh terms, five are main effects and 26 are interactions; isotropic/idiosyncratic energy therefore has an expected main share of 5/31 (0.1613). Activation and especially logit representations show weak aggregate main-effect enrichment above shuffled labels. Weight-space enrichment is substantively tiny.

No individual named term survives BH correction across its complete 31-term representation family (minimum q=0.124). Named traits are therefore not promoted from exploratory hints to confirmed trait vectors.

## All-32 primary versus fixed exposure sensitivities

| Representation / subset | Pairs | Spearman rho | BH q (24 tests) |
|---|---:|---:|---:|
| Activation / all_32 | 496 | 0.096807 | 0.233377 |
| Logits / all_32 | 496 | 0.111258 | 0.161089 |
| Activation / pair_row_ratio_le_1_5 | 117 | 0.255653 | 0.047984 |
| Activation / pair_row_ratio_le_2_0 | 181 | 0.266138 | 0.015995 |
| Logits / pair_row_ratio_le_1_5 | 117 | 0.262255 | 0.033989 |
| Logits / pair_row_ratio_le_2_0 | 181 | 0.294074 | 0.015995 |
| Logits / rows_ge_50000 | 231 | 0.200538 | 0.033989 |
| Weights / pair_row_ratio_le_2_0 | 181 | -0.183496 | 0.141286 |

The all-32 continuous alignment results are primary and weak. Fixed exposure subsets are exploratory sensitivity analyses. They show modest, statistically supported row-weighted continuous-target alignment for activation and logit distances, but not for weight-space distance. No favorable subset, profile, layer, prompt group, or trait was selected after inspection.

## Exposure is representation-dependent

| Effect measure | Retained-token Spearman rho | Interpretation |
|---|---:|---|
| parameter_update_magnitude | 0.944282 | very strong parameter-space scaling |
| activation_effect_magnitude | 0.487170 | moderate activation association |
| logit_effect_magnitude | 0.174487 | weak logit association |

Unequal exposure nearly determines LoRA update magnitude, is moderately related to activation-effect magnitude, and is only weakly related to logit-effect magnitude. The corrected report therefore avoids calling every representation uniformly exposure-dominated.

## Corrected conclusion

All 32 adapters learn strong, reproducible identities. Their parameter magnitudes are heavily shaped by unequal training exposure. Most geometry is idiosyncratic rather than five stable trait vectors. Nevertheless, activation and logit representations contain a weak aggregate OCEAN component, and continuous alignment is modest but statistically supported in fixed exposure-matched comparisons.

This is evidence about adapter geometry and OCEAN-associated response structure, not proof of human personality or intended profile validity.
