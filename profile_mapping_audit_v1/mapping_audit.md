# Read-only end-to-end profile-mapping audit

## Conclusion

The intended mapping is verified end to end. **No mapping discrepancy was
found, and Category B is retained.** No model experiment was rerun.

The mapping is:

```text
ptype = 16*O_high + 8*C_high + 4*E_high + 2*A_high + N_high
high  = raw trait score > 50
```

The full 3,006,566-row PANDORA redistribution has zero mismatches under that
formula. All raw values equal to 50 decode low; substituting `>=50` creates
between 64,411 and 134,977 mismatches per trait. Globally reversing the
polarity disagrees on every row for every trait.

The requested pilot decodes are exact:

| adapter | O | C | E | A | N | packed value |
|---|---:|---:|---:|---:|---:|---:|
| `ptype_0` | 0 | 0 | 0 | 0 | 0 | 0 |
| `ptype_31` | 1 | 1 | 1 | 1 | 1 | 31 |
| `ptype_9` | 0 | 1 | 0 | 0 | 1 | 9 |
| `ptype_23` | 1 | 0 | 1 | 1 | 1 | 23 |

## Chain trace

1. **Continuous thresholds.** The download-only importer concatenates the
   upstream splits and preserves their columns without label transformation.
   The strict `>50` decoder was independently tested against every stored
   integer `ptype`: zero mismatches.
2. **Polarity.** Higher values produce `_high=1`; every score exactly 50
   produces `_high=0`. There is no reversed trait.
3. **Trait ordering.** The raw schema is `O,C,E,A,N,ptype`; the recovered
   weights are `16,8,4,2,1`. Neither alphabetical ordering (`A,C,E,N,O`) nor
   incidental dataframe ordering is used by training.
4. **Integer calculation.** Independently packing the five strict-threshold
   bits reproduces all 3,006,566 upstream integers.
5. **Training class assignment.** Training takes an integer from the explicit
   partition and filters with `ex["ptype"] == ptype`. It does not derive a
   label from trait columns or their order.
6. **Adapter naming.** The same loop integer is used in both the filter and
   `out_root / f"ptype_{ptype}"`. The live volume contains exactly
   `ptype_0`–`ptype_31`; the four pilot weight hashes match both the benchmark
   and activation manifests.
7. **Benchmark targets.** Every one of the 32 target rows was independently
   regenerated from `raw[raw.ptype == ptype]`. Row counts, distinct five-trait
   tuples, bits, means, medians, and population standard deviations match;
   maximum floating-point difference is `1.42e-14`.
8. **Activation lookup.** The analysis strips the numeric suffix from each
   adapter key, indexes the target table by that integer, and requests trait
   columns explicitly in the literal order `OCEAN`.

The repository does not contain a separate original script that generated
`ocean_profile_targets.csv`; this is a provenance gap, not a mapping mismatch.
The frozen table itself is fully reproducible from the exact raw class records,
and its SHA-256 is recorded in `mapping_audit.json`.

## Representative original training rows

The table below shows one deterministic representative per adapter. The CSV
output contains first, middle, and last source-index representatives—12 rows
total—and all match.

| adapter | source row | raw O,C,E,A,N | recomputed O,C,E,A,N | recomputed/stored ptype | training path | target bits | status |
|---|---:|---|---|---|---|---|---|
| `ptype_0` | 56,660 | 17, 8, 16, 41, 25 | 0,0,0,0,0 | 0 / 0 | `/workspace/adapters/ptype_0` | 0,0,0,0,0 | MATCH |
| `ptype_31` | 40,173 | 85, 97, 61, 87, 67 | 1,1,1,1,1 | 31 / 31 | `/workspace/adapters/ptype_31` | 1,1,1,1,1 | MATCH |
| `ptype_9` | 18,584 | 1, 71, 0, 28, 59 | 0,1,0,0,1 | 9 / 9 | `/workspace/adapters/ptype_9` | 0,1,0,0,1 | MATCH |
| `ptype_23` | 408,468 | 54, 20, 59, 69, 88 | 1,0,1,1,1 | 23 / 23 | `/workspace/adapters/ptype_23` | 1,0,1,1,1 | MATCH |

## Adapter identity and rename check

The live adapter paths and SHA-256 values are:

| path | adapter-model SHA-256 |
|---|---|
| `/workspace/adapters/ptype_0` | `e7eba65dbc9560dac32ca75c0f8a9fc9c7f19ff09b8503c57b00bf36dd20754a` |
| `/workspace/adapters/ptype_31` | `2919d58c713d9b3e2318d9b5518801ff4e3c5ee93e60add67b2acbb6e672c2fb` |
| `/workspace/adapters/ptype_9` | `7c1ef6e75d7f69791c562e8e12a118e910b6a73b5e20f9fb507400bafd5fa5bd` |
| `/workspace/adapters/ptype_23` | `4e8043f301afe136ef1bd390d9e84d7635efeacc51dc28b6db359f6d6d9c5aed` |

No code path renames or reassigns an adapter, and current names and contents
match the frozen manifests. A historical filesystem rename cannot be disproved
without an external filesystem event log, but there is no evidence of one and
no identity inconsistency in the available provenance.

## Exhaustive alternative-mapping diagnostic

All `5! × 2 = 240` position/polarity mappings were evaluated against all three
stored behavioral interfaces (720 rows) and the 192 completed activation
conditions. This was diagnostic only; nothing was selected or recomputed.

| behavioral interface | intended binary correlation | intended target-directed changes | intended Hamming distance | diagnostic best by directed-count-first rule | code evidence? |
|---|---:|---:|---:|---|---|
| A: label rotation | -0.1231 | 11/20 | 9/20 | `OCENA`, high=1 (13/20) | No |
| B: context-calibrated labels | 0.1452 | 8/20 | 9/20 | `OCENA`, inverted (14/20) | No |
| C: calibrated verbal anchors | -0.0240 | 13/20 | 10/20 | `ONCAE`, high=1 (13/20) | No |

The behaviorally best alternatives conflict with one another and have no
support in preprocessing or training code. More fundamentally, permuting trait
dimensions or globally reflecting polarity preserves Euclidean and Hamming
distances. Consequently all 240 alternatives produce the same activation
distance correlations (median Pearson `-0.116442`, median Spearman `-0.057977`)
as the intended mapping. Alternative encoding cannot rescue the Category B
activation result.

## Outputs

- `mapping_audit.json`: machine-readable conclusion and evidence hashes.
- `representative_training_rows.csv`: required raw-row audit table.
- `threshold_audit.csv`: strict/non-strict/polarity checks per trait.
- `target_recomputation_all_32.csv`: exact target-table regeneration audit.
- `alternative_mapping_diagnostics.csv`: all 720 behavioral mapping tests.
- `activation_mapping_invariance.csv`: all 240 activation mapping tests.
- `behavioral_alternative_summary.csv`: compact intended-versus-diagnostic summary.
- `audit_profile_mapping.py`: complete read-only audit code.

Supported conclusion: the stored adapters are correctly associated with their
original numeric PANDORA classes and the benchmark/activation target lookup is
consistent with those classes. The prior activation-study conclusion remains
**Category B: adapter-specific but non-OCEAN structure**.

## Continuous-target distribution extension

The follow-up read-only distribution audit is in
`continuous_target_distribution_audit.md`. It reports all 32 class
distributions, continuous centroid geometry, pilot outcome comparisons,
target-use provenance, and within-versus-between class variation. Its result
also retains Category B.
