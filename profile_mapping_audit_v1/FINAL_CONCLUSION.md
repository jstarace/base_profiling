# Final profile-mapping audit conclusion

## Terminal result

The corrected audit-only standard-deviation comparison uses the frozen target
table's population convention, `ddof=0`. The correction resolves the earlier
audit-comparison artifact and reveals no pipeline mapping discrepancy.

**The terminal activation decision does not change:**

> **Category B: adapter-specific but non-OCEAN-aligned structure**

No model inference, activation capture, questionnaire scoring, permutation
analysis, adapter training, or target selection was rerun to produce this final
package.

## Mapping confirmation

- Raw PANDORA rows audited: **3,006,566**.
- Strict `>50` threshold mismatches: **0**.
- Values equal to 50 are low; the rule is not `>=50`.
- Trait-bit order: **O, C, E, A, N**.
- Integer packing:
  `ptype = 16*O_high + 8*C_high + 4*E_high + 2*A_high + N_high`.
- Training filters the stored integer class and saves the same integer to
  `/workspace/adapters/ptype_<integer>`.
- All 32 class memberships, target bits, row counts, unique-tuple counts,
  means, medians, and population standard deviations reproduce the frozen
  target table. Maximum floating-point difference: **1.42e-14**.
- The live pilot adapter hashes match both the behavioral and activation
  manifests; the verified directory set is exactly `ptype_0`–`ptype_31`.

Exact pilot mappings:

| adapter | O | C | E | A | N |
|---|---:|---:|---:|---:|---:|
| `ptype_0` | 0 | 0 | 0 | 0 | 0 |
| `ptype_31` | 1 | 1 | 1 | 1 | 1 |
| `ptype_9` | 0 | 1 | 0 | 0 | 1 |
| `ptype_23` | 1 | 0 | 1 | 1 | 1 |

## Continuous-target conclusion

The package reports all 32 ptypes × five traits with row counts, means,
medians, population standard deviations, 5th/25th/75th/95th percentiles,
minima, maxima, row-weighted centroids, and unique-tuple-weighted centroids.

For `ptype_0` versus `ptype_31`:

- row-weighted centroid distance: **107.2411**;
- unique-tuple-weighted centroid distance: **112.3667**;
- idealized binary 0/100-corner distance: **223.6068**;
- pooled within-class RMS: **31.5739**;
- pooled within-RMS / actual row-centroid separation: **0.2944**.

Across all classes, within-class RMS was 32.0%–84.3% of the nearest class
centroid separation and exceeded it for 0/32 classes. The thresholded classes
therefore retain substantial internal heterogeneity, but the audit did not find
within-class RMS greater than nearest-centroid separation.

The activation Category A/B target-alignment decision used row-weighted and
unique-tuple-weighted continuous targets; binary alignment was reported but
could not establish Category A. The behavioral authorization gate was mixed:
row-weighted continuous correlation was its explicit correlation criterion,
while binary direction and endpoint-sign checks were also gate conditions.
Binary bits were not substituted as 0/100 continuous centroids.

## Final disposition

The mapping, class membership, adapter identity, and continuous target lookup
are consistent end to end. The adapters remain reliably distinguishable in
held-out activation space, while corrected continuous OCEAN-target alignment
was not established. **Retain Category B and stop.**
