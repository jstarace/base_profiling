# Continuous-target distribution audit

## Conclusion

The continuous targets are correctly derived from the exact training classes,
and the activation study’s Category A/B decision used the continuous centroids
as its primary alignment target. The behavioral gate is more accurately
described as **mixed**: row-weighted continuous correlation was its explicit
target-correlation criterion, but binary target-direction and endpoint-sign
checks were also co-primary authorization conditions.

No test substituted binary values encoded as 0 and 100 for a continuous
centroid. Binary targets were stored as 0/1 and used for directional or Hamming
checks. Category B remains unchanged.

## Full class distributions

`continuous_target_distributions_all_32.csv` contains 160 rows: every one of
32 ptypes × five traits. Each row reports:

- training-row count;
- mean, median, population standard deviation;
- 5th, 25th, 75th, and 95th percentiles;
- minimum and maximum;
- row-weighted and unique-trait-tuple-weighted centroids;
- independent centroid-recomputation differences.

Both centroid definitions reproduce the frozen target table to floating-point
precision.

## Four pilot distributions

| adapter | trait | rows | mean | median | SD | p05 | p25 | p75 | p95 | min–max | unique-tuple centroid |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| ptype_0 | O | 159,137 | 24.55 | 23 | 13.46 | 4 | 15 | 34 | 48 | 0–50 | 22.58 |
| ptype_0 | C | 159,137 | 27.70 | 30 | 15.85 | 8 | 14 | 45 | 50 | 4–50 | 27.99 |
| ptype_0 | E | 159,137 | 23.52 | 21 | 13.78 | 1 | 15 | 33 | 49 | 0–50 | 24.14 |
| ptype_0 | A | 159,137 | 21.73 | 18 | 15.19 | 0 | 10 | 33 | 46 | 0–50 | 19.46 |
| ptype_0 | N | 159,137 | 23.50 | 25 | 16.63 | 1 | 8 | 39 | 50 | 0–50 | 25.72 |
| ptype_31 | O | 93,178 | 68.91 | 67 | 13.40 | 53 | 57 | 85 | 85 | 51.5–98 | 70.39 |
| ptype_31 | C | 93,178 | 76.66 | 77 | 14.29 | 58 | 58 | 86 | 97 | 50.5–100 | 81.89 |
| ptype_31 | E | 93,178 | 76.05 | 68 | 16.40 | 53 | 61 | 98 | 98 | 52–98 | 70.65 |
| ptype_31 | A | 93,178 | 71.16 | 74 | 11.60 | 54 | 59 | 79 | 93 | 53–98 | 74.61 |
| ptype_31 | N | 93,178 | 67.46 | 67 | 8.79 | 55 | 62 | 71 | 86 | 54–99 | 72.95 |
| ptype_9 | O | 345,069 | 19.22 | 12 | 16.98 | 1 | 3 | 38 | 50 | 0–50 | 21.13 |
| ptype_9 | C | 345,069 | 76.64 | 75 | 13.16 | 54 | 61 | 89 | 95 | 51–98 | 77.90 |
| ptype_9 | E | 345,069 | 12.25 | 9 | 13.43 | 1 | 1 | 15 | 46 | 0–50 | 16.26 |
| ptype_9 | A | 345,069 | 19.42 | 18 | 16.30 | 1 | 4 | 33 | 50 | 0–50 | 18.49 |
| ptype_9 | N | 345,069 | 84.60 | 90 | 12.06 | 61 | 72 | 94 | 98 | 52–99 | 84.40 |
| ptype_23 | O | 486 | 83.52 | 91 | 13.22 | 54 | 79 | 91 | 91 | 54–91 | 70.20 |
| ptype_23 | C | 486 | 14.01 | 12 | 3.43 | 12 | 12 | 19 | 20 | 2–20 | 14.40 |
| ptype_23 | E | 486 | 60.52 | 56 | 9.90 | 56 | 56 | 59 | 85 | 56–85 | 71.20 |
| ptype_23 | A | 486 | 58.89 | 51 | 13.95 | 51 | 51 | 69 | 94 | 51–94 | 74.60 |
| ptype_23 | N | 486 | 75.27 | 75 | 6.11 | 63 | 75 | 75 | 88 | 56–88 | 69.80 |

The table’s mean is also the row-weighted centroid. The separate unique-tuple
column shows how prolific authors/trait tuples can shift row weighting. This is
especially visible for sparse `ptype_23` in O, E, and A.

## ptype_0 versus ptype_31

| trait | ptype_0 centroid | ptype_31 centroid | actual difference |
|---|---:|---:|---:|
| O | 24.55 | 68.91 | 44.36 |
| C | 27.70 | 76.66 | 48.96 |
| E | 23.52 | 76.05 | 52.53 |
| A | 21.73 | 71.16 | 49.43 |
| N | 23.50 | 67.46 | 43.96 |

- Row-weighted five-dimensional distance: **107.24**.
- Unique-tuple-weighted distance: **112.37**.
- Idealized binary 0/100-corner distance: **223.61**.
- Actual row-weighted separation is **47.96%** of the binary-corner distance.

Thus `ptype_0` and `ptype_31` are opposite threshold cells, not empirical
extreme endpoints. Neither class centroid lies near an all-0 or all-100 corner.

## Pilot pairwise comparison

Activation values below are fixed middle-layer (8–24) median prompt-pair
distances. Effect contrast is the absolute difference between each adapter’s
mean delta-L2 magnitude; it is included descriptively because effect magnitude
is not itself a pairwise direction.

| pair | row centroid | unique centroid | activation | effect contrast | behavioral A | behavioral B | behavioral C |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–31 | 107.24 | 112.37 | 3.94 | 0.06 | 2.30 | 5.05 | 0.95 |
| 0–9 | 79.31 | 77.46 | 4.08 | 0.49 | 5.85 | 5.31 | 1.49 |
| 0–23 | 95.37 | 98.24 | 5.35 | 4.55 | 2.56 | 4.56 | 1.32 |
| 31–9 | 97.52 | 93.17 | 4.11 | 0.43 | 4.04 | 1.14 | 0.79 |
| 31–23 | 67.76 | 67.56 | 5.21 | 4.61 | 0.92 | 1.00 | 1.89 |
| 9–23 | 109.70 | 113.23 | 5.42 | 5.04 | 4.03 | 1.65 | 2.53 |

Across only six pairs, row-weighted distance correlations were:

- activation-centroid median: Pearson **-0.147**, Spearman **-0.314**;
- middle-layer activation prompt distance: Pearson **-0.073**, Spearman **0.143**;
- effect-magnitude contrast: Pearson **-0.114**, Spearman **-0.086**;
- behavioral A/B/C: Pearson **0.176 / 0.125 / -0.076**.

These six-pair statistics are descriptive and severely underpowered. They do
not alter the prior corrected target-alignment tests.

## Within-class versus between-class variation

| adapter | within-class RMS | nearest class-centroid distance | ratio |
|---|---:|---:|---:|
| ptype_0 | 33.61 | 49.86 | 0.674 |
| ptype_31 | 29.40 | 47.67 | 0.617 |
| ptype_9 | 32.45 | 46.58 | 0.697 |
| ptype_23 | 22.72 | 50.13 | 0.453 |

Across all 32 classes, within-class RMS ranged from **32.0% to 84.3%** of the
nearest between-class centroid distance; no class exceeded 100%. Therefore the
audit does **not** find that within-class variance greatly exceeds between-class
separation under this preregistered RMS/nearest-centroid comparison.

It is nevertheless material. Thresholding discards substantial continuous
variation inside each cell, creates heterogeneous classes, and does not make
all-low/all-high cells extreme points. This is a limitation of the 32-class
training design, though not the stronger failure condition posed in the audit.

## Verification of target use

### Behavioral pilot: mixed

The behavioral analysis loaded row-weighted and unique-tuple-weighted targets
directly from the target table and reported both correlations. Its explicit
target-correlation authorization criterion was positive correlation with the
**row-weighted continuous target**. However, it also used binary bits for
adapter-minus-base target direction, endpoint-sign expectations, and Hamming
distance. Those binary checks were co-primary gate conditions.

Accordingly, it would be inaccurate to say every behavioral alignment test was
continuous. It is accurate to say continuous row-weighted correlation was the
primary correlation target and binary bits were never substituted as 0/100
continuous centroids.

### Activation study: continuous primary

The activation analysis loaded binary, row-weighted, and unique-tuple-weighted
targets separately and reported all three. The Category A decision then
explicitly filtered to `row_weighted` and `unique_tuple_weighted`; the binary
target could not satisfy Category A. This verifies continuous centroids as the
primary activation-alignment targets.

## Outputs

- `continuous_target_distributions_all_32.csv`
- `continuous_target_distributions_pilot.csv`
- `pilot_pairwise_continuous_comparisons.csv`
- `pilot_pairwise_distance_correlations.csv`
- `ptype_0_vs_ptype_31_training_separation.csv`
- `within_between_class_variance.csv`
- `continuous_target_distribution_audit.json`
- `continuous_target_distribution_audit.py`

No model inference or experiment was rerun. **Category B is retained.**
