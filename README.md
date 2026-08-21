# Base Profiling

This repository contains the data pipeline, training code, verification tools,
and corrected analysis record for an exploratory study of 32 OCEAN-profile
LoRA adapters trained on `meta-llama/Llama-3.1-8B`.

The main result is narrower than the original hypothesis. All 32 adapters
developed strong, reproducible adapter-specific identities. Unequal training
exposure strongly shaped parameter magnitude, most learned geometry was
idiosyncratic, and only a weak aggregate OCEAN component appeared in activation
and logit representations. These results do not establish human personality in
a language model or validate the profile labels as human traits.

## Study at a glance

- Base model: `meta-llama/Llama-3.1-8B`, frozen revision
  `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`
- Profiles: all 32 combinations of high and low O, C, E, A, and N labels
- Training data: 3,006,566 PANDORA comments with continuous Big Five scores
- Measurements: LoRA weight updates, hidden-state activations, output logits,
  deterministic continuations, and an IPIP-NEO-120 behavioral pilot
- Correction policy: analysis-only; no adapter was retrained, merged, edited,
  or rerun to produce the corrected package

## Key findings

- Every adapter was readable, finite, loadable, and produced a nonzero effect.
- Adapter identity was strongly reproducible across internal representations.
- Retained-token exposure correlated strongly with LoRA update magnitude
  (`rho = 0.944`), moderately with activation magnitude (`rho = 0.487`), and
  weakly with logit magnitude (`rho = 0.174`).
- Aggregate OCEAN main-effect energy exceeded a shuffled-label baseline in
  activation and logit space, but the all-profile continuous alignment was weak.
- Exposure-matched analyses found modest continuous alignment and are reported
  as exploratory sensitivity analyses, not confirmatory evidence.

The corrected decision record contains the statistical results and multiplicity
rules: [`full_profile_drift_decision.md`](full_profile_drift_study_v1_corrected/full_profile_drift_decision.md).

## Repository guide

- [`dataset_construction/`](dataset_construction/) builds and cleans the source
  text dataset.
- [`training/`](training/) partitions records into the 32 profile classes,
  trains one adapter at a time, and contains the independent adapter verifier.
- [`adapter_verification/`](adapter_verification/) contains the machine-readable
  and human-readable verification reports for all 32 adapters.
- [`profile_mapping_audit_v1/`](profile_mapping_audit_v1/) traces the profile
  encoding from raw traits through training and downstream target lookup.
- [`full_profile_drift_study_v1_corrected/`](full_profile_drift_study_v1_corrected/)
  is the corrected, checksum-covered analysis package and source of record.

## Data and benchmark attribution

The adapters were trained on the
[`jingjietan/pandora-big5`](https://huggingface.co/datasets/jingjietan/pandora-big5)
redistribution of the PANDORA Reddit corpus. The Hugging Face repository labels
that redistribution Apache-2.0. The underlying dataset was introduced by
Gjurković et al. in
[*PANDORA Talks: Personality and Demographics on Reddit*](https://aclanthology.org/2021.socialnlp-1.12/)
(SocialNLP 2021, DOI `10.18653/v1/2021.socialnlp-1.12`).

The behavioral pilot used Johnson's public-domain IPIP-NEO-120 items. The
canonical item and scoring page is maintained by the
[International Personality Item Pool](https://ipip.ori.org/30FacetNEO-PI-RItems.htm).
The associated publication is Johnson (2014), *Measuring thirty facets of the
Five Factor Model with a 120-item public domain inventory: Development of the
IPIP-NEO-120*, DOI `10.1016/j.jrp.2014.05.003`.

Additional ingestion code references `jingjietan/kaggle-mbti`,
`minhaozhang/mbti`, and `jingjietan/essays-big5`. Their roles, source links,
known licenses, and provenance limitations are recorded in
[`DATASET_ATTRIBUTION.md`](DATASET_ATTRIBUTION.md). No raw source dataset is
redistributed in this repository.

The verified profile encoding is:

```text
ptype = 16*O_high + 8*C_high + 4*E_high + 2*A_high + N_high
```

Each high bit uses strict `> 50` thresholding in the order O, C, E, A, N.
The mapping audit found zero threshold mismatches across 3,006,566 rows.

## Reproducibility

Install the pinned top-level dependencies in Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the corrected package from its directory:

```bash
cd full_profile_drift_study_v1_corrected
PYTHONPATH=src python -m pytest
sha256sum -c ARTIFACT_CHECKSUMS.sha256
```

The corrected package documents its exact base-model revision, environments,
commands, source hashes, frozen inputs, and evidential boundaries. Adapter
weights, raw corpora, private handoff files, scratch captures, and duplicate ZIP
archives are intentionally not tracked here.

## Status

This is an archival research release. The corrected package supersedes the
earlier local study directory. No model experiment was rerun to produce the
correction.
