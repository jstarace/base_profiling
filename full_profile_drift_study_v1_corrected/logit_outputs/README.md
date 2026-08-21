# Raw logit storage

The float16 full-vocabulary adapter-minus-base logit shards are intentionally
not duplicated into the compact local package. Authoritative shards are split
between:

- `/workspace/full_profile_drift_study_v1/logit_outputs` for base and ptype_0
  through ptype_11;
- `/root/full_profile_drift_spill/logit_outputs` for ptype_12 through ptype_31.

See `../audit/capture_shard_index.csv` for every path, shape, dtype, adapter
hash, prompt/study fingerprint, and SHA-256.
