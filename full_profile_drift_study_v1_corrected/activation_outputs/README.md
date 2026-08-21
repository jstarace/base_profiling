# Raw activation storage

The float32 pooled activation shards are intentionally not duplicated into the
compact local package. Authoritative shards are split between:

- `/workspace/full_profile_drift_study_v1/activation_outputs` for base and
  ptype_0 through ptype_11;
- `/root/full_profile_drift_spill/activation_outputs` for ptype_12 through
  ptype_31.

See `../audit/capture_shard_index.csv` for every path, shape, dtype, adapter
hash, prompt/study fingerprint, and SHA-256.
