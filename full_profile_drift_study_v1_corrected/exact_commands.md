# Exact execution commands

Paths below are the literal RunPod paths used. The pod endpoint command wrapper
was SSH from the local workstation; model work ran inside the pod.

## Catalog and preflight

```bash
PYTHONPATH=src /opt/adapter-verify-venv/bin/python -m full_profile_drift.catalog --project /workspace/full_profile_drift_study_v1 --adapters /workspace/adapters
PYTHONPATH=src /opt/adapter-verify-venv/bin/python -m full_profile_drift.preflight --project /workspace/full_profile_drift_study_v1 --adapters /workspace/adapters --model-snapshot /workspace/.hf-cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
```

## Training exposure

```bash
PYTHONPATH=src /opt/adapter-verify-venv/bin/python -m full_profile_drift.exposure --project /workspace/full_profile_drift_study_v1 --model-snapshot /workspace/.hf-cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --dataset-parquet /workspace/full_profile_drift_study_v1/audit/frozen_inputs/tan_pandora.parquet --tokenizer /workspace/adapters/ptype_0 --batch-size 512 --checkpoint-every 100
```

## Exact weight geometry and prompt freeze

```bash
PYTHONPATH=src /opt/adapter-verify-venv/bin/python -m full_profile_drift.weight_geometry --project /workspace/full_profile_drift_study_v1 --adapters /workspace/adapters --device cuda
PYTHONPATH=src /opt/adapter-verify-venv/bin/python -m full_profile_drift.prompts --legacy-raw audit/frozen_inputs/legacy_prompts_raw.json --tokenizer /workspace/adapters/ptype_0 --output-dir prompt_manifest
PYTHONPATH=src /opt/adapter-verify-venv/bin/python -m full_profile_drift.manifest --project /workspace/full_profile_drift_study_v1
```

## GPU smoke

```bash
PYTHONPATH=src /opt/adapter-verify-venv/bin/python -m full_profile_drift.smoke --project /workspace/full_profile_drift_study_v1 --model-cache /workspace/.hf-cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --adapter-root /workspace/adapters --batch-size 6
```

## Full capture

Base and ptype_0 through ptype_11 were written to the network project root:

```bash
PYTHONPATH=src /opt/adapter-verify-venv/bin/python -m full_profile_drift.runner --model-key base --project /workspace/full_profile_drift_study_v1 --model-cache /workspace/.hf-cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --adapter-root /workspace/adapters --batch-size 8
for i in $(seq 0 31); do PYTHONPATH=src /opt/adapter-verify-venv/bin/python -m full_profile_drift.runner --model-key ptype_$i --project /workspace/full_profile_drift_study_v1 --model-cache /workspace/.hf-cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --adapter-root /workspace/adapters --batch-size 8 || exit 1; done
```

The network volume raised hidden quota error 122 during ptype_12. No completed
shard was changed or removed. Ptype_12 through ptype_31 were resumed to the
verified 52 GB pod overlay using the storage-routing-only runner patch:

```bash
for i in $(seq 12 31); do PYTHONPATH=/root/full_profile_drift_spill/src /opt/adapter-verify-venv/bin/python -m full_profile_drift.runner --model-key ptype_$i --project /workspace/full_profile_drift_study_v1 --storage-root /root/full_profile_drift_spill --model-cache /workspace/.hf-cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --adapter-root /workspace/adapters --batch-size 8 || exit 1; done
PYTHONPATH=/root/full_profile_drift_spill/src /opt/adapter-verify-venv/bin/python -m full_profile_drift.index_capture --project /workspace/full_profile_drift_study_v1 --spill /root/full_profile_drift_spill
```

## Continuations and analyses

```bash
for key in base $(for i in $(seq 0 31); do echo ptype_$i; done); do PYTHONPATH=/root/full_profile_drift_spill/src /opt/adapter-verify-venv/bin/python -m full_profile_drift.continuation --model-key $key --project /workspace/full_profile_drift_study_v1 --output-root /root/full_profile_drift_spill --model-cache /workspace/.hf-cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --adapter-root /workspace/adapters --batch-size 16 || exit 1; done
PYTHONPATH=/root/full_profile_drift_spill/src /opt/adapter-verify-venv/bin/python -m full_profile_drift.analyze --project /workspace/full_profile_drift_study_v1 --capture-root /root/full_profile_drift_spill --output-root /root/full_profile_drift_spill
PYTHONPATH=/root/full_profile_drift_spill/src /opt/adapter-verify-venv/bin/python -m full_profile_drift.supplemental --project /workspace/full_profile_drift_study_v1 --output-root /root/full_profile_drift_spill
PYTHONPATH=/root/full_profile_drift_spill/src /opt/adapter-verify-venv/bin/python -m full_profile_drift.continuation_analysis --project /workspace/full_profile_drift_study_v1 --output-root /root/full_profile_drift_spill
PYTHONPATH=/root/full_profile_drift_spill/src /opt/adapter-verify-venv/bin/python -m full_profile_drift.figures --project /workspace/full_profile_drift_study_v1 --output-root /root/full_profile_drift_spill
PYTHONPATH=/root/full_profile_drift_spill/src /opt/adapter-verify-venv/bin/python -m full_profile_drift.report --project /workspace/full_profile_drift_study_v1 --output-root /root/full_profile_drift_spill
```
