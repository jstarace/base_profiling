"""Build the versioned study manifest from frozen inputs and source hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from full_profile_drift import BASE_MODEL, BASE_REVISION
from full_profile_drift.io import atomic_json


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--project",type=Path,required=True); args=p.parse_args()
    source_files=sorted(path for path in (args.project/"src").rglob("*.py"))
    frozen_files=sorted(path for path in (args.project/"audit/frozen_inputs").iterdir() if path.is_file())
    prompt_manifest=json.loads((args.project/"prompt_manifest/prompt_manifest.json").read_text())
    preflight=json.loads((args.project/"audit/preflight.json").read_text())
    source_hashes={path.relative_to(args.project).as_posix():sha256(path) for path in source_files}
    code_version=hashlib.sha256(json.dumps(source_hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    payload={
      "study_name":"full_profile_drift_study_v1","purpose":"exploratory complete geometry of all 32 existing OCEAN-profile LoRA adapters",
      "base_model":BASE_MODEL,"base_revision":BASE_REVISION,"offline_loading":True,"deterministic_inference":True,
      "adapter_lifecycle":"one adapter at a time; unload; exact base-logit and GPU-memory restoration",
      "adapter_conditions":[f"ptype_{p}" for p in range(32)],"profile_encoding":"16*O_high + 8*C_high + 4*E_high + 2*A_high + N_high","factorial_coding":{"low":-1,"high":1},
      "max_input_tokens":512,"capture_layers":list(range(32)),"pooling_rules":["final_token","mean_tokens"],"activation_storage_dtype":"float32","logit_storage":"full adapter-minus-base final-input-token vectors float16; analysis float32",
      "prompt_plan":{"total":1080,"ipip_stems":120,"neutral_controls":240,"naturalistic":720,"naturalistic_categories":12,"per_category":60,"legacy_core_count":360,"split_seed":20260803,"splits":{"train":0.6,"validation":0.2,"test":0.2}},
      "random_seed":20260803,"bootstrap_resamples":1000,"adapter_label_permutations":100,"minimum_capture_free_bytes":60*1024**3,
      "fixed_layer_summaries":{"early":[0,7],"middle":[8,24],"late":[25,31]},
      "sensitivity_subsets":["all_32","rows_ge_1000","rows_ge_10000","rows_ge_50000","unique_tuples_ge_10","unique_tuples_ge_25","pair_row_ratio_le_2","pair_row_ratio_le_1_5","exposure_quartiles"],
      "forbidden_operations":["adapter retraining","adapter merging","adapter normalization during inference","modification of legacy evidence"],
      "prompt_manifest_fingerprint":prompt_manifest["prompt_manifest_fingerprint"],
      "tokenizer_sha256":preflight["tokenizer"]["tokenizer.json"]["adapter_0"],
      "code_version":code_version,
      "source_sha256":source_hashes,
      "frozen_input_sha256":{path.name:sha256(path) for path in frozen_files},
    }
    canonical=json.dumps(payload,sort_keys=True,separators=(",",":")); payload["study_manifest_fingerprint"]=hashlib.sha256(canonical.encode()).hexdigest(); atomic_json(args.project/"study_manifest.json",payload)
    print(payload["study_manifest_fingerprint"])


if __name__=="__main__": main()
