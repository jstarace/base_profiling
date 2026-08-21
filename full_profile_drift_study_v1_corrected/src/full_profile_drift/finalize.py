"""Build final compact-package checksums and source/audit manifest."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from .io import atomic_json

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project",type=Path,required=True); args=ap.parse_args(); root=args.project
    excluded={"ARTIFACT_CHECKSUMS.sha256","final_audit_manifest.json"}; files=sorted(x for x in root.rglob("*") if x.is_file() and x.name not in excluded and not ({"__pycache__",".pytest_cache",".venv","node_modules"}&set(x.parts)) and x.name not in {"tan_pandora.parquet",".DS_Store"} and not x.name.endswith((".pyc",".pyo")))
    hashes={x.relative_to(root).as_posix():sha(x) for x in files}; text="".join(f"{digest}  {path}\n" for path,digest in hashes.items()); (root/"ARTIFACT_CHECKSUMS.sha256").write_text(text)
    source={k:v for k,v in hashes.items() if k.startswith("src/")}; payload={"study":"full_profile_drift_study_v1_corrected","compact_files":len(hashes),"compact_hashes":hashes,"source_hashes":source,"raw_capture_index":"audit/capture_shard_index.csv","raw_capture_storage_note":"Raw arrays remain split across the RunPod network project and /root/full_profile_drift_spill; every authoritative shard is checksum-indexed.","correction_scope":"analysis-only over existing compact outputs","model_experiments_rerun":False,"frozen_legacy_inputs_modified":False,"adapter_training_or_modification":False}; atomic_json(root/"final_audit_manifest.json",payload); print(json.dumps({"files":len(hashes),"sources":len(source)}))
if __name__=="__main__":main()
