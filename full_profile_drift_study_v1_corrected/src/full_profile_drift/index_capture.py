"""Verify and index split-root raw capture without copying large arrays."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from .io import atomic_json,sha256

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project",type=Path,required=True); ap.add_argument("--spill",type=Path,required=True); args=ap.parse_args(); selected={}; duplicates=[]
    for priority,root in enumerate((args.project,args.spill)):
      for kind in ("activation_outputs","logit_outputs"):
       for metadata_path in (root/kind/"shards").rglob("*.metadata.json"):
        metadata=json.loads(metadata_path.read_text()); key=(metadata["model_key"],metadata["prompt_group"],metadata["representation"],metadata.get("pooling_rule")); data_path=Path(str(metadata_path).removesuffix(".metadata.json")+".npz"); valid=data_path.exists() and sha256(data_path)==metadata["data_sha256"]
        row={"model_key":key[0],"prompt_group":key[1],"representation":key[2],"pooling_rule":key[3],"storage_root":str(root),"data_path":str(data_path),"metadata_path":str(metadata_path),"array_name":metadata["array_name"],"array_shape":json.dumps(metadata["array_shape"]),"dtype":metadata["dtype"],"data_sha256":metadata["data_sha256"],"checksum_valid":valid,"adapter_hash":metadata.get("adapter_hash"),"prompt_manifest_fingerprint":metadata["prompt_manifest_fingerprint"],"study_manifest_fingerprint":metadata["study_manifest_fingerprint"]}
        if key in selected: duplicates.append(selected[key]|{"superseded_by":str(data_path)})
        selected[key]=row
    rows=list(selected.values()); frame=pd.DataFrame(rows).sort_values(["model_key","prompt_group","representation","pooling_rule"])
    expected={(model,group,rep,pool) for model in ["base"]+[f"ptype_{i}" for i in range(32)] for group in ("ipip_stems","neutral_controls","naturalistic_behavioral") for rep,pool in (("activation","final_token"),("activation","mean_tokens"),("final_token_logits",None))}; actual=set(selected); missing=sorted(expected-actual); extra=sorted(actual-expected)
    records=[]
    for root in (args.project,args.spill):
      for path in (root/"audit/run_records").glob("*.json"): records.append(json.loads(path.read_text()))
    by_model={r["model_key"]:r for r in records}; restoration_failures=[k for k,r in by_model.items() if k!="base" and not r.get("restoration",{}).get("exact")]
    out=args.spill/"audit"; out.mkdir(parents=True,exist_ok=True); frame.to_csv(out/"capture_shard_index.csv",index=False); atomic_json(out/"capture_shard_index.json",rows); pd.DataFrame(duplicates).to_csv(out/"preserved_duplicate_partial_shards.csv",index=False)
    summary={"pass":not missing and not extra and frame.checksum_valid.all() and len(by_model)==33 and not restoration_failures,"authoritative_shards":len(frame),"expected_shards":297,"conditions":len(by_model),"missing":missing,"extra":extra,"checksum_failures":frame.loc[~frame.checksum_valid,"data_path"].tolist(),"restoration_failures":restoration_failures,"preserved_superseded_partial_shards":len(duplicates),"storage_roots":[str(args.project),str(args.spill)]}; atomic_json(out/"capture_integrity_summary.json",summary); print(json.dumps(summary))
    if not summary["pass"]: raise SystemExit(1)
if __name__=="__main__":main()
