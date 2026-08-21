"""One-model atomic/resumable capture CLI; never loads two adapters concurrently."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .capture import capture_prompts, paths, valid_shard, write_group_shards
from .io import atomic_json


GROUPS=("ipip_stems","neutral_controls","naturalistic_behavioral")


def load_npz(path: Path) -> np.ndarray:
    with np.load(path) as f: return f[f.files[0]]


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--model-key",required=True); ap.add_argument("--project",type=Path,required=True); ap.add_argument("--storage-root",type=Path); ap.add_argument("--model-cache",type=Path,required=True); ap.add_argument("--adapter-root",type=Path,required=True); ap.add_argument("--batch-size",type=int,default=8); args=ap.parse_args(); storage=args.storage_root or args.project
    if args.model_key!="base" and args.model_key not in {f"ptype_{i}" for i in range(32)}: raise ValueError(args.model_key)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    prompts=json.loads((args.project/"prompt_manifest/prompt_manifest.json").read_text()); study=json.loads((args.project/"study_manifest.json").read_text()); catalog=__import__("pandas").read_csv(args.project/"ptype_catalog.csv")
    if prompts["prompt_manifest_fingerprint"]!=study["prompt_manifest_fingerprint"]: raise ValueError("prompt fingerprint mismatch")
    tokenizer=AutoTokenizer.from_pretrained("/workspace/adapters/ptype_0",local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(str(args.model_cache),local_files_only=True,dtype=torch.bfloat16,device_map="auto").eval()
    sentinel=prompts["records"][0]["text"]; encoded={k:v.to(next(model.parameters()).device) for k,v in tokenizer(sentinel,return_tensors="pt").items()}
    with torch.inference_mode(): initial=model(**encoded).logits[:,-1,:].float().cpu()
    torch.cuda.empty_cache(); memory_before={"allocated":torch.cuda.memory_allocated(),"reserved":torch.cuda.memory_reserved()}
    adapter_hash=None
    if args.model_key!="base":
        row=catalog[catalog.adapter==args.model_key].iloc[0]; adapter_hash=row.adapter_sha256; model=PeftModel.from_pretrained(model,str(args.adapter_root/args.model_key),is_trainable=False).eval()
    completed=[]; started=time.time(); grouped={g:[x for x in prompts["records"] if x["group"]==g] for g in GROUPS}
    for group,records in grouped.items():
        expected={"model_key":args.model_key,"prompt_group":group,"prompt_manifest_fingerprint":study["prompt_manifest_fingerprint"],"study_manifest_fingerprint":study["study_manifest_fingerprint"],"base_revision":study["base_revision"]}
        shard_specs=[(*paths(storage,"activation_outputs",args.model_key,group,p),expected|{"representation":"activation"}) for p in ("final_token","mean_tokens")]
        shard_specs.append((*paths(storage,"logit_outputs",args.model_key,group),expected|{"representation":"final_token_logits"}))
        if all(valid_shard(dp,mp,ex) for dp,mp,ex in shard_specs):
            completed.extend(json.loads(mp.read_text()) for _,mp,_ in shard_specs); continue
        arrays=capture_prompts(model,tokenizer,records,args.batch_size)
        base=None
        if args.model_key!="base":
            base={p:load_npz(paths(args.project,"activation_outputs","base",group,p)[0]) for p in ("final_token","mean_tokens")}
            base["logits"]=load_npz(paths(args.project,"logit_outputs","base",group)[0])
        completed.extend(write_group_shards(storage,args.model_key,group,records,arrays,base,study,adapter_hash,args.batch_size))
        atomic_json(storage/"progress.json",{"project":"full_profile_drift_study_v1","stage":"capture","model_key":args.model_key,"prompt_group_complete":group,"storage_root":str(storage),"integrity_failure":None})
    restoration=None
    if args.model_key!="base":
        model=model.unload().eval(); torch.cuda.empty_cache()
        with torch.inference_mode(): restored=model(**encoded).logits[:,-1,:].float().cpu()
        delta=restored-initial; torch.cuda.empty_cache(); memory_after={"allocated":torch.cuda.memory_allocated(),"reserved":torch.cuda.memory_reserved()}
        restoration={"logits":{"max_abs":float(delta.abs().max()),"mean_abs":float(delta.abs().mean()),"l2":float(delta.norm())},"exact":bool(torch.equal(restored,initial)),"memory_before_attachment":memory_before,"memory_after_unload_empty_cache":memory_after}
        if not restoration["exact"]: raise RuntimeError("base restoration failure")
    record={"model_key":args.model_key,"human_readable_profile":"untouched base" if args.model_key=="base" else catalog[catalog.adapter==args.model_key].iloc[0].human_readable_profile,"command_line":sys.argv,"batch_size":args.batch_size,"elapsed_seconds":time.time()-started,"restoration":restoration,"study_manifest_fingerprint":study["study_manifest_fingerprint"],"prompt_manifest_fingerprint":prompts["prompt_manifest_fingerprint"],"python":platform.python_version(),"packages":subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True).splitlines(),"completed_shards":completed}
    atomic_json(storage/"audit/run_records"/f"{args.model_key}.json",record); atomic_json(storage/"progress.json",{"project":"full_profile_drift_study_v1","stage":"capture_model_complete","model_key":args.model_key,"storage_root":str(storage),"integrity_failure":None}); print(json.dumps({"model_key":args.model_key,"elapsed_seconds":record["elapsed_seconds"],"storage_root":str(storage),"restoration":restoration}),flush=True)


if __name__=="__main__": main()
