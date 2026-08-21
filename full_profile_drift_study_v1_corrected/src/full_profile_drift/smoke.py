"""GPU integrity smoke for base, ptype_0, ptype_16, and ptype_31."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .capture import capture_prompts
from .io import atomic_json


def memory(torch): return {"allocated":torch.cuda.memory_allocated(),"reserved":torch.cuda.memory_reserved()}


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--project",type=Path,required=True); ap.add_argument("--model-cache",type=Path,required=True); ap.add_argument("--adapter-root",type=Path,required=True); ap.add_argument("--batch-size",type=int,default=6); args=ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    manifest=json.loads((args.project/"prompt_manifest/prompt_manifest.json").read_text())
    records=[]
    for group in ("ipip_stems","neutral_controls","naturalistic_behavioral"): records.extend([r for r in manifest["records"] if r["group"]==group][:2])
    tokenizer=AutoTokenizer.from_pretrained(str(args.adapter_root/"ptype_0"),local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(str(args.model_cache),local_files_only=True,dtype=torch.bfloat16,device_map="auto").eval()
    sentinel={k:v.to(next(model.parameters()).device) for k,v in tokenizer(records[0]["text"],return_tensors="pt").items()}
    with torch.inference_mode(): pristine=model(**sentinel).logits[:,-1,:].float().cpu()
    base1=capture_prompts(model,tokenizer,records,args.batch_size); base2=capture_prompts(model,tokenizer,records,args.batch_size)
    base_repeat={k:bool(np.array_equal(base1[k],base2[k])) for k in base1}; reports=[]
    for key in ("ptype_0","ptype_16","ptype_31"):
        torch.cuda.empty_cache(); before=memory(torch); model=PeftModel.from_pretrained(model,str(args.adapter_root/key),is_trainable=False).eval()
        first=capture_prompts(model,tokenizer,records,args.batch_size); second=capture_prompts(model,tokenizer,records,args.batch_size)
        condition={"model_key":key,"finite":all(np.isfinite(first[k]).all() for k in first),"repeat_exact":{k:bool(np.array_equal(first[k],second[k])) for k in first},"shapes":{k:list(v.shape) for k,v in first.items()},"effect_max_abs":{k:float(np.max(np.abs(first[k].astype("float32")-base1[k].astype("float32")))) for k in first}}
        model=model.unload().eval(); torch.cuda.empty_cache()
        with torch.inference_mode(): restored=model(**sentinel).logits[:,-1,:].float().cpu()
        condition["base_restoration"]={"exact":bool(torch.equal(restored,pristine)),"max_abs":float((restored-pristine).abs().max()),"memory_before":before,"memory_after":memory(torch)}
        reports.append(condition)
        if not condition["finite"] or not all(condition["repeat_exact"].values()) or not condition["base_restoration"]["exact"] or not all(v>0 for v in condition["effect_max_abs"].values()): raise RuntimeError(f"smoke integrity failure: {key}")
    report={"pass":all(base_repeat.values()),"prompt_manifest_fingerprint":manifest["prompt_manifest_fingerprint"],"prompt_ids":[r["prompt_id"] for r in records],"base_repeat_exact":base_repeat,"base_shapes":{k:list(v.shape) for k,v in base1.items()},"adapter_conditions":reports}
    if not report["pass"]: raise RuntimeError("base determinism smoke failure")
    atomic_json(args.project/"audit/gpu_smoke/smoke_test_report.json",report); print(json.dumps(report),flush=True)


if __name__=="__main__": main()
