"""Bounded deterministic continuation capture for the 120 legacy naturalistic stems."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from .io import atomic_json


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--model-key",required=True); ap.add_argument("--project",type=Path,required=True); ap.add_argument("--output-root",type=Path); ap.add_argument("--model-cache",type=Path,required=True); ap.add_argument("--adapter-root",type=Path,required=True); ap.add_argument("--batch-size",type=int,default=8); args=ap.parse_args(); output_root=args.output_root or args.project
    output=output_root/"continuation_outputs"/"conditions"/f"{args.model_key}.json"
    if output.exists(): print(f"CONTINUATION_ALREADY_COMPLETE {args.model_key}"); return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    manifest=json.loads((args.project/"prompt_manifest/prompt_manifest.json").read_text()); prompts=[r for r in manifest["records"] if r["group"]=="naturalistic_behavioral" and r["legacy_core"]]
    if len(prompts)!=120: raise RuntimeError(f"expected 120 frozen continuation stems, got {len(prompts)}")
    tokenizer=AutoTokenizer.from_pretrained(str(args.adapter_root/"ptype_0"),local_files_only=True); tokenizer.pad_token=tokenizer.eos_token; tokenizer.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(str(args.model_cache),local_files_only=True,dtype=torch.bfloat16,device_map="auto").eval()
    sentinel={k:v.to(next(model.parameters()).device) for k,v in tokenizer(prompts[0]["text"],return_tensors="pt").items()}
    with torch.inference_mode(): pristine=model(**sentinel).logits[:,-1,:].float().cpu()
    if args.model_key!="base": model=PeftModel.from_pretrained(model,str(args.adapter_root/args.model_key),is_trainable=False).eval()
    rows=[]; started=time.time()
    for start in range(0,len(prompts),args.batch_size):
        batch=prompts[start:start+args.batch_size]; encoded=tokenizer([x["text"] for x in batch],return_tensors="pt",padding=True,add_special_tokens=True); input_length=encoded["input_ids"].shape[1]; encoded={k:v.to(next(model.parameters()).device) for k,v in encoded.items()}
        with torch.inference_mode(): generated=model.generate(**encoded,max_new_tokens=64,do_sample=False,num_beams=1,use_cache=True,pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id)
        continuation_ids=generated[:,input_length:].cpu().tolist()
        for prompt,ids in zip(batch,continuation_ids):
            if tokenizer.eos_token_id in ids: ids=ids[:ids.index(tokenizer.eos_token_id)+1]
            text=tokenizer.decode(ids,skip_special_tokens=True); rows.append({"model_key":args.model_key,"prompt_id":prompt["prompt_id"],"category":prompt["category"],"prompt_text":prompt["text"],"continuation_token_ids":ids,"continuation_token_count":len(ids),"continuation_text":text,"continuation_sha256":hashlib.sha256(text.encode()).hexdigest()})
    restoration=None
    if args.model_key!="base":
        model=model.unload().eval(); torch.cuda.empty_cache()
        with torch.inference_mode(): restored=model(**sentinel).logits[:,-1,:].float().cpu()
        restoration={"exact":bool(torch.equal(restored,pristine)),"max_abs":float((restored-pristine).abs().max())}
        if not restoration["exact"]: raise RuntimeError("base restoration failure after continuation")
    payload={"schema_version":"1.0","model_key":args.model_key,"prompt_manifest_fingerprint":manifest["prompt_manifest_fingerprint"],"generation_config":{"max_new_tokens":64,"do_sample":False,"num_beams":1,"eos_token_id":tokenizer.eos_token_id,"pad_token_id":tokenizer.eos_token_id},"command_line":sys.argv,"elapsed_seconds":time.time()-started,"restoration":restoration,"records":rows}
    atomic_json(output,payload); atomic_json(output_root/"progress.json",{"project":"full_profile_drift_study_v1","stage":"continuation_model_complete","model_key":args.model_key,"integrity_failure":None}); print(json.dumps({"model_key":args.model_key,"records":len(rows),"elapsed_seconds":payload["elapsed_seconds"],"restoration":restoration}),flush=True)


if __name__=="__main__": main()
