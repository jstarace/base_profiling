"""Pooled residual and full final-token-logit capture for frozen prompts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import BASE_REVISION, LAYERS
from .io import atomic_json, atomic_npz, sha256


def decoder_layers(model):
    for path in ("model.layers","base_model.model.model.layers","base_model.model.layers"):
        try:
            layers=model.get_submodule(path)
            if len(layers)==32: return layers
        except (AttributeError,KeyError,TypeError): pass
    raise ValueError("could not locate exact 32 decoder blocks")


def capture_prompts(model,tokenizer,prompts,batch_size=8,selected_layers=LAYERS):
    import torch
    tokenizer.pad_token=tokenizer.eos_token; tokenizer.padding_side="left"; selected_layers=tuple(selected_layers)
    captured={}; handles=[]
    for layer_index,layer in enumerate(decoder_layers(model)):
        if layer_index in selected_layers:
            handles.append(layer.register_forward_hook(lambda module,args,output,index=layer_index:captured.__setitem__(index,(output[0] if isinstance(output,tuple) else output).detach())))
    finals=[]; means=[]; logits=[]
    try:
        for start in range(0,len(prompts),batch_size):
            batch=prompts[start:start+batch_size]; texts=[x["text"] for x in batch]
            for text,record in zip(texts,batch):
                if tokenizer.encode(text,add_special_tokens=True)!=record["token_ids"]: raise ValueError("prompt token IDs differ from frozen manifest")
            encoded=tokenizer(texts,return_tensors="pt",padding=True,add_special_tokens=True)
            device=next(model.parameters()).device; encoded={k:v.to(device) for k,v in encoded.items()}; captured.clear()
            with torch.inference_mode(): output=model(**encoded)
            if set(captured)!=set(selected_layers): raise ValueError("incomplete hook capture")
            states=torch.stack([captured[index].float().cpu() for index in selected_layers],dim=1)
            mask=encoded["attention_mask"].float().cpu()[:,None,:,None]
            finals.append(states[:,:,-1,:].numpy().astype("float32",copy=False))
            means.append(((states*mask).sum(dim=2)/mask.sum(dim=2)).numpy().astype("float32",copy=False))
            logits.append(output.logits[:,-1,:].detach().to(torch.float16).cpu().numpy())
    finally:
        for handle in handles: handle.remove()
    result={"final_token":np.concatenate(finals),"mean_tokens":np.concatenate(means),"logits":np.concatenate(logits)}
    if not all(np.isfinite(x).all() for x in result.values()): raise ValueError("nonfinite activation or logits")
    return result


def paths(root: Path,kind: str,model_key: str,group: str,pooling: str|None=None):
    stem=root/kind/"shards"/model_key/group/(pooling or "final_token_logits")
    return stem.with_suffix(".npz"),stem.with_suffix(".metadata.json")


def valid_shard(data_path: Path,metadata_path: Path,expected: dict) -> bool:
    if not data_path.exists() or not metadata_path.exists(): return False
    metadata=json.loads(metadata_path.read_text())
    if any(metadata.get(k)!=v for k,v in expected.items()): raise ValueError(f"resume metadata mismatch: {data_path}")
    if metadata.get("data_sha256")!=sha256(data_path): raise ValueError(f"resume checksum mismatch: {data_path}")
    return True


def write_shard(data_path: Path,metadata_path: Path,array_name: str,array: np.ndarray,metadata: dict,prompt_ids: list[str],layers: tuple[int,...]|None=None) -> dict:
    if not np.isfinite(array).all(): raise ValueError(f"nonfinite shard: {data_path}")
    expected={k:metadata[k] for k in ("model_key","prompt_group","representation","prompt_manifest_fingerprint","study_manifest_fingerprint","base_revision")}
    if valid_shard(data_path,metadata_path,expected): return json.loads(metadata_path.read_text())
    arrays={array_name:array,"prompt_ids":np.asarray(prompt_ids)}
    if layers is not None: arrays["layers"]=np.asarray(layers,dtype="int16")
    atomic_npz(data_path,**arrays)
    payload=metadata|{"array_name":array_name,"array_shape":list(array.shape),"dtype":str(array.dtype),"data_sha256":sha256(data_path)}
    atomic_json(metadata_path,payload); return payload


def write_group_shards(project: Path,model_key: str,group: str,records: list[dict],arrays: dict,base: dict|None,study: dict,adapter_hash: str|None,batch_size: int):
    ids=[x["prompt_id"] for x in records]; common={"model_key":model_key,"prompt_group":group,"prompt_manifest_fingerprint":study["prompt_manifest_fingerprint"],"study_manifest_fingerprint":study["study_manifest_fingerprint"],"base_revision":BASE_REVISION,"tokenizer_hash":study["tokenizer_sha256"],"adapter_hash":adapter_hash,"code_version":study["code_version"],"batch_size":batch_size,"prompt_ids":ids}
    completed=[]
    for pooling in ("final_token","mean_tokens"):
        data_path,metadata_path=paths(project,"activation_outputs",model_key,group,pooling)
        value=arrays[pooling] if model_key=="base" else (arrays[pooling]-base[pooling]).astype("float32",copy=False)
        name="base_activation" if model_key=="base" else "delta_h"
        completed.append(write_shard(data_path,metadata_path,name,value.astype("float32",copy=False),common|{"representation":"activation","pooling_rule":pooling,"layer_ids":list(LAYERS)},ids,LAYERS))
    data_path,metadata_path=paths(project,"logit_outputs",model_key,group)
    value=arrays["logits"] if model_key=="base" else (arrays["logits"].astype("float32")-base["logits"].astype("float32")).astype("float16")
    name="base_logits" if model_key=="base" else "delta_logits"
    completed.append(write_shard(data_path,metadata_path,name,value.astype("float16",copy=False),common|{"representation":"final_token_logits","vocabulary_size":int(value.shape[1])},ids))
    return completed
