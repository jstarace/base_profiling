"""Streaming, resumable tokenizer-exposure audit for all 32 ptypes."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from full_profile_drift.io import atomic_json


def empty_stats() -> dict:
    return {str(p):{"rows":0,"raw_tokens":0,"retained_tokens":0,"truncated":0,"sum_len":0,"sum_sq_len":0,"hist":{}} for p in range(32)}


def percentile(hist: dict[str, int], q: float) -> float:
    items = sorted((int(k), int(v)) for k,v in hist.items())
    n = sum(v for _,v in items)
    if not n: return float("nan")
    target = q * (n - 1)
    lo, hi = math.floor(target), math.ceil(target)
    def value_at(rank: int) -> int:
        seen = 0
        for value,count in items:
            if seen + count > rank: return value
            seen += count
        return items[-1][0]
    a,b=value_at(lo),value_at(hi)
    return float(a + (b-a)*(target-lo))


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--project",type=Path,required=True); p.add_argument("--model-snapshot",type=Path,required=True); p.add_argument("--dataset-parquet",type=Path,required=True); p.add_argument("--tokenizer",type=Path,required=True); p.add_argument("--batch-size",type=int,default=512); p.add_argument("--checkpoint-every",type=int,default=100); args=p.parse_args()
    out=args.project/"training_exposure"; out.mkdir(parents=True,exist_ok=True); checkpoint=out/"exposure_checkpoint.json"
    state=json.loads(checkpoint.read_text()) if checkpoint.exists() else {"next_index":0,"stats":empty_stats(),"complete":False}
    if state.get("complete") and (out/"all_32_training_exposure.csv").exists(): print("EXPOSURE_ALREADY_COMPLETE"); return
    parquet=pq.ParquetFile(args.dataset_parquet); total_rows=parquet.metadata.num_rows
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    start=int(state["next_index"]); stats=state["stats"]
    batch_number=0; offset=0
    for record_batch in parquet.iter_batches(batch_size=args.batch_size,columns=["text","ptype"]):
        original_rows=record_batch.num_rows
        if offset + original_rows <= start:
            offset += original_rows
            continue
        if start > offset:
            record_batch=record_batch.slice(start-offset)
            offset=start
        batch_number += 1
        batch=record_batch.to_pydict(); texts=batch["text"]; ptypes=batch["ptype"]
        encoded=tokenizer(texts,add_special_tokens=True,truncation=False,return_attention_mask=False)
        for ptype,ids in zip(ptypes,encoded["input_ids"]):
            key=str(int(ptype)); length=len(ids); retained=min(length,512); item=stats[key]
            item["rows"]+=1; item["raw_tokens"]+=length; item["retained_tokens"]+=retained; item["truncated"]+=int(length>512); item["sum_len"]+=length; item["sum_sq_len"]+=length*length; item["hist"][str(length)]=item["hist"].get(str(length),0)+1
        next_index=offset+record_batch.num_rows; offset=next_index
        if batch_number % args.checkpoint_every == 0 or next_index == total_rows:
            atomic_json(checkpoint,{"next_index":next_index,"total_rows":total_rows,"stats":stats,"complete":next_index==total_rows})
            atomic_json(args.project/"progress.json",{"project":"full_profile_drift_study_v1","stage":"training_exposure","exposure_rows_complete":next_index,"exposure_rows_total":total_rows,"integrity_failure":None})
            print(json.dumps({"rows":next_index,"total":total_rows}),flush=True)
    catalog=pd.read_csv(args.project/"ptype_catalog.csv").set_index("ptype"); rows=[]
    for ptype in range(32):
        item=stats[str(ptype)]; n=item["rows"]; mean=item["sum_len"]/n; variance=max(item["sum_sq_len"]/n-mean*mean,0)
        rows.append({"ptype":ptype,"adapter":f"ptype_{ptype}","human_readable_profile":catalog.loc[ptype,"human_readable_profile"],"total_raw_rows":n,"total_tokens_before_truncation":item["raw_tokens"],"total_retained_tokens_after_512":item["retained_tokens"],"mean_token_length":mean,"median_token_length":percentile(item["hist"],.5),"token_length_std_population":math.sqrt(variance),"token_length_p05":percentile(item["hist"],.05),"token_length_p25":percentile(item["hist"],.25),"token_length_p75":percentile(item["hist"],.75),"token_length_p95":percentile(item["hist"],.95),"fraction_rows_truncated":item["truncated"]/n,"nominal_optimizer_steps":int(catalog.loc[ptype,"nominal_optimizer_steps"]),"estimated_retained_tokens_processed_training":item["retained_tokens"],"unique_trait_tuple_proxy_count":int(catalog.loc[ptype,"unique_trait_tuple_proxy_count"])})
    pd.DataFrame(rows).to_csv(out/"all_32_training_exposure.csv",index=False)
    atomic_json(out/"exposure_summary.json",{"complete":True,"rows":total_rows,"max_length":max(int(k) for s in stats.values() for k in s["hist"]),"tokenizer":str(args.tokenizer),"dataset_parquet":str(args.dataset_parquet),"max_seq_length":512})
    atomic_json(args.project/"progress.json",{"project":"full_profile_drift_study_v1","stage":"training_exposure_complete","exposure_rows_complete":total_rows,"integrity_failure":None})


if __name__=="__main__": main()
