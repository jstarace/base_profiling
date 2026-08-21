"""Exact all-32 LoRA update geometry computed directly from low-rank factors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from scipy.cluster.hierarchy import linkage

from full_profile_drift.io import atomic_json


def effective_rank(eigenvalues: np.ndarray) -> float:
    values=np.clip(np.asarray(eigenvalues,dtype=np.float64),0,None)
    if values.sum() == 0: return 0.0
    p=values/values.sum(); p=p[p>0]
    return float(np.exp(-(p*np.log(p)).sum()))


def centered_eigensystem(gram: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    n=len(gram); h=np.eye(n)-np.ones((n,n))/n; centered=h@gram@h
    values,vectors=np.linalg.eigh((centered+centered.T)/2); order=np.argsort(values)[::-1]
    return np.clip(values[order],0,None),vectors[:,order]


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--project",type=Path,required=True); ap.add_argument("--adapters",type=Path,required=True); ap.add_argument("--device",default="cuda"); args=ap.parse_args()
    out=args.project/"weight_geometry"; out.mkdir(parents=True,exist_ok=True)
    catalog=pd.read_csv(args.project/"ptype_catalog.csv").sort_values("ptype"); names=catalog.adapter.tolist(); n=len(names)
    if n != 32: raise RuntimeError(f"expected 32 adapters, got {n}")
    paths=[args.adapters/name/"adapter_model.safetensors" for name in names]
    handles=[safe_open(str(path),framework="pt",device="cpu") for path in paths]
    a_keys=sorted(k for k in handles[0].keys() if ".lora_A." in k)
    if len(a_keys) != 224: raise RuntimeError(f"expected 224 LoRA A matrices, got {len(a_keys)}")
    aggregate=np.zeros((n,n),dtype=np.float64); block_rows=[]; pair_rows=[]
    for block_index,a_key in enumerate(a_keys):
        b_key=a_key.replace(".lora_A.",".lora_B.")
        if any(a_key not in h.keys() or b_key not in h.keys() for h in handles): raise RuntimeError(f"factor mismatch: {a_key}")
        aa=torch.stack([h.get_tensor(a_key).float() for h in handles]).to(args.device)
        bb=torch.stack([h.get_tensor(b_key).float() for h in handles]).to(args.device)
        # <B_i A_i, B_j A_j> = sum_rs <B_i[:,r],B_j[:,s]><A_i[r,:],A_j[s,:]>.
        gb=torch.einsum("ior,jos->ijrs",bb,bb)
        ga=torch.einsum("irk,jsk->ijrs",aa,aa)
        gram=(4.0*(gb*ga).sum(dim=(-1,-2))).double().cpu().numpy()
        gram=(gram+gram.T)/2; aggregate += gram
        diag=np.clip(np.diag(gram),0,None); denom=np.sqrt(np.outer(diag,diag)); cosine=np.divide(gram,denom,out=np.zeros_like(gram),where=denom>0)
        distance=np.sqrt(np.clip(diag[:,None]+diag[None,:]-2*gram,0,None))
        layer=int(a_key.split(".layers.")[1].split(".")[0]); module=a_key.split(".")[-3]
        vals,_=centered_eigensystem(gram)
        np.savez_compressed(out/f"layer_{layer:02d}_{module}_geometry.npz",gram=gram,cosine=cosine,distance=distance,ptypes=np.arange(32))
        block_rows.append({"layer":layer,"target_module":module,"matrix_shape":f"{bb.shape[1]}x{aa.shape[2]}","aggregate_update_frobenius_norm":float(np.sqrt(diag.sum())),"mean_profile_update_frobenius_norm":float(np.sqrt(diag).mean()),"profile_kernel_effective_rank":effective_rank(vals),"kernel_top_eigenvalue_fraction":float(vals[0]/vals.sum()) if vals.sum() else 0.0})
        iu=np.triu_indices(n,1)
        for i,j,co,di,inner in zip(iu[0],iu[1],cosine[iu],distance[iu],gram[iu]):
            pair_rows.append({"layer":layer,"target_module":module,"ptype_a":int(i),"profile_a":catalog.iloc[i].human_readable_profile,"ptype_b":int(j),"profile_b":catalog.iloc[j].human_readable_profile,"frobenius_inner_product":float(inner),"cosine_similarity":float(co),"frobenius_distance":float(di)})
        del aa,bb,gb,ga; torch.cuda.empty_cache()
        if (block_index+1)%7==0:
            atomic_json(args.project/"progress.json",{"project":"full_profile_drift_study_v1","stage":"weight_geometry","blocks_complete":block_index+1,"blocks_total":len(a_keys),"integrity_failure":None})
            print(json.dumps({"blocks":block_index+1,"total":len(a_keys)}),flush=True)
    diag=np.clip(np.diag(aggregate),0,None); denom=np.sqrt(np.outer(diag,diag)); cosine=np.divide(aggregate,denom,out=np.zeros_like(aggregate),where=denom>0); distance=np.sqrt(np.clip(diag[:,None]+diag[None,:]-2*aggregate,0,None))
    values,vectors=centered_eigensystem(aggregate); coords=vectors*np.sqrt(values)[None,:]
    np.savez_compressed(out/"model_aggregated_geometry.npz",gram=aggregate,cosine=cosine,distance=distance,eigenvalues=values,pca_coordinates=coords,ptypes=np.arange(32))
    pd.DataFrame(block_rows).sort_values(["layer","target_module"]).to_csv(out/"layer_target_module_summary.csv",index=False)
    pd.DataFrame(pair_rows).to_parquet(out/"layer_target_module_pairwise.parquet",index=False)
    long=[]
    for i in range(n):
        for j in range(n): long.append({"ptype_a":i,"profile_a":catalog.iloc[i].human_readable_profile,"ptype_b":j,"profile_b":catalog.iloc[j].human_readable_profile,"frobenius_inner_product":aggregate[i,j],"cosine_similarity":cosine[i,j],"frobenius_distance":distance[i,j]})
    pd.DataFrame(long).to_csv(out/"model_aggregated_pairwise.csv",index=False)
    nearest=[]
    for i in range(n):
        order=np.argsort(np.where(np.arange(n)==i,np.inf,distance[i]))
        for rank,j in enumerate(order[:5],1): nearest.append({"ptype":i,"profile":catalog.iloc[i].human_readable_profile,"neighbor_rank":rank,"neighbor_ptype":int(j),"neighbor_profile":catalog.iloc[j].human_readable_profile,"distance":distance[i,j],"cosine":cosine[i,j]})
    pd.DataFrame(nearest).to_csv(out/"profile_nearest_neighbors.csv",index=False)
    pd.DataFrame({"ptype":range(n),"profile":catalog.human_readable_profile,**{f"PC{k+1}":coords[:,k] for k in range(min(10,n))}}).to_csv(out/"weight_pca_coordinates.csv",index=False)
    np.savetxt(out/"hierarchical_clustering_linkage.csv",linkage(distance[np.triu_indices(n,1)],method="average"),delimiter=",",header="cluster_a,cluster_b,distance,count",comments="")
    atomic_json(out/"weight_geometry_summary.json",{"complete":True,"adapters":32,"blocks":len(a_keys),"factor_scale":2.0,"dense_updates_materialized":False,"exact_low_rank_inner_products":True,"model_aggregate_profile_kernel_effective_rank":effective_rank(values),"total_update_frobenius_norm":float(np.sqrt(diag.sum()))})
    atomic_json(args.project/"progress.json",{"project":"full_profile_drift_study_v1","stage":"weight_geometry_complete","blocks_complete":len(a_keys),"integrity_failure":None})


if __name__ == "__main__": main()
