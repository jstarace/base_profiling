"""Extract verified per-adapter layer/module norms alongside exact geometry."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project",type=Path,required=True); args=ap.parse_args(); data=json.loads((args.project/"audit/frozen_inputs/adapter_verification.json").read_text()); catalog=pd.read_csv(args.project/"ptype_catalog.csv").set_index("adapter"); rows=[]
    for adapter in data["adapters"]:
      for item in adapter["structural"]["effective_updates"]:
        rows.append({"ptype":adapter["ptype"],"adapter":adapter["adapter"],"human_readable_profile":catalog.loc[adapter["adapter"],"human_readable_profile"],"layer":item["layer"],"target_module":item["target_module"],"matrix_rows":item["matrix_shape"][0],"matrix_columns":item["matrix_shape"][1],"rank":item["rank"],"lora_alpha":item["lora_alpha"],"scale":item["scale_lora_alpha_over_rank"],"a_norm":item["a_norm"],"b_norm":item["b_norm"],"effective_update_frobenius_norm":item["frobenius_norm"],"effective_update_spectral_norm":item["spectral_norm"],"spectral_to_frobenius_ratio":item["spectral_norm"]/item["frobenius_norm"]})
    frame=pd.DataFrame(rows); frame.to_csv(args.project/"weight_geometry/layer_target_module_adapter_stats.csv",index=False); frame.groupby(["layer","target_module"],as_index=False).agg(mean_frobenius=("effective_update_frobenius_norm","mean"),median_frobenius=("effective_update_frobenius_norm","median"),mean_spectral_to_frobenius=("spectral_to_frobenius_ratio","mean"),median_spectral_to_frobenius=("spectral_to_frobenius_ratio","median")).to_csv(args.project/"weight_geometry/layer_target_module_norm_summary.csv",index=False)
    modules=("down_proj","gate_proj","k_proj","o_proj","q_proj","up_proj","v_proj")
    for layer in range(32):
        gram=sum(np.load(args.project/"weight_geometry"/f"layer_{layer:02d}_{module}_geometry.npz")["gram"] for module in modules); diag=np.clip(np.diag(gram),0,None); denom=np.sqrt(np.outer(diag,diag)); cosine=np.divide(gram,denom,out=np.zeros_like(gram),where=denom>0); distance=np.sqrt(np.clip(diag[:,None]+diag[None,:]-2*gram,0,None)); np.savez_compressed(args.project/"weight_geometry"/f"layer_{layer:02d}_aggregated_geometry.npz",gram=gram,cosine=cosine,distance=distance,ptypes=np.arange(32))
if __name__=="__main__":main()
