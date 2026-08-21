"""Exposure sensitivity, low-count subsets, multivariate alignment, and compact audit tables."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score

from .io import atomic_json

SEED=20260803; TRAITS=tuple("OCEAN")


def bootstrap_corr(x,y,fn,rng,n=1000):
    values=[]
    for _ in range(n):
        idx=rng.integers(0,len(x),len(x)); values.append(fn(x[idx],y[idx]).statistic)
    return np.nanpercentile(values,[2.5,97.5])


def classical_mds(distance,k=5):
    n=len(distance); h=np.eye(n)-np.ones((n,n))/n; gram=-.5*h@(distance**2)@h; vals,vecs=np.linalg.eigh((gram+gram.T)/2); order=np.argsort(vals)[::-1]; vals=np.clip(vals[order][:k],0,None); return vecs[:,order[:k]]*np.sqrt(vals)


def procrustes_stat(x,y):
    x=x-x.mean(0); y=y-y.mean(0); x=x/(np.linalg.norm(x) or 1); y=y/(np.linalg.norm(y) or 1); u,_,vt=np.linalg.svd(x.T@y,full_matrices=False); aligned=x@u@vt; return float(np.sum((aligned-y)**2))


def loo_target_prediction(x,y):
    pred=np.zeros_like(y,dtype=float)
    for i in range(len(x)):
        train=np.arange(len(x))!=i; xx=np.column_stack([np.ones(train.sum()),x[train]]); coef=np.linalg.solve(xx.T@xx+1e-5*np.eye(xx.shape[1]),xx.T@y[train]); pred[i]=np.r_[1,x[i]]@coef
    return pred


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project",type=Path,required=True); ap.add_argument("--output-root",type=Path); args=ap.parse_args(); p=args.project; output_root=args.output_root or p; out=output_root/"analysis_outputs"; rng=np.random.default_rng(SEED)
    catalog=pd.read_csv(p/"ptype_catalog.csv").sort_values("ptype"); exposure=pd.read_csv(p/"training_exposure/all_32_training_exposure.csv").sort_values("ptype"); effects=pd.read_csv(out/"effect_magnitudes.csv")
    activation=effects[effects.representation=="activation"].groupby("ptype").effect_l2_rms_over_prompts.mean(); logits=effects[effects.representation=="logits"].groupby("ptype").effect_l2_rms_over_prompts.mean()
    metrics=pd.DataFrame({"ptype":range(32),"row_count":exposure.total_raw_rows,"retained_tokens":exposure.total_retained_tokens_after_512,"nominal_steps":exposure.nominal_optimizer_steps,"unique_tuple_count":exposure.unique_trait_tuple_proxy_count,"update_frobenius_norm":catalog.effective_update_frobenius_norm,"update_spectral_norm":catalog.effective_update_spectral_aggregate_norm,"activation_effect_magnitude":activation.values,"logit_effect_magnitude":logits.values})
    metrics.to_csv(out/"profile_exposure_effect_metrics.csv",index=False)
    corr=[]; cols=list(metrics.columns[1:])
    for i,a in enumerate(cols):
      for b in cols[i+1:]:
        x=metrics[a].to_numpy(float); y=metrics[b].to_numpy(float); xp=np.log1p(x) if a in {"row_count","retained_tokens","nominal_steps","unique_tuple_count"} else x; yp=np.log1p(y) if b in {"row_count","retained_tokens","nominal_steps","unique_tuple_count"} else y
        pc=pearsonr(xp,yp); sc=spearmanr(x,y); pci=bootstrap_corr(xp,yp,pearsonr,rng); sci=bootstrap_corr(x,y,spearmanr,rng)
        corr.append({"variable_a":a,"variable_b":b,"pearson_r":pc.statistic,"pearson_p":pc.pvalue,"pearson_ci_low":pci[0],"pearson_ci_high":pci[1],"spearman_rho":sc.statistic,"spearman_p":sc.pvalue,"spearman_ci_low":sci[0],"spearman_ci_high":sci[1],"bootstrap_resamples":1000})
    pd.DataFrame(corr).to_csv(out/"exposure_effect_correlations.csv",index=False)
    distances=pd.read_parquet(out/"all_representation_pairwise_distances.parquet"); pair_cov=[]
    for i,j in zip(*np.triu_indices(32,1)):
        pair_cov.append({"ptype_a":i,"ptype_b":j,"ocean_hamming":sum(catalog.iloc[i][f"{t}_high"]!=catalog.iloc[j][f"{t}_high"] for t in TRAITS),"row_weighted_centroid_distance":np.linalg.norm(catalog.iloc[i][[f"row_weighted_{t}_centroid" for t in TRAITS]].to_numpy(float)-catalog.iloc[j][[f"row_weighted_{t}_centroid" for t in TRAITS]].to_numpy(float)),"unique_tuple_centroid_distance":np.linalg.norm(catalog.iloc[i][[f"unique_tuple_weighted_{t}_centroid" for t in TRAITS]].to_numpy(float)-catalog.iloc[j][[f"unique_tuple_weighted_{t}_centroid" for t in TRAITS]].to_numpy(float)),"row_count_difference":abs(metrics.row_count[i]-metrics.row_count[j]),"row_count_ratio":max(metrics.row_count[i],metrics.row_count[j])/min(metrics.row_count[i],metrics.row_count[j]),"retained_token_difference":abs(metrics.retained_tokens[i]-metrics.retained_tokens[j]),"retained_token_ratio":max(metrics.retained_tokens[i],metrics.retained_tokens[j])/min(metrics.retained_tokens[i],metrics.retained_tokens[j]),"nominal_step_difference":abs(metrics.nominal_steps[i]-metrics.nominal_steps[j]),"nominal_step_ratio":max(metrics.nominal_steps[i],metrics.nominal_steps[j])/min(metrics.nominal_steps[i],metrics.nominal_steps[j])})
    pair_cov=pd.DataFrame(pair_cov); weight=distances[(distances.representation=="weights") & (distances.layer==31)][["ptype_a","ptype_b","distance"]].rename(columns={"distance":"weight_distance"}); pair_cov.merge(weight,on=["ptype_a","ptype_b"]).to_csv(out/"weight_distance_covariates.csv",index=False)
    # Fixed sparse-profile and matched-exposure sensitivity subsets.
    subset_defs={"all_32":set(range(32)),"rows_ge_1000":set(metrics.index[metrics.row_count>=1000]),"rows_ge_10000":set(metrics.index[metrics.row_count>=10000]),"rows_ge_50000":set(metrics.index[metrics.row_count>=50000]),"unique_tuples_ge_10":set(metrics.index[metrics.unique_tuple_count>=10]),"unique_tuples_ge_25":set(metrics.index[metrics.unique_tuple_count>=25])}
    quart=pd.qcut(metrics.row_count,4,labels=False,duplicates="drop"); subset_defs.update({f"exposure_quartile_{q+1}":set(metrics.index[quart==q]) for q in sorted(quart.unique())})
    target_rw=catalog[[f"row_weighted_{t}_centroid" for t in TRAITS]].to_numpy(float); target_uw=catalog[[f"unique_tuple_weighted_{t}_centroid" for t in TRAITS]].to_numpy(float); sensitivity=[]
    group_cols=["representation","prompt_group","pooling_rule","layer"]
    for keys,frame in distances.groupby(group_cols,dropna=False):
      for subset,members in subset_defs.items():
        use=frame[frame.ptype_a.isin(members)&frame.ptype_b.isin(members)];
        if len(use)<3: continue
        for target_name,target in (("row_weighted_continuous",target_rw),("unique_tuple_weighted_continuous",target_uw)):
            tv=np.array([np.linalg.norm(target[int(a)]-target[int(b)]) for a,b in zip(use.ptype_a,use.ptype_b)]); sensitivity.append(dict(zip(group_cols,keys))|{"subset":subset,"profiles":len(members),"pairs":len(use),"target":target_name,"pearson_r":pearsonr(use.distance,tv).statistic,"spearman_rho":spearmanr(use.distance,tv).statistic})
      for ratio in (2.0,1.5):
        merged=frame.merge(pair_cov[["ptype_a","ptype_b","row_count_ratio"]],on=["ptype_a","ptype_b"]); use=merged[merged.row_count_ratio<=ratio]
        if len(use)<3: continue
        for target_name,target in (("row_weighted_continuous",target_rw),("unique_tuple_weighted_continuous",target_uw)):
            tv=np.array([np.linalg.norm(target[int(a)]-target[int(b)]) for a,b in zip(use.ptype_a,use.ptype_b)]); sensitivity.append(dict(zip(group_cols,keys))|{"subset":f"pair_row_ratio_le_{str(ratio).replace('.','_')}","profiles":len(set(use.ptype_a)|set(use.ptype_b)),"pairs":len(use),"target":target_name,"pearson_r":pearsonr(use.distance,tv).statistic,"spearman_rho":spearmanr(use.distance,tv).statistic})
    pd.DataFrame(sensitivity).to_csv(out/"sparse_profile_sensitivity.csv",index=False)
    # Multivariate alignment diagnostics: MDS, Procrustes, CCA, and leave-profile-out target prediction.
    mult=[]; iu=np.triu_indices(32,1)
    for keys,frame in distances.groupby(group_cols,dropna=False):
        matrix=np.zeros((32,32)); matrix[frame.ptype_a.astype(int),frame.ptype_b.astype(int)]=frame.distance; matrix+=matrix.T; coords=classical_mds(matrix,5)
        for target_name,target in (("row_weighted_continuous",target_rw),("unique_tuple_weighted_continuous",target_uw)):
            standardized=(target-target.mean(0))/target.std(0); cca=CCA(n_components=5,max_iter=1000).fit(coords,standardized); xc,yc=cca.transform(coords,standardized); pred=loo_target_prediction(coords,standardized)
            row=dict(zip(group_cols,keys))|{"target":target_name,"procrustes_disparity":procrustes_stat(coords,standardized),"mean_canonical_correlation":np.mean([pearsonr(xc[:,i],yc[:,i]).statistic for i in range(5)]),"leave_profile_out_multivariate_r2":r2_score(standardized,pred,multioutput="variance_weighted")}
            for i,t in enumerate(TRAITS): row[f"leave_profile_out_{t}_r2"]=r2_score(standardized[:,i],pred[:,i])
            mult.append(row)
    pd.DataFrame(mult).to_csv(out/"multivariate_target_alignment.csv",index=False)
    # Common drift projection, direction-independent norm, and profile layer fingerprints.
    common=[]; fingerprint=[]
    for file in sorted(out.glob("activation_centroids_*.npz")):
        name=file.stem.removeprefix("activation_centroids_"); group,pool=next((g,name[len(g)+1:]) for g in ("ipip_stems","neutral_controls","naturalistic_behavioral") if name.startswith(g+"_")); y=np.load(file)["centroids"]
        for layer in range(32):
            mean=y[:,layer].mean(0); denom=np.dot(mean,mean)
            for ptype in range(32):
                projection=np.dot(y[ptype,layer],mean)/math.sqrt(denom) if denom else 0; residual=y[ptype,layer]-(np.dot(y[ptype,layer],mean)/denom)*mean if denom else y[ptype,layer]
                common.append({"prompt_group":group,"pooling_rule":pool,"layer":layer,"ptype":ptype,"profile":catalog.iloc[ptype].human_readable_profile,"common_drift_projection":projection,"residual_direction_norm":np.linalg.norm(residual)})
                fingerprint.append({"prompt_group":group,"pooling_rule":pool,"layer":layer,"ptype":ptype,"profile":catalog.iloc[ptype].human_readable_profile,"centroid_norm":np.linalg.norm(y[ptype,layer])})
    pd.DataFrame(common).to_csv(out/"common_drift_projections.csv",index=False); pd.DataFrame(fingerprint).to_csv(out/"profile_drift_fingerprints.csv",index=False)
    atomic_json(out/"supplemental_analysis_summary.json",{"complete":True,"bootstrap_resamples":1000,"fixed_sensitivity_subsets":list(subset_defs)+["pair_row_ratio_le_2_0","pair_row_ratio_le_1_5"],"continuous_targets_are_primary":True,"binary_bits_used_only_for_hamming_geometry":True})
    atomic_json(output_root/"progress.json",{"project":"full_profile_drift_study_v1","stage":"supplemental_analysis_complete","integrity_failure":None})

if __name__=="__main__": main()
