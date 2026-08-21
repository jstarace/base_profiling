"""Integrated all-32 factorial, uniqueness, alignment, exposure, and RSA analyses."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .capture import paths
from .io import atomic_json

SEED=20260803
TRAITS=tuple("OCEAN")
WEIGHTS=(16,8,4,2,1)
GROUPS=("ipip_stems","neutral_controls","naturalistic_behavioral")
POOLS=("final_token","mean_tokens")
SUMMARY_LAYERS=(0,8,16,24,31)
CAPTURE_ROOTS=[]


def load_named(path: Path,name: str) -> np.ndarray:
    with np.load(path) as f: return f[name]


def capture_path(kind,model_key,group,pool=None):
    candidates=[paths(root,kind,model_key,group,pool)[0] for root in CAPTURE_ROOTS]
    existing=[path for path in candidates if path.exists()]
    if not existing: raise FileNotFoundError(candidates)
    return existing[-1]


def terms(max_order=5):
    result=[]
    for order in range(max_order+1):
        for subset in itertools.combinations(range(5),order): result.append(subset)
    return result


TERMS=terms(); TERM_NAMES={s:("intercept" if not s else "×".join(TRAITS[i] for i in s)) for s in TERMS}


def factorial_design(max_order=5):
    bits=np.array([[1 if p&w else -1 for w in WEIGHTS] for p in range(32)],dtype=np.float64)
    use=[s for s in TERMS if len(s)<=max_order]
    x=np.column_stack([np.ones(32) if not s else np.prod(bits[:,s],axis=1) for s in use])
    return x,use


def pc_removed(y):
    centered=y-y.mean(0); gram=centered@centered.T; vals,vecs=np.linalg.eigh(gram); score=vecs[:,-1]*math.sqrt(max(vals[-1],0))
    if np.dot(score,score)==0:return centered
    direction=score@centered/np.dot(score,score)
    return centered-score[:,None]*direction[None,:]


def transformations(y,exposure_x):
    centered=y-y.mean(0); norms=np.linalg.norm(centered,axis=1,keepdims=True)
    residual=y-exposure_x@np.linalg.pinv(exposure_x)@y
    return {"raw":y,"profile_centered":centered,"unit_norm":np.divide(centered,norms,out=np.zeros_like(centered),where=norms>0),"exposure_residualized":residual,"common_pc1_removed":pc_removed(y)}


def factorial_rows(y,representation,group,pool,layer,exposure_x):
    x,_=factorial_design(5); rows=[]
    for transform,data in transformations(y,exposure_x).items():
        beta=x.T@data/32.0; energy=np.einsum("ij,ij->i",beta,beta); total=energy.sum()
        for subset,value in zip(TERMS,energy): rows.append({"representation":representation,"prompt_group":group,"pooling_rule":pool,"layer":layer,"transformation":transform,"term":TERM_NAMES[subset],"interaction_order":len(subset),"energy":value,"energy_fraction":value/total if total else np.nan})
    return rows


def trait_flip_rows(y,representation,group,pool,layer,rng):
    rows=[]
    for trait,w in zip(TRAITS,WEIGHTS):
        dirs=np.stack([y[p+w]-y[p] for p in range(32) if not p&w]); norm=np.linalg.norm(dirs,axis=1); cosine=(dirs@dirs.T)/np.outer(norm,norm); off=cosine[np.triu_indices(16,1)]
        first=dirs[:8].mean(0); second=dirs[8:].mean(0); split=float(first@second/(np.linalg.norm(first)*np.linalg.norm(second)))
        random_values=[]
        for _ in range(100):
            perm=rng.permutation(32); rd=np.stack([y[perm[2*i+1]]-y[perm[2*i]] for i in range(16)]); rn=np.linalg.norm(rd,axis=1); rc=(rd@rd.T)/np.outer(rn,rn); random_values.append(np.nanmean(rc[np.triu_indices(16,1)]))
        rows.append({"representation":representation,"prompt_group":group,"pooling_rule":pool,"layer":layer,"trait":trait,"matched_pairs":16,"mean_pairwise_cosine":np.nanmean(off),"median_pairwise_cosine":np.nanmedian(off),"split_half_cosine":split,"random_pair_mean_cosine":np.mean(random_values),"random_pair_sd":np.std(random_values,ddof=0),"agreement_above_random":np.nanmean(off)-np.mean(random_values)})
    return rows


def cv_reconstruction(y,exposure_covars):
    rng=np.random.default_rng(SEED); order=rng.permutation(32); folds=np.array_split(order,8); rows=[]
    for order_max,label in ((1,"main_effects"),(2,"main_plus_two_way"),(3,"through_third_order")):
        x,_=factorial_design(order_max)
        for add_exp in (False,True):
            xx=np.column_stack([x,exposure_covars]) if add_exp else x; pred=np.zeros_like(y)
            for test in folds:
                train=np.setdiff1d(np.arange(32),test); scale=1e-6*np.trace(xx[train].T@xx[train])/xx.shape[1]
                coef=np.linalg.solve(xx[train].T@xx[train]+scale*np.eye(xx.shape[1]),xx[train].T@y[train]); pred[test]=xx[test]@coef
            sse=np.sum((y-pred)**2); sst=np.sum((y-y.mean(0))**2)
            rows.append({"factorial_model":label,"exposure_covariates_added":add_exp,"profile_disjoint_folds":8,"reconstruction_rmse":math.sqrt(np.mean((y-pred)**2)),"explained_variance":1-sse/sst if sst else np.nan})
    return rows


def distance_rows(y,representation,group,pool,layer,catalog):
    d=squareform(pdist(y)); rows=[]
    for i,j in zip(*np.triu_indices(32,1)):
        rows.append({"representation":representation,"prompt_group":group,"pooling_rule":pool,"layer":layer,"ptype_a":i,"profile_a":catalog.iloc[i].human_readable_profile,"ptype_b":j,"profile_b":catalog.iloc[j].human_readable_profile,"distance":d[i,j]})
    return d,rows


def target_matrices(catalog):
    rw=catalog[[f"row_weighted_{t}_centroid" for t in TRAITS]].to_numpy(); uw=catalog[[f"unique_tuple_weighted_{t}_centroid" for t in TRAITS]].to_numpy(); bits=catalog[[f"{t}_high" for t in TRAITS]].to_numpy()
    return {"row_weighted_continuous":squareform(pdist(rw)),"unique_tuple_weighted_continuous":squareform(pdist(uw)),"binary_hamming":squareform(pdist(bits,metric="hamming"))}


def alignment_rows(distance,targets,metadata,rng,permutations=1000):
    iu=np.triu_indices(32,1); observed=distance[iu]; rows=[]
    for name,target in targets.items():
        tv=target[iu]; pr=pearsonr(observed,tv).statistic; sr=spearmanr(observed,tv).statistic; null=[]
        for _ in range(permutations):
            perm=rng.permutation(32); null.append(pearsonr(observed,target[np.ix_(perm,perm)][iu]).statistic)
        p=(1+sum(abs(x)>=abs(pr) for x in null))/(permutations+1)
        rows.append(metadata|{"target_geometry":name,"pearson_r":pr,"spearman_rho":sr,"mantel_permutations":permutations,"mantel_two_sided_p":p,"null_mean":np.mean(null),"null_sd":np.std(null,ddof=0)})
    return rows


def bh(frame,pcol="mantel_two_sided_p"):
    if frame.empty:return frame
    p=frame[pcol].to_numpy(); order=np.argsort(p); q=np.empty(len(p)); q[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]; frame[pcol.replace("p","q")]=np.clip(q,0,1); return frame


def uniqueness(project,group,pool,records,catalog,rng):
    # A frozen coordinate subset avoids favorable dimension selection and makes all-layer classification tractable.
    coordinates=np.sort(rng.choice(4096,128,replace=False)); samples=[]
    for p in range(32): samples.append(load_named(capture_path("activation_outputs",f"ptype_{p}",group,pool),"delta_h")[:,:,coordinates])
    data=np.stack(samples); split=np.array([r["split"] for r in records]); train=np.where(split=="train")[0]; val=np.where(split=="validation")[0]; test=np.where(split=="test")[0]
    rows=[]; confusions={}
    for layer in range(32):
        train_cent=data[:,train,layer].mean(1); test_x=data[:,test,layer].reshape(-1,128); true=np.repeat(np.arange(32),len(test)); dist=((test_x[:,None,:]-train_cent[None,:,:])**2).sum(2); pred=dist.argmin(1); margin=np.partition(dist,1,axis=1)[:,1]-np.min(dist,axis=1)
        rows.append({"classifier":"nearest_centroid","prompt_group":group,"pooling_rule":pool,"layer":layer,"balanced_accuracy":balanced_accuracy_score(true,pred),"macro_f1":f1_score(true,pred,average="macro"),"mean_classification_margin":margin.mean(),"test_samples":len(true)})
        confusions[("nearest_centroid",layer)]=confusion_matrix(true,pred,labels=range(32))
        if layer in SUMMARY_LAYERS:
            scaler=StandardScaler().fit(data[:,train,layer].reshape(-1,128)); tx=scaler.transform(data[:,train,layer].reshape(-1,128)); ty=np.repeat(np.arange(32),len(train)); vx=scaler.transform(data[:,val,layer].reshape(-1,128)); vy=np.repeat(np.arange(32),len(val)); test_scaled=scaler.transform(test_x)
            best=None
            for c in (.01,.1,1.0):
                model=LogisticRegression(C=c,max_iter=300,solver="lbfgs").fit(tx,ty); score=balanced_accuracy_score(vy,model.predict(vx)); best=(score,c,model) if best is None or score>best[0] else best
            model=LogisticRegression(C=best[1],max_iter=500,solver="lbfgs").fit(np.vstack([tx,vx]),np.r_[ty,vy]); lp=model.predict(test_scaled)
            rows.append({"classifier":"regularized_logistic","prompt_group":group,"pooling_rule":pool,"layer":layer,"balanced_accuracy":balanced_accuracy_score(true,lp),"macro_f1":f1_score(true,lp,average="macro"),"mean_classification_margin":np.nan,"test_samples":len(true),"selected_C":best[1]}); confusions[("regularized_logistic",layer)]=confusion_matrix(true,lp,labels=range(32))
            try:
                lda=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto").fit(np.vstack([tx,vx]),np.r_[ty,vy]); pp=lda.predict(test_scaled); rows.append({"classifier":"shrinkage_lda","prompt_group":group,"pooling_rule":pool,"layer":layer,"balanced_accuracy":balanced_accuracy_score(true,pp),"macro_f1":f1_score(true,pp,average="macro"),"mean_classification_margin":np.nan,"test_samples":len(true)})
            except Exception as exc: rows.append({"classifier":"shrinkage_lda","prompt_group":group,"pooling_rule":pool,"layer":layer,"failure":str(exc),"balanced_accuracy":np.nan,"macro_f1":np.nan,"test_samples":len(true)})
            # Adapter-label permutation baseline using the already fixed nearest-centroid test geometry.
            null=[]
            for _ in range(100): null.append(balanced_accuracy_score(true,rng.permutation(32)[pred]))
            rows.append({"classifier":"nearest_centroid_label_permutation","prompt_group":group,"pooling_rule":pool,"layer":layer,"balanced_accuracy":np.mean(null),"permutation_sd":np.std(null,ddof=0),"permutations":100,"macro_f1":np.nan,"test_samples":len(true)})
    # Held-out profile signature dispersion at the final layer.
    final=data[:,:,31]; cent=final.mean(1); within=np.mean(np.linalg.norm(final-cent[:,None,:],axis=2)); between=np.mean(pdist(cent)); sil=silhouette_score(final.reshape(-1,128),np.repeat(range(32),final.shape[1]),sample_size=min(10000,32*final.shape[1]),random_state=SEED)
    summary={"prompt_group":group,"pooling_rule":pool,"fixed_coordinate_count":128,"fixed_coordinate_indices":coordinates.tolist(),"within_adapter_prompt_dispersion":within,"between_adapter_centroid_distance":between,"uniqueness_score_between_over_within":between/within,"silhouette_score":sil}
    return rows,confusions,summary


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--project",type=Path,required=True); ap.add_argument("--capture-root",type=Path,action="append",default=[]); ap.add_argument("--output-root",type=Path); args=ap.parse_args(); project=args.project; global CAPTURE_ROOTS; CAPTURE_ROOTS=[project]+args.capture_root; output_root=args.output_root or project; out=output_root/"analysis_outputs"; out.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(SEED)
    catalog=pd.read_csv(project/"ptype_catalog.csv").sort_values("ptype").reset_index(drop=True); exposure=pd.read_csv(project/"training_exposure/all_32_training_exposure.csv").sort_values("ptype")
    exp=np.column_stack([np.ones(32),np.log1p(exposure.total_raw_rows),np.log1p(exposure.total_retained_tokens_after_512),np.log1p(exposure.nominal_optimizer_steps),np.log1p(exposure.unique_trait_tuple_proxy_count)]); exp[:,1:]=(exp[:,1:]-exp[:,1:].mean(0))/exp[:,1:].std(0)
    targets=target_matrices(catalog); manifest=json.loads((project/"prompt_manifest/prompt_manifest.json").read_text()); all_records=manifest["records"]
    factorial=[]; flips=[]; distances=[]; alignments=[]; predictive=[]; effects=[]; uniqueness_rows=[]; uniqueness_summaries=[]; centroid_files={}; representation_distances={}
    for group in GROUPS:
      records=[r for r in all_records if r["group"]==group]
      for pool in POOLS:
        centroid=np.empty((32,32,4096),dtype=np.float32)
        for p in range(32):
            arr=load_named(capture_path("activation_outputs",f"ptype_{p}",group,pool),"delta_h"); centroid[p]=arr.mean(0); layer_norm=np.sqrt(np.einsum("plh,plh->l",arr,arr)/(arr.shape[0]))
            for layer,value in enumerate(layer_norm): effects.append({"representation":"activation","prompt_group":group,"pooling_rule":pool,"ptype":p,"profile":catalog.iloc[p].human_readable_profile,"layer":layer,"effect_l2_rms_over_prompts":value})
        centroid_path=out/f"activation_centroids_{group}_{pool}.npz"; np.savez_compressed(centroid_path,centroids=centroid); centroid_files[f"{group}/{pool}"]=str(centroid_path)
        ur,conf,us=uniqueness(project,group,pool,records,catalog,np.random.default_rng(SEED+len(uniqueness_rows))); uniqueness_rows.extend(ur); uniqueness_summaries.append(us)
        for (classifier,layer),matrix in conf.items(): np.savetxt(out/f"confusion_{group}_{pool}_{classifier}_layer{layer:02d}.csv",matrix,delimiter=",",fmt="%d")
        for layer in range(32):
            y=centroid[:,layer].astype(np.float64); factorial.extend(factorial_rows(y,"activation",group,pool,layer,exp)); flips.extend(trait_flip_rows(y,"activation",group,pool,layer,rng)); d,dr=distance_rows(y,"activation",group,pool,layer,catalog); distances.extend(dr); representation_distances[("activation",group,pool,layer)]=d
            alignments.extend(alignment_rows(d,targets,{"representation":"activation","prompt_group":group,"pooling_rule":pool,"layer":layer},rng));
            for row in cv_reconstruction(y,exp[:,1:]): predictive.append({"representation":"activation","prompt_group":group,"pooling_rule":pool,"layer":layer}|row)
        atomic_json(output_root/"progress.json",{"project":"full_profile_drift_study_v1","stage":"analysis_activation","prompt_group":group,"pooling_rule":pool,"integrity_failure":None})
    # Logit centroids and geometry.
    for group in GROUPS:
        centroid=[]
        for p in range(32):
            arr=load_named(capture_path("logit_outputs",f"ptype_{p}",group),"delta_logits").astype("float32"); centroid.append(arr.mean(0)); effects.append({"representation":"logits","prompt_group":group,"pooling_rule":"final_token","ptype":p,"profile":catalog.iloc[p].human_readable_profile,"layer":32,"effect_l2_rms_over_prompts":float(np.sqrt(np.einsum("ij,ij->",arr,arr)/arr.shape[0]))})
        centroid=np.stack(centroid).astype("float64"); np.savez_compressed(out/f"logit_centroids_{group}.npz",centroids=centroid.astype("float32")); factorial.extend(factorial_rows(centroid,"logits",group,"final_token",32,exp)); flips.extend(trait_flip_rows(centroid,"logits",group,"final_token",32,rng)); d,dr=distance_rows(centroid,"logits",group,"final_token",32,catalog); distances.extend(dr); representation_distances[("logits",group,"final_token",32)]=d; alignments.extend(alignment_rows(d,targets,{"representation":"logits","prompt_group":group,"pooling_rule":"final_token","layer":32},rng));
        for row in cv_reconstruction(centroid,exp[:,1:]): predictive.append({"representation":"logits","prompt_group":group,"pooling_rule":"final_token","layer":32}|row)
    # Weight-space layer aggregates from the exact seven module kernels.
    for layer in range(32):
        gram=sum(np.load(project/"weight_geometry"/f"layer_{layer:02d}_{module}_geometry.npz")["gram"] for module in ("down_proj","gate_proj","k_proj","o_proj","q_proj","up_proj","v_proj")); vals,vecs=np.linalg.eigh((gram+gram.T)/2); keep=vals>max(vals.max()*1e-12,1e-12); y=vecs[:,keep]*np.sqrt(vals[keep]); factorial.extend(factorial_rows(y,"weights","all","none",layer,exp)); flips.extend(trait_flip_rows(y,"weights","all","none",layer,rng)); d,dr=distance_rows(y,"weights","all","none",layer,catalog); distances.extend(dr); representation_distances[("weights","all","none",layer)]=d; alignments.extend(alignment_rows(d,targets,{"representation":"weights","prompt_group":"all","pooling_rule":"none","layer":layer},rng));
    pd.DataFrame(factorial).to_parquet(out/"walsh_hadamard_energy.parquet",index=False); pd.DataFrame(factorial).groupby(["representation","prompt_group","pooling_rule","layer","transformation","interaction_order"],as_index=False).energy.sum().to_csv(out/"walsh_hadamard_energy_by_order.csv",index=False)
    pd.DataFrame(flips).to_csv(out/"matched_trait_flip_stability.csv",index=False); pd.DataFrame(distances).to_parquet(out/"all_representation_pairwise_distances.parquet",index=False); bh(pd.DataFrame(alignments)).to_csv(out/"continuous_target_alignment.csv",index=False); pd.DataFrame(predictive).to_csv(out/"factorial_predictive_models.csv",index=False); pd.DataFrame(effects).to_csv(out/"effect_magnitudes.csv",index=False); pd.DataFrame(uniqueness_rows).to_csv(out/"adapter_uniqueness_classification.csv",index=False); pd.DataFrame(uniqueness_summaries).to_csv(out/"adapter_uniqueness_summary.csv",index=False)
    # Cross-representation RSA on fixed aggregate summaries.
    keys=list(representation_distances); rsa=[]
    for i,a in enumerate(keys):
        for b in keys[i:]:
            va=representation_distances[a][np.triu_indices(32,1)]; vb=representation_distances[b][np.triu_indices(32,1)]; rsa.append({"representation_a":"/".join(map(str,a)),"representation_b":"/".join(map(str,b)),"pearson_distance_r":pearsonr(va,vb).statistic,"spearman_distance_rho":spearmanr(va,vb).statistic})
    pd.DataFrame(rsa).to_csv(out/"cross_representation_similarity.csv",index=False)
    atomic_json(out/"analysis_summary.json",{"complete":True,"profiles":32,"activation_conditions":len(GROUPS)*len(POOLS)*32,"logit_conditions":len(GROUPS),"weight_layers":32,"fixed_seed":SEED,"confirmatory_primary":"raw captured deltas and preregistered all-profile factorial analyses","exploratory_transformations":["profile_centered","unit_norm","exposure_residualized","common_pc1_removed"],"centroid_files":centroid_files})
    atomic_json(output_root/"progress.json",{"project":"full_profile_drift_study_v1","stage":"integrated_analysis_complete","integrity_failure":None})


if __name__=="__main__": main()
