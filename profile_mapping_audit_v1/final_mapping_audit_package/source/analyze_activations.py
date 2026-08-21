"""Preregistered all-layer activation analyses and terminal A/B/C decision."""
from __future__ import annotations

import argparse,itertools,json,math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy.stats import pearsonr,spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,f1_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

from activation_study.io import atomic_json

ADAPTERS=("ptype_0","ptype_31","ptype_9","ptype_23"); GROUPS=("ipip_stems","neutral_controls","naturalistic_behavioral"); POOLS=("final_token","mean_tokens")
UPDATE_MAG={"ptype_0":85.08547792018027,"ptype_31":65.57962989112887,"ptype_9":131.5950875940946,"ptype_23":3.599366320296153}
RNG_SEED=20260801; BOOT=1000; LABEL_PERMS=100


def load_delta(root,adapter,group,pool):
    with np.load(root/"shards"/adapter/group/f"{pool}.npz") as z: return z["delta"].astype("float32")


def bootstrap_ci(values,rng,stat=np.mean):
    values=np.asarray(values); estimates=np.array([stat(values[rng.integers(0,len(values),len(values))]) for _ in range(BOOT)])
    return float(np.quantile(estimates,.025)),float(np.quantile(estimates,.975))


def mean_pairwise_cosine(x):
    norms=np.linalg.norm(x,axis=1); valid=norms>0; z=x[valid]/norms[valid,None]; n=len(z)
    return float((np.dot(z.sum(0),z.sum(0))-n)/(n*(n-1))) if n>1 else float("nan")


def split_half(x,rng):
    scores=[]; n=len(x)
    for _ in range(BOOT):
        order=rng.permutation(n); a=x[order[:n//2]].mean(0); b=x[order[n//2:]].mean(0); den=np.linalg.norm(a)*np.linalg.norm(b)
        scores.append(float(np.dot(a,b)/den) if den else 0.0)
    return float(np.mean(scores)),float(np.quantile(scores,.025)),float(np.quantile(scores,.975))


def bh(pvalues):
    p=np.asarray(pvalues,float); order=np.argsort(p); ranked=p[order]; q=ranked*len(p)/np.arange(1,len(p)+1); q=np.minimum.accumulate(q[::-1])[::-1]; result=np.empty_like(q); result[order]=np.minimum(q,1); return result


def prompt_folds(n,seed):
    rng=np.random.default_rng(seed); order=rng.permutation(n); return [np.asarray(x) for x in np.array_split(order,5)]


def prepare_fold(X,train_idx,test_idx):
    scaler=StandardScaler(); train=scaler.fit_transform(X[train_idx]); test=scaler.transform(X[test_idx]); components=min(16,train.shape[0]-1,train.shape[1]); pca=PCA(n_components=components,random_state=RNG_SEED)
    return pca.fit_transform(train),pca.transform(test)


def separability(X4,rng):
    # X4 shape adapter,prompt,feature; every fold is prompt-disjoint.
    a,n,h=X4.shape; X=X4.reshape(a*n,h); y=np.repeat(np.arange(a),n); prompt=np.tile(np.arange(n),a); folds=prompt_folds(n,RNG_SEED)
    predictions={"nearest_centroid":np.full(len(y),-1),"logistic_regression":np.full(len(y),-1)}; transformed=[]; chosen_cs=[]
    for fold_index,test_prompts in enumerate(folds):
        test=np.isin(prompt,test_prompts); train=~test; Xt,Xv=prepare_fold(X,np.where(train)[0],np.where(test)[0]); yt=y[train]
        predictions["nearest_centroid"][test]=NearestCentroid().fit(Xt,yt).predict(Xv)
        # Inner prompt-disjoint C selection; preprocessing is refit inside every inner training fold.
        train_prompts=np.setdiff1d(np.arange(n),test_prompts); inner=np.array_split(train_prompts,3); c_scores={c:[] for c in (.1,1.0,10.0)}
        for valid_prompts in inner:
            iv=np.isin(prompt,valid_prompts)&train; it=train&~np.isin(prompt,valid_prompts); iXt,iXv=prepare_fold(X,np.where(it)[0],np.where(iv)[0])
            for c in c_scores: c_scores[c].append(balanced_accuracy_score(y[iv],LogisticRegression(C=c,max_iter=500,random_state=RNG_SEED).fit(iXt,y[it]).predict(iXv)))
        chosen=max(c_scores,key=lambda c:(np.mean(c_scores[c]),-c)); chosen_cs.append(chosen)
        predictions["logistic_regression"][test]=LogisticRegression(C=chosen,max_iter=500,random_state=RNG_SEED).fit(Xt,yt).predict(Xv)
        transformed.append((train,test,Xt,Xv,yt,chosen))
    results=[]
    for classifier,pred in predictions.items():
        observed_ba=balanced_accuracy_score(y,pred); observed_f1=f1_score(y,pred,average="macro"); boot_ba=[]; boot_f1=[]
        for _ in range(BOOT):
            sampled=rng.integers(0,n,n); idx=np.concatenate([np.where(prompt==p)[0] for p in sampled]); boot_ba.append(balanced_accuracy_score(y[idx],pred[idx])); boot_f1.append(f1_score(y[idx],pred[idx],average="macro"))
        null_ba=[]; null_f1=[]
        for permutation in range(LABEL_PERMS):
            yp=y.copy()
            for p in range(n):
                idx=np.where(prompt==p)[0]; yp[idx]=rng.permutation(yp[idx])
            pp=np.full(len(y),-1)
            for train,test,Xt,Xv,yt,chosen in transformed:
                model=NearestCentroid() if classifier=="nearest_centroid" else LogisticRegression(C=chosen,max_iter=500,random_state=RNG_SEED+permutation)
                pp[test]=model.fit(Xt,yp[train]).predict(Xv)
            null_ba.append(balanced_accuracy_score(yp,pp)); null_f1.append(f1_score(yp,pp,average="macro"))
        results.append({"classifier":classifier,"balanced_accuracy":observed_ba,"balanced_accuracy_ci_low":np.quantile(boot_ba,.025),"balanced_accuracy_ci_high":np.quantile(boot_ba,.975),
          "macro_f1":observed_f1,"macro_f1_ci_low":np.quantile(boot_f1,.025),"macro_f1_ci_high":np.quantile(boot_f1,.975),
          "balanced_accuracy_permutation_p":(1+sum(x>=observed_ba for x in null_ba))/(LABEL_PERMS+1),"macro_f1_permutation_p":(1+sum(x>=observed_f1 for x in null_f1))/(LABEL_PERMS+1),
          "null_balanced_accuracy_mean":np.mean(null_ba),"null_macro_f1_mean":np.mean(null_f1),"chosen_C_median":np.median(chosen_cs)})
    return results


def target_vectors(targets):
    rows=targets.set_index("ptype"); binary=[]; row=[]; unique=[]
    for adapter in ADAPTERS:
        p=int(adapter[6:]); binary.append([rows.loc[p,f"{t}_high"] for t in "OCEAN"]); row.append([rows.loc[p,f"row_weighted_{t}_mean"] for t in "OCEAN"]); unique.append([rows.loc[p,f"unique_trait_tuple_weighted_{t}_mean"] for t in "OCEAN"])
    return {"binary":np.asarray(binary,float),"row_weighted":np.asarray(row,float),"unique_tuple_weighted":np.asarray(unique,float)}


def target_alignment(X4,target_sets,rng):
    # Mean of prompt-level adapter pair distances; prompt bootstrap is cheap and grouped.
    prompt_dist=np.stack([pdist(X4[:,p,:],metric="euclidean") for p in range(X4.shape[1])]); activation=prompt_dist.mean(0); rows=[]
    for name,target in target_sets.items():
        td=pdist(target,metric="euclidean"); pear=pearsonr(activation,td).statistic; spear=spearmanr(activation,td).statistic; bpear=[]; bspear=[]
        for _ in range(BOOT):
            act=prompt_dist[rng.integers(0,len(prompt_dist),len(prompt_dist))].mean(0); bpear.append(pearsonr(act,td).statistic); bspear.append(spearmanr(act,td).statistic)
        nullp=[]; nulls=[]
        for perm in itertools.permutations(range(4)):
            nd=pdist(target[list(perm)],metric="euclidean"); nullp.append(pearsonr(activation,nd).statistic); nulls.append(spearmanr(activation,nd).statistic)
        rows.append({"target_definition":name,"pearson_r":pear,"pearson_ci_low":np.quantile(bpear,.025),"pearson_ci_high":np.quantile(bpear,.975),
          "pearson_permutation_p":(1+sum(x>=pear for x in nullp))/(1+len(nullp)),"spearman_r":spear,"spearman_ci_low":np.quantile(bspear,.025),"spearman_ci_high":np.quantile(bspear,.975),
          "spearman_permutation_p":(1+sum(x>=spear for x in nulls))/(1+len(nulls))})
    return rows,prompt_dist


def centroid_geometry(X4):
    centroids=X4.mean(1); centered=centroids-centroids.mean(0); u,s,vt=np.linalg.svd(centered,full_matrices=False); variance=s*s; ratios=variance/variance.sum() if variance.sum() else variance
    nonzero=ratios[ratios>0]; effective=float(np.exp(-(nonzero*np.log(nonzero)).sum())) if len(nonzero) else 0; participation=float(1/(ratios*ratios).sum()) if ratios.sum() else 0
    coords=centered@vt[:2].T; return centroids,coords,ratios,effective,participation,squareform(pdist(centroids))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--activation-root",type=Path,required=True); p.add_argument("--targets",type=Path,required=True); p.add_argument("--output",type=Path,required=True); args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(RNG_SEED); targets=pd.read_csv(args.targets); target_sets=target_vectors(targets); effect=[]; update_rel=[]; endpoint=[]; separability_rows=[]; target_rows=[]; pca_rows=[]; centroid_rows=[]; distance_rows=[]; endpoint_centroids={}
    for group in GROUPS:
      for pool in POOLS:
        data={adapter:load_delta(args.activation_root,adapter,group,pool) for adapter in ADAPTERS}
        for layer in range(32):
          X4=np.stack([data[a][:,layer,:] for a in ADAPTERS])
          mean_norms=[]
          for ai,adapter in enumerate(ADAPTERS):
            x=X4[ai]; norms=np.linalg.norm(x,axis=1); lo,hi=bootstrap_ci(norms,rng,np.mean); split=split_half(x,rng); mean_norms.append(norms.mean())
            effect.append({"prompt_group":group,"pooling_rule":pool,"layer":layer,"adapter":adapter,"mean_delta_l2":norms.mean(),"median_delta_l2":np.median(norms),"mean_delta_l2_ci_low":lo,"mean_delta_l2_ci_high":hi,
              "mean_pairwise_prompt_cosine":mean_pairwise_cosine(x),"split_half_cosine_mean":split[0],"split_half_ci_low":split[1],"split_half_ci_high":split[2],"effective_update_frobenius":UPDATE_MAG[adapter]})
          update_rel.append({"prompt_group":group,"pooling_rule":pool,"layer":layer,"pearson_r":pearsonr(mean_norms,[UPDATE_MAG[a] for a in ADAPTERS]).statistic,"spearman_r":spearmanr(mean_norms,[UPDATE_MAG[a] for a in ADAPTERS]).statistic})
          ep=X4[1]-X4[0]; split=split_half(ep,rng); norms=np.linalg.norm(ep,axis=1); lo,hi=bootstrap_ci(norms,rng,np.mean); centroid=ep.mean(0); endpoint_centroids[(group,pool,layer)]=centroid
          endpoint.append({"prompt_group":group,"pooling_rule":pool,"layer":layer,"mean_endpoint_l2":norms.mean(),"mean_endpoint_l2_ci_low":lo,"mean_endpoint_l2_ci_high":hi,
            "split_half_cosine_mean":split[0],"split_half_ci_low":split[1],"split_half_ci_high":split[2]})
          for row in separability(X4,rng): separability_rows.append({"prompt_group":group,"pooling_rule":pool,"layer":layer}|row)
          alignment,prompt_dist=target_alignment(X4,target_sets,rng)
          for row in alignment: target_rows.append({"prompt_group":group,"pooling_rule":pool,"layer":layer}|row)
          centroids,coords,ratios,erank,participation,dist=centroid_geometry(X4)
          pca_rows.append({"prompt_group":group,"pooling_rule":pool,"layer":layer,"explained_variance_pc1":ratios[0],"explained_variance_pc2":ratios[1],"explained_variance_pc3":ratios[2],"effective_rank":erank,"participation_ratio":participation})
          for ai,adapter in enumerate(ADAPTERS): centroid_rows.append({"prompt_group":group,"pooling_rule":pool,"layer":layer,"adapter":adapter,"pc1":coords[ai,0],"pc2":coords[ai,1]})
          for i,j in itertools.combinations(range(4),2): distance_rows.append({"prompt_group":group,"pooling_rule":pool,"layer":layer,"adapter_a":ADAPTERS[i],"adapter_b":ADAPTERS[j],"centroid_distance":dist[i,j],"mean_prompt_pair_distance":prompt_dist[:,list(itertools.combinations(range(4),2)).index((i,j))].mean()})
    # Endpoint stability across every pair of prompt-group/pooling conditions.
    endpoint_cross=[]
    conditions=list(itertools.product(GROUPS,POOLS))
    for layer in range(32):
      for a,b in itertools.combinations(conditions,2):
        x=endpoint_centroids[(a[0],a[1],layer)]; y=endpoint_centroids[(b[0],b[1],layer)]; den=np.linalg.norm(x)*np.linalg.norm(y)
        endpoint_cross.append({"layer":layer,"condition_a":f"{a[0]}:{a[1]}","condition_b":f"{b[0]}:{b[1]}","cosine":float(np.dot(x,y)/den) if den else 0})
    frames={"effect_magnitude":pd.DataFrame(effect),"effect_update_relationship":pd.DataFrame(update_rel),"endpoint_stability":pd.DataFrame(endpoint),"endpoint_cross_condition":pd.DataFrame(endpoint_cross),
      "separability":pd.DataFrame(separability_rows),"target_alignment":pd.DataFrame(target_rows),"pca_summary":pd.DataFrame(pca_rows),"adapter_centroids":pd.DataFrame(centroid_rows),"pairwise_distances":pd.DataFrame(distance_rows)}
    # BH within each analysis family/group/pool/classifier-or-target across layers.
    sep=frames["separability"]
    for metric in ("balanced_accuracy","macro_f1"):
      sep[f"{metric}_permutation_q_bh"] = sep.groupby(["prompt_group","pooling_rule","classifier"])[f"{metric}_permutation_p"].transform(lambda x:bh(x.values))
    ta=frames["target_alignment"]
    for metric in ("pearson","spearman"):
      ta[f"{metric}_permutation_q_bh"] = ta.groupby(["prompt_group","pooling_rule","target_definition"])[f"{metric}_permutation_p"].transform(lambda x:bh(x.values))
    for name,frame in frames.items(): frame.to_csv(args.output/f"{name}.csv",index=False)
    # Fixed middle-layer summaries are medians over layers 8..24, never selected layers.
    middle={}
    for name in ("effect_magnitude","endpoint_stability","separability","target_alignment","pca_summary"):
      f=frames[name]; numeric=f.select_dtypes(include="number").columns.difference(["layer"]); grouping=[c for c in f.columns if c not in numeric and c!="layer"]
      mid=f[f.layer.between(8,24)].groupby(grouping,dropna=False,as_index=False)[list(numeric)].median(); mid.to_csv(args.output/f"middle_layers_8_24_{name}.csv",index=False); middle[name]=mid
    # Decision: A requires held-out separation and corrected continuous alignment in >1 groups; B requires separation; else C.
    mids=middle["separability"]; separable_conditions=mids[(mids.balanced_accuracy_ci_low>.25)&(mids.balanced_accuracy_permutation_q_bh<.05)]
    reliable_separation=separable_conditions.prompt_group.nunique()>=2
    align=frames["target_alignment"]
    supported=align[(align.layer.between(8,24))&(align.target_definition.isin(["row_weighted","unique_tuple_weighted"]))&(align.pearson_r>0)&(align.spearman_r>0)&(align.pearson_permutation_q_bh<.05)&(align.spearman_permutation_q_bh<.05)]
    aligned_groups=supported.prompt_group.nunique(); category="A" if reliable_separation and aligned_groups>=2 else ("B" if reliable_separation else "C")
    decision={"category":category,"category_label":{"A":"Target-aligned latent structure","B":"Adapter-specific but non-OCEAN structure","C":"No reproducible held-out adapter structure"}[category],
      "reliable_held_out_adapter_separation":bool(reliable_separation),"continuous_target_alignment_supported_prompt_groups":int(aligned_groups),
      "rule":"A requires held-out separation plus positive Pearson/Spearman continuous-target alignment with permutation support after BH in more than one prompt group; B requires held-out separation without A; otherwise C.",
      "scope_caution":"Only four adapters; target analyses are exploratory and underpowered.","prohibited_inference":"Low-rank or reproducible structure is not by itself five-dimensional OCEAN geometry."}
    atomic_json(args.output/"activation_decision.json",decision)
    # Compact figures at fixed layer 16 plus full-layer effect curves; source tables above contain all layers.
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    figdir=args.output/"figures"; figdir.mkdir(exist_ok=True)
    for group in GROUPS:
      for pool in POOLS:
        c=frames["adapter_centroids"]; c=c[(c.prompt_group==group)&(c.pooling_rule==pool)&(c.layer==16)]
        fig,ax=plt.subplots(figsize=(6,5)); ax.scatter(c.pc1,c.pc2,s=70)
        for _,r in c.iterrows(): ax.annotate(r.adapter,(r.pc1,r.pc2),xytext=(4,4),textcoords="offset points")
        ax.set(title=f"Adapter centroids — {group}, {pool}, layer 16",xlabel="PC1",ylabel="PC2"); fig.tight_layout(); stem=figdir/f"centroids_{group}_{pool}_layer16"; fig.savefig(stem.with_suffix(".png"),dpi=180); fig.savefig(stem.with_suffix(".svg")); plt.close(fig)
        d=frames["pairwise_distances"]; d=d[(d.prompt_group==group)&(d.pooling_rule==pool)&(d.layer==16)]; matrix=np.zeros((4,4))
        for _,r in d.iterrows(): i=ADAPTERS.index(r.adapter_a); j=ADAPTERS.index(r.adapter_b); matrix[i,j]=matrix[j,i]=r.centroid_distance
        fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(matrix,cmap="viridis"); ax.set_xticks(range(4),ADAPTERS,rotation=30,ha="right"); ax.set_yticks(range(4),ADAPTERS); ax.set_title(f"Centroid distances — {group}, {pool}, layer 16"); fig.colorbar(im,ax=ax); fig.tight_layout(); stem=figdir/f"distances_{group}_{pool}_layer16"; fig.savefig(stem.with_suffix(".png"),dpi=180); fig.savefig(stem.with_suffix(".svg")); plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,12),sharex=True)
    for ax,(group,pool) in zip(axes.flat,itertools.product(GROUPS,POOLS)):
      f=frames["effect_magnitude"]; f=f[(f.prompt_group==group)&(f.pooling_rule==pool)]
      for adapter in ADAPTERS: ax.plot(f[f.adapter==adapter].layer,f[f.adapter==adapter].mean_delta_l2,label=adapter)
      ax.set_title(f"{group} — {pool}"); ax.set_ylabel("Mean delta L2")
    axes[-1,0].set_xlabel("Layer"); axes[-1,1].set_xlabel("Layer"); axes[0,0].legend(fontsize=8); fig.tight_layout(); fig.savefig(figdir/"effect_magnitude_by_layer.png",dpi=180); fig.savefig(figdir/"effect_magnitude_by_layer.svg"); plt.close(fig)
    lines=["# Activation-study decision","",f"## Category {category}: {decision['category_label']}","",decision["rule"],"",f"Reliable held-out adapter separation: **{reliable_separation}**.",f"Continuous-target aligned prompt groups with corrected support: **{aligned_groups}**.","",decision["scope_caution"],decision["prohibited_inference"]]
    (args.output/"activation_decision.md").write_text("\n".join(lines)+"\n"); print(json.dumps(decision))

if __name__=="__main__":main()
