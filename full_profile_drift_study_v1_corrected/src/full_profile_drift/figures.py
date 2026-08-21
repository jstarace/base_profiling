"""Generate the 18 preregistered publication figures with CSV source data."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist,squareform
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

TRAITS=tuple("OCEAN")

def finish(fig,root,number,slug,data):
    root.mkdir(parents=True,exist_ok=True); (root/"data").mkdir(exist_ok=True); data.to_csv(root/"data"/f"figure_{number:02d}_{slug}.csv",index=False)
    fig.tight_layout(); fig.savefig(root/f"figure_{number:02d}_{slug}.png",dpi=220,bbox_inches="tight"); fig.savefig(root/f"figure_{number:02d}_{slug}.svg",bbox_inches="tight"); plt.close(fig)

def heat(matrix,title,cbar="Distance"):
    fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(matrix,cmap="viridis"); ax.set(title=title,xlabel="ptype",ylabel="ptype"); ax.set_xticks(range(0,32,2)); ax.set_yticks(range(0,32,2)); fig.colorbar(im,ax=ax,label=cbar); return fig

def mds(distance,k=2):
    n=len(distance); h=np.eye(n)-np.ones((n,n))/n; g=-.5*h@(distance**2)@h; v,u=np.linalg.eigh(g); order=np.argsort(v)[::-1]; return u[:,order[:k]]*np.sqrt(np.clip(v[order[:k]],0,None))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project",type=Path,required=True); ap.add_argument("--output-root",type=Path,required=True); args=ap.parse_args(); p=args.project; o=args.output_root; a=o/"analysis_outputs"; f=o/"figures"
    catalog=pd.read_csv(p/"ptype_catalog.csv").sort_values("ptype"); exp=pd.read_csv(p/"training_exposure/all_32_training_exposure.csv").sort_values("ptype"); metrics=pd.read_csv(a/"profile_exposure_effect_metrics.csv"); dist=pd.read_parquet(a/"all_representation_pairwise_distances.parquet")
    # 1 catalog/exposure map.
    d=catalog[["ptype","human_readable_profile","row_count","effective_update_frobenius_norm"]].merge(exp[["ptype","total_retained_tokens_after_512"]],on="ptype"); fig,ax=plt.subplots(figsize=(9,6)); sc=ax.scatter(d.row_count,d.total_retained_tokens_after_512,s=35+200*d.effective_update_frobenius_norm/d.effective_update_frobenius_norm.max(),c=d.ptype,cmap="turbo"); [ax.annotate(str(r.ptype),(r.row_count,r.total_retained_tokens_after_512),fontsize=7) for r in d.itertuples()]; ax.set(xscale="log",yscale="log",xlabel="Training rows",ylabel="Retained tokens",title="All-32 profile catalog and exposure map"); fig.colorbar(sc,ax=ax,label="ptype"); finish(fig,f,1,"profile_catalog_exposure",d)
    # 2 exposure versus update norm.
    fig,ax=plt.subplots(figsize=(8,6)); ax.scatter(metrics.retained_tokens,metrics.update_frobenius_norm,c=metrics.ptype,cmap="turbo"); [ax.annotate(str(r.ptype),(r.retained_tokens,r.update_frobenius_norm),fontsize=7) for r in metrics.itertuples()]; ax.set(xscale="log",xlabel="Retained-token exposure",ylabel="Effective-update Frobenius norm",title="Exposure versus LoRA update magnitude"); finish(fig,f,2,"exposure_update_norm",metrics)
    def average_matrix(sub):
        frame=sub.groupby(["ptype_a","ptype_b"],as_index=False).distance.mean(); m=np.zeros((32,32)); m[frame.ptype_a.astype(int),frame.ptype_b.astype(int)]=frame.distance; return m+m.T,frame
    wm,wdata=average_matrix(dist[dist.representation=="weights"]); finish(heat(wm,"All-32 weight-space distance"),f,3,"weight_distance_matrix",wdata)
    am,adata=average_matrix(dist[dist.representation=="activation"]); finish(heat(am,"All-32 activation distance (all layers/contexts/pools mean)"),f,4,"activation_distance_matrix",adata)
    lm,ldata=average_matrix(dist[dist.representation=="logits"]); finish(heat(lm,"All-32 logit distance (prompt groups mean)"),f,5,"logit_distance_matrix",ldata)
    # 6 fixed held-out confusion.
    cm=np.loadtxt(a/"confusion_naturalistic_behavioral_final_token_nearest_centroid_layer31.csv",delimiter=","); finish(heat(cm,"32-way held-out confusion: fixed naturalistic/final/layer 31","Count"),f,6,"heldout_confusion",pd.DataFrame(cm).reset_index().melt("index",var_name="predicted",value_name="count").rename(columns={"index":"true"}))
    # 7 per-profile recall and uniqueness.
    recall=np.diag(cm)/np.maximum(cm.sum(1),1); ud=pd.DataFrame({"ptype":range(32),"profile":catalog.human_readable_profile,"heldout_recall":recall}); fig,ax=plt.subplots(figsize=(10,5)); ax.bar(ud.ptype,ud.heldout_recall); ax.axhline(1/32,color="red",ls="--",label="chance"); ax.set(xlabel="ptype",ylabel="Recall",title="Adapter uniqueness by profile"); ax.legend(); finish(fig,f,7,"adapter_uniqueness",ud)
    # 8 Hamming-one graph on weight MDS.
    xy=mds(wm); gd=pd.DataFrame({"ptype":range(32),"x":xy[:,0],"y":xy[:,1],"row_count":catalog.row_count,"profile":catalog.human_readable_profile}); edges=[]; fig,ax=plt.subplots(figsize=(9,7));
    for i in range(32):
      for j in range(i+1,32):
        h=sum(catalog.iloc[i][f"{t}_high"]!=catalog.iloc[j][f"{t}_high"] for t in TRAITS)
        if h==1: ax.plot(xy[[i,j],0],xy[[i,j],1],color="0.8",lw=.6,zorder=1); edges.append({"ptype_a":i,"ptype_b":j,"hamming":h})
    ax.scatter(xy[:,0],xy[:,1],c=range(32),cmap="turbo",zorder=2); [ax.annotate(str(i),xy[i],fontsize=7) for i in range(32)]; ax.set(title="Weight nearest-neighbor geometry with Hamming-distance-1 edges",xlabel="MDS 1",ylabel="MDS 2"); finish(fig,f,8,"hamming_neighbor_graph",pd.DataFrame(edges))
    # 9 weight PCA/MDS exposure-sized.
    fig,ax=plt.subplots(figsize=(8,6)); size=30+220*np.log1p(catalog.row_count)/np.log1p(catalog.row_count).max(); ax.scatter(xy[:,0],xy[:,1],s=size,c=range(32),cmap="turbo",alpha=.8); [ax.annotate(str(i),xy[i],fontsize=7) for i in range(32)]; ax.set(title="Weight-space projection (node size = training exposure)",xlabel="MDS 1",ylabel="MDS 2"); finish(fig,f,9,"weight_projection_exposure",gd)
    # Aggregate activation final layer, then residualize fixed exposure covariates.
    cents=[]
    for file in sorted(a.glob("activation_centroids_*.npz")): cents.append(np.load(file)["centroids"][:,31])
    cy=np.mean(cents,axis=0); ex=np.column_stack([np.ones(32),np.log1p(exp.total_raw_rows),np.log1p(exp.total_retained_tokens_after_512),np.log1p(exp.nominal_optimizer_steps),np.log1p(exp.unique_trait_tuple_proxy_count)]); residual=cy-ex@np.linalg.pinv(ex)@cy; rxy=mds(squareform(pdist(residual))); rd=pd.DataFrame({"ptype":range(32),"x":rxy[:,0],"y":rxy[:,1],"profile":catalog.human_readable_profile}); fig,ax=plt.subplots(figsize=(8,6)); ax.scatter(rxy[:,0],rxy[:,1],s=size,c=range(32),cmap="turbo"); [ax.annotate(str(i),rxy[i],fontsize=7) for i in range(32)]; ax.set(title="Activation projection after exposure residualization",xlabel="MDS 1",ylabel="MDS 2"); finish(fig,f,10,"activation_projection_exposure_residualized",rd)
    # 11 Walsh energy order by layer.
    wh=pd.read_csv(a/"walsh_hadamard_energy_by_order.csv"); wd=wh[(wh.transformation=="raw")&(wh.representation=="activation")].groupby(["layer","interaction_order"],as_index=False).energy.mean(); pivot=wd.pivot(index="layer",columns="interaction_order",values="energy"); pivot=pivot.div(pivot.sum(1),axis=0); fig,ax=plt.subplots(figsize=(10,6)); [ax.plot(pivot.index,pivot[c],label=f"order {c}") for c in pivot.columns]; ax.set(xlabel="Layer",ylabel="Energy fraction",title="Walsh–Hadamard energy by interaction order and layer"); ax.legend(ncol=3); finish(fig,f,11,"walsh_energy_by_layer",wd)
    # 12 trait-flip stability.
    flips=pd.read_csv(a/"matched_trait_flip_stability.csv"); fd=flips[flips.representation=="activation"].groupby(["layer","trait"],as_index=False).mean_pairwise_cosine.mean(); fig,ax=plt.subplots(figsize=(10,6)); [ax.plot(g.layer,g.mean_pairwise_cosine,label=t) for t,g in fd.groupby("trait")]; ax.axhline(0,color="0.5",lw=.8); ax.set(xlabel="Layer",ylabel="Mean matched-flip cosine",title="Trait-flip direction stability across all 16 matched pairs"); ax.legend(); finish(fig,f,12,"trait_flip_stability",fd)
    # 13 observed order share versus the combinatorial term-count baseline.
    md=wh[(wh.transformation=="raw")&(wh.interaction_order>=1)].copy(); totals=md.groupby(["representation","prompt_group","pooling_rule","layer"]).energy.transform("sum"); md["energy_share"]=md.energy/totals; md=md.groupby(["representation","interaction_order"],as_index=False).energy_share.median(); counts={1:5,2:10,3:10,4:5,5:1}; md["combinatorial_expected_share"]=md.interaction_order.map(lambda order:counts[order]/31); fig,ax=plt.subplots(figsize=(10,6));
    for representation,group in md.groupby("representation"):
        ax.plot(group.interaction_order,group.energy_share,marker="o",label=f"{representation} observed")
    ax.plot(range(1,6),[counts[order]/31 for order in range(1,6)],color="black",ls="--",marker="o",label="term-count expectation"); ax.set(xlabel="Walsh interaction order",ylabel="Median share of non-intercept energy",title="Walsh order energy requires a combinatorial baseline",xticks=range(1,6)); ax.legend(); finish(fig,f,13,"main_vs_interaction_context",md)
    # 14 alignment before/after exposure removal, fixed all-context centroid by layer.
    target=catalog[[f"row_weighted_{t}_centroid" for t in TRAITS]].to_numpy(); tv=pdist(target); al=[]
    for layer in range(32):
        raw=np.mean([np.load(file)["centroids"][:,layer] for file in sorted(a.glob("activation_centroids_*.npz"))],axis=0); res=raw-ex@np.linalg.pinv(ex)@raw
        al.extend([{"layer":layer,"transformation":"raw","spearman_rho":spearmanr(pdist(raw),tv).statistic},{"layer":layer,"transformation":"exposure_residualized","spearman_rho":spearmanr(pdist(res),tv).statistic}])
    al=pd.DataFrame(al); fig,ax=plt.subplots(figsize=(10,5)); [ax.plot(g.layer,g.spearman_rho,label=t) for t,g in al.groupby("transformation")]; ax.axhline(0,color="0.5",lw=.8); ax.set(xlabel="Layer",ylabel="Spearman distance correlation",title="Continuous-target alignment before and after exposure removal"); ax.legend(); finish(fig,f,14,"alignment_exposure_removal",al)
    # 15 compact cross-representation RSA.
    target_m=squareform(pdist(target)); exposure_m=squareform(pdist(np.log1p(exp.total_retained_tokens_after_512.to_numpy())[:,None])); reps={"weights":wm,"activations":am,"logits":lm,"continuous targets":target_m,"token exposure":exposure_m}; labels=list(reps); rsa=np.array([[spearmanr(squareform(reps[x],checks=False),squareform(reps[y],checks=False)).statistic for y in labels] for x in labels]); fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(rsa,vmin=-1,vmax=1,cmap="coolwarm"); ax.set_xticks(range(len(labels)),labels,rotation=30,ha="right"); ax.set_yticks(range(len(labels)),labels); ax.set_title("Cross-representation similarity"); fig.colorbar(im,ax=ax,label="Spearman rho"); finish(fig,f,15,"cross_representation_similarity",pd.DataFrame(rsa,index=labels,columns=labels).reset_index().melt("index",var_name="representation_b",value_name="spearman_rho").rename(columns={"index":"representation_a"}))
    # 16 sparse-profile sensitivity.
    sens=pd.read_csv(a/"sparse_profile_sensitivity.csv"); sd=sens[(sens.target=="row_weighted_continuous")].groupby("subset",as_index=False).pearson_r.mean().sort_values("pearson_r"); fig,ax=plt.subplots(figsize=(10,6)); ax.barh(sd.subset,sd.pearson_r); ax.axvline(0,color="0.5"); ax.set(xlabel="Mean Pearson distance correlation",title="Sparse-profile sensitivity comparison"); finish(fig,f,16,"sparse_profile_sensitivity",sd)
    # 17 fingerprints.
    fp=pd.read_csv(a/"profile_drift_fingerprints.csv"); fp=fp.groupby(["ptype","layer"],as_index=False).centroid_norm.mean(); fig,ax=plt.subplots(figsize=(11,7)); [ax.plot(g.layer,g.centroid_norm,lw=.8,alpha=.75,label=str(pt)) for pt,g in fp.groupby("ptype")]; ax.set(xlabel="Layer",ylabel="Mean centroid norm",title="Profile drift fingerprints across layers"); finish(fig,f,17,"profile_drift_fingerprints",fp)
    # 18 common drift.
    cp=pd.read_csv(a/"common_drift_projections.csv"); cp=cp.groupby("ptype",as_index=False).common_drift_projection.mean(); fig,ax=plt.subplots(figsize=(10,5)); ax.bar(cp.ptype,cp.common_drift_projection); ax.set(xlabel="ptype",ylabel="Mean common-drift projection",title="Common-drift projection by adapter"); finish(fig,f,18,"common_drift_projection",cp)
    atomic={"complete":True,"figures":18,"formats":["png","svg"],"underlying_csv_directory":str(f/"data")}; (f/"figure_manifest.json").write_text(json.dumps(atomic,indent=2)+"\n")

if __name__=="__main__":main()
