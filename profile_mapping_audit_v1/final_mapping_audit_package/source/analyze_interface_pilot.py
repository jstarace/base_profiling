"""Strictly merge bounded shards and apply the preregistered behavioral gate."""
from __future__ import annotations

import argparse,hashlib,json,math
from pathlib import Path

import numpy as np
import pandas as pd

from ocean_interface.io import atomic_json,atomic_parquet,strict_merge,validate_manifest
from ocean_interface.spec import INTERFACES,TEMPLATES

MODELS=("base","ptype_0","ptype_31","ptype_9","ptype_23"); TRAITS=list("OCEAN")


def expected_keys(asset):
    keys=set()
    for model in MODELS:
        for interface in INTERFACES:
            permutations=range(5) if interface in INTERFACES[:2] else (None,)
            for item in asset["items"]:
                for template in TEMPLATES:
                    for permutation in permutations: keys.add((model,interface,item["item_id"],template,permutation))
    return keys


def decode(frame):
    for column in ("mapping","candidate_continuations","real_candidate_scores","null_candidate_scores","calibrated_primary_scores","raw_label_probabilities","semantic_probabilities"):
        frame[column]=frame[column].map(lambda x:json.loads(x) if isinstance(x,str) else x)
    return frame


def aggregate(frame):
    item=frame.groupby(["model_key","ptype","interface","item_id","trait","facet_code","facet_name"],dropna=False,as_index=False).post_reversal_score.mean()
    facet=item.groupby(["model_key","ptype","interface","trait","facet_code","facet_name"],dropna=False,as_index=False).agg(facet_score=("post_reversal_score","mean"),item_count=("item_id","nunique"))
    domain=facet.groupby(["model_key","ptype","interface","trait"],dropna=False,as_index=False).agg(domain_mean_1_5=("facet_score","mean"),facet_count=("facet_code","nunique"))
    domain["observed_0_100"]=25*(domain.domain_mean_1_5-1)
    return item,facet,domain


def grouped_domain(frame,column):
    subset=frame if column!="permutation_id" else frame[frame.interface.isin(INTERFACES[:2])]
    item=subset.groupby(["model_key","ptype","interface","trait",column,"item_id"],dropna=False,as_index=False).post_reversal_score.mean()
    result=item.groupby(["model_key","ptype","interface","trait",column],dropna=False,as_index=False).post_reversal_score.mean()
    result["observed_0_100"]=25*(result.post_reversal_score-1); return result


def corr(a,b): return float(pd.Series(a).corr(pd.Series(b)))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--asset",type=Path,required=True); p.add_argument("--targets",type=Path,required=True); p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True); manifest=json.loads(args.manifest.read_text()); fingerprint=validate_manifest(manifest); asset=json.loads(args.asset.read_text())
    paths=sorted(args.root.glob("shards/*/*.parquet")); frame=decode(strict_merge(paths,expected_keys(asset),fingerprint))
    atomic_parquet(args.output/"interface_item_level.parquet",frame)
    item,facet,domain=aggregate(frame); facet.to_csv(args.output/"interface_facet_scores.csv",index=False); domain.to_csv(args.output/"interface_domain_scores.csv",index=False)
    by_template=grouped_domain(frame,"template_id"); by_perm=grouped_domain(frame,"permutation_id")
    by_template.to_csv(args.output/"interface_domain_scores_by_template.csv",index=False); by_perm.to_csv(args.output/"interface_domain_scores_by_permutation.csv",index=False)
    targets=pd.read_csv(args.targets); target_rows=[]
    for _,row in targets[targets.ptype.isin((0,31,9,23))].iterrows():
        for trait in TRAITS: target_rows.append({"ptype":int(row.ptype),"trait":trait,"binary_target_high":int(row[f"{trait}_high"]),
            "row_weighted_target":row[f"row_weighted_{trait}_mean"],"unique_tuple_weighted_target":row[f"unique_trait_tuple_weighted_{trait}_mean"]})
    adapters=domain[domain.model_key!="base"].merge(pd.DataFrame(target_rows),on=["ptype","trait"])
    base=domain[domain.model_key=="base"][["interface","trait","observed_0_100"]].rename(columns={"observed_0_100":"base_0_100"})
    comparisons=adapters.merge(base,on=["interface","trait"]); comparisons["adapter_minus_base"]=comparisons.observed_0_100-comparisons.base_0_100
    comparisons["target_direction"]=np.where(comparisons.binary_target_high.eq(1),1,-1); comparisons["target_directed"]=comparisons.adapter_minus_base*comparisons.target_direction>0
    comparisons["observed_high"]=comparisons.observed_0_100>50; comparisons["bit_correct"]=comparisons.observed_high.eq(comparisons.binary_target_high.astype(bool))
    comparisons.to_csv(args.output/"interface_adapter_target_comparisons.csv",index=False)

    endpoints=[]; gates={}; template_contrasts=[]
    records={record["model_key"]:record for path in (args.root/"run_records").glob("*.json") for record in [json.loads(path.read_text())]}
    deterministic=all(all(v.get("exact",True) and v.get("max_abs_score_difference",0)==0 for v in rec["repeat"].values()) for rec in records.values())
    unload=all(rec["unload_restoration"]=={"max_abs":0.0,"mean_abs":0.0,"l2":0.0} for key,rec in records.items() if key!="base")
    for interface in INTERFACES:
        d=domain[domain.interface==interface]; p0=d[d.model_key=="ptype_0"].set_index("trait"); p31=d[d.model_key=="ptype_31"].set_index("trait")
        bt=by_template[by_template.interface==interface]
        bp=by_perm[by_perm.interface==interface]
        endpoint_pass=True; stable_pass=True; variation_pass=True; ratios=[]
        for trait in TRAITS:
            separation=float(p31.loc[trait,"observed_0_100"]-p0.loc[trait,"observed_0_100"])
            template_values={}
            template_signs=[]
            for template in TEMPLATES:
                a=float(bt[(bt.model_key=="ptype_31")&(bt.trait==trait)&(bt.template_id==template)].observed_0_100.iloc[0])
                b=float(bt[(bt.model_key=="ptype_0")&(bt.trait==trait)&(bt.template_id==template)].observed_0_100.iloc[0])
                template_values[template]=a-b; template_signs.append(a-b>0); template_contrasts.append({"interface":interface,"trait":trait,"template_id":template,"ptype_31_minus_ptype_0":a-b,"intended_sign":a-b>0})
            ranges=[]
            for model in ("ptype_0","ptype_31"):
                vals=bt[(bt.model_key==model)&(bt.trait==trait)].observed_0_100; ranges.append(float(vals.max()-vals.min()))
                if interface in INTERFACES[:2]:
                    vals=bp[(bp.model_key==model)&(bp.trait==trait)].observed_0_100; ranges.append(float(vals.max()-vals.min()))
            variation=max(ranges); intended=separation>0; stable=all(template_signs); larger=abs(separation)>variation
            endpoint_pass &= intended; stable_pass &= stable; variation_pass &= larger; ratios.append(abs(separation)/(variation+1e-12))
            endpoints.append({"interface":interface,"trait":trait,"ptype_31_minus_ptype_0":separation,"intended_sign":intended,
                              "all_template_signs_intended":stable,"domain_prompt_mapping_max_range":variation,
                              "separation_exceeds_variation":larger,"separation_to_variation_ratio":ratios[-1]})
        c=comparisons[comparisons.interface==interface]
        correlations={"binary":corr(c.observed_0_100,c.binary_target_high),"row_weighted":corr(c.observed_0_100,c.row_weighted_target),
                      "unique_tuple_weighted":corr(c.observed_0_100,c.unique_tuple_weighted_target)}
        directed=int(c.target_directed.sum())
        criteria={"all_five_endpoint_signs":endpoint_pass,"endpoint_signs_stable_all_templates":stable_pass,
                  "at_least_15_of_20_target_directed":directed>=15,"positive_row_weighted_correlation":correlations["row_weighted"]>0,
                  "all_endpoint_separations_exceed_prompt_mapping_variation":variation_pass,
                  "deterministic_repeats_and_unload_exact":deterministic and unload}
        gates[interface]={"criteria":criteria,"pass":all(criteria.values()),"target_directed_count":directed,"correlations":correlations,
                          "median_endpoint_separation_to_sensitivity":float(np.median(ratios)),
                          "hamming":c.groupby("model_key").bit_correct.apply(lambda x:int((~x).sum())).to_dict()}
    pd.DataFrame(endpoints).to_csv(args.output/"interface_endpoint_contrasts.csv",index=False); pd.DataFrame(template_contrasts).to_csv(args.output/"interface_endpoint_contrasts_by_template.csv",index=False)

    # Raw candidate preference and keying diagnostics, without selecting subsets.
    raw=[]
    for _,row in frame.iterrows():
        scores=row["real_candidate_scores"]
        for candidate,data in scores.items(): raw.append({"model_key":row.model_key,"interface":row.interface,"trait":row.trait,"template_id":row.template_id,
            "permutation_id":row.permutation_id,"candidate":candidate,"total_log_likelihood":data["total_log_likelihood"],"mean_token_log_likelihood":data["mean_token_log_likelihood"],
            "next_token_logit":data["next_token_logit"]})
    pd.DataFrame(raw).groupby(["model_key","interface","trait","template_id","permutation_id","candidate"],dropna=False,as_index=False).mean(numeric_only=True).to_csv(args.output/"interface_raw_candidate_summary.csv",index=False)
    frame.groupby(["model_key","interface","trait","negative_keyed"],dropna=False,as_index=False).agg(n=("item_id","size"),unique_items=("item_id","nunique"),
        pre_reversal_mean=("pre_reversal_score","mean"),post_reversal_mean=("post_reversal_score","mean")).to_csv(args.output/"interface_positive_negative_key_summary.csv",index=False)
    passing=[name for name,value in gates.items() if value["pass"]]
    selected=max(passing,key=lambda name:gates[name]["median_endpoint_separation_to_sensitivity"]) if passing else None
    authorization={"interfaces":gates,"passing_interfaces":passing,"selected_interface":selected,
        "full_32_adapter_run_authorized":bool(selected),"determinism_exact":deterministic,"unload_restoration_exact":unload,
        "decision":("AUTHORIZED_FOR_LATER_REVIEW_NOT_LAUNCHED" if selected else "STOP_NO_INTERFACE_PASSED"),
        "failure_conclusion":None if selected else "Questionnaire self-report scoring is not a suitable behavioral interface for this pretrained base under the tested methods."}
    atomic_json(args.output/"behavioral_authorization_gate.json",authorization)
    lines=["# Bounded scoring-interface pilot","",f"Decision: **{authorization['decision']}**.","",f"Full 32-adapter run launched: **No**.",""]
    for name,result in gates.items():
        lines += [f"## {name}","",f"Overall pass: **{result['pass']}**; target-directed changes: **{result['target_directed_count']}/20**; row-target correlation: **{result['correlations']['row_weighted']:.6f}**.",""]
        lines += [f"- {criterion}: {value}" for criterion,value in result["criteria"].items()]+[""]
    if authorization["failure_conclusion"]: lines += ["## Required stop conclusion","",authorization["failure_conclusion"],""]
    lines += ["No interface, template, mapping, item, facet, or trait was selected based on favorable individual results."]
    (args.output/"behavioral_authorization_gate.md").write_text("\n".join(lines)+"\n")
    audit={"shard_count":len(paths),"merged_rows":len(frame),"expected_rows":len(expected_keys(asset)),"duplicate_keys":0,"missing_keys":0,
           "manifest_fingerprint":fingerprint,"raw_benchmark_sha256":hashlib.sha256(args.asset.read_bytes()).hexdigest(),"base_revision":manifest["base_revision"]}
    atomic_json(args.output/"strict_merge_audit.json",audit); print(json.dumps(authorization))

if __name__=="__main__":main()
