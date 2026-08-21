"""Build the frozen 1,080-prompt corpus while preserving the 360 legacy texts."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


NATURAL_CATEGORIES=("planning","novelty","routine","uncertainty","social_interaction","cooperation","disagreement","conflict","deadlines","emotional_response","risk","personal_reflection")
CONTEXTS=("community garden","software release","repair project","travel schedule","shared workspace","research note","delivery route","class project","neighborhood event","household task","meeting agenda","weekend activity")
NEW_NEUTRAL_OBJECTS=("granite cylinder","brass washer","acrylic panel","maple dowel","slate tile","linen strip","iron bracket","porcelain disk","sandstone block","silicon wafer","wool cord","zinc plate")
NEUTRAL_STEMS=(
 "The measurement ledger lists the {object} on the numbered shelf. The next recorded field is",
 "A technical diagram places the {object} beside the reference scale. The caption continues with",
 "The inventory sheet assigns the {object} a storage code. The following entry gives",
 "A surface scan of the {object} was saved with the other samples. The file header records",
 "The balance was calibrated before the {object} was weighed. The next line in the log shows",
 "A temperature probe was positioned near the {object}. The observation table continues with",
 "The reference photograph shows the {object} against a ruled background. Beneath it appears",
 "The materials catalog indexes the {object} by specimen number. Its description continues with",
 "The floor plan marks the {object} near the north wall. The adjacent annotation reads",
 "The inspection form identifies the {object} by serial number. The remaining field states",
)
VARIANTS={
 "planning":("The work sequence for the {context} was drafted before one dependency changed. The revised first step was","Several stages of the {context} depended on one another. After the order was reviewed, the next step was","The {context} had a written schedule with two tasks competing for the same resource. The plan continued by","A checklist for the {context} revealed an unfinished prerequisite. The sequence was adjusted to"),
 "novelty":("A method not previously used in the {context} became available. After examining it, the next move was","An unfamiliar tool appeared during the {context}, along with a short technical note. The work continued by","The usual approach to the {context} was replaced by a newly proposed procedure. The immediate response was","A new possibility emerged halfway through the {context}. Before proceeding, the group decided to"),
 "routine":("The established procedure for the {context} began at the usual time. Once the first check was complete,","The {context} followed the same sequence used on earlier occasions. The next familiar action was","A standard checklist guided the {context} from the opening step. The process continued with","The recurring {context} reached the point marked in the regular schedule. At that point,"),
 "uncertainty":("A key detail about the {context} remained unconfirmed. With the available information, the next step was","Two reports about the {context} gave different estimates. Before committing to either,","The outcome of the {context} depended on information expected later. In the meantime,","An ambiguous note changed how the {context} might proceed. The immediate response was"),
 "social_interaction":("During the {context}, another participant asked for a concise update. The reply began","A new participant joined the {context} and asked what had happened so far. The explanation started","Someone involved in the {context} requested clarification about the next step. The response was","At a pause in the {context}, a colleague opened a conversation about progress. The answer began"),
 "cooperation":("Two participants in the {context} needed to divide the remaining tasks. They proceeded by","The {context} required several people to coordinate access to one resource. The group arranged to","A shared part of the {context} could not be completed by one person alone. The collaborators began by","Progress on the {context} depended on combining work from separate participants. The next joint step was"),
 "disagreement":("Two proposed approaches to the {context} led to different conclusions. The discussion moved forward when","Participants in the {context} disagreed about which evidence mattered most. The next contribution was","A difference of opinion interrupted the {context} before a decision was made. The conversation continued with","Two interpretations of the {context} remained in contention. To clarify the choice, someone"),
 "conflict":("A tense exchange interrupted the {context} while work was still underway. The immediate response was","Competing demands during the {context} produced an open conflict. Before resuming the work,","An argument over responsibilities stalled the {context}. The next thing said was","The {context} reached an impasse after two participants challenged each other. The situation continued when"),
 "deadlines":("The completion time for the {context} moved earlier with little notice. The schedule was revised by","A deadline attached to the {context} was approaching while one task remained unfinished. The next action was","The available time for the {context} was reduced after work began. In response,","A final deliverable from the {context} was due sooner than expected. The work continued with"),
 "emotional_response":("An unexpected setback occurred during the {context}, changing the atmosphere immediately. The first response was","News about the {context} produced a strong reaction among those present. After a brief pause,","A difficult result emerged from the {context} without warning. The immediate response began","The {context} ended with an outcome no one had anticipated. In the moments that followed,"),
 "risk":("One option in the {context} offered a larger benefit but had an uncertain downside. Before choosing,","The next step in the {context} involved a small chance of a costly failure. The decision proceeded by","A proposed change to the {context} could save time while introducing an untested condition. The response was","The {context} presented a choice between a predictable outcome and a variable one. The next consideration was","New information made the safest route through the {context} less certain. The decision continued with"),
 "personal_reflection":("After the {context} ended, one participant reviewed what had gone differently than expected. The reflection began","A quiet moment after the {context} provided time to consider the earlier choices. The first thought was","Looking back on the {context}, one detail seemed especially important. The written reflection continued","The outcome of the {context} prompted a private review of the decisions made along the way. It began","Later that day, the {context} came to mind again. The reflection turned to"),
}


def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()


def assigned_splits(rows: list[dict]) -> None:
    strata={}
    for row in rows: strata.setdefault((row["group"],row["category"]),[]).append(row)
    for (_, _),items in strata.items():
        items.sort(key=lambda x: hashlib.sha256(x["prompt_id"].encode()).hexdigest())
        n=len(items)
        if n==4: counts=(2,1,1)
        elif n==60: counts=(36,12,12)
        else:
            train=round(.6*n); validation=round(.2*n); counts=(train,validation,n-train-validation)
        for split,count in zip(("train","validation","test"),counts):
            for row in items[:count]: row["split"]=split
            del items[:count]


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--legacy-raw",type=Path,required=True); ap.add_argument("--tokenizer",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); args=ap.parse_args()
    legacy=json.loads(args.legacy_raw.read_text())["records"]
    if len(legacy)!=360: raise RuntimeError("legacy corpus is not 360 prompts")
    rows=[]
    category_alias={"emotional_reactions":"emotional_response"}
    for item in legacy:
        row=dict(item); row["category"]=category_alias.get(row["category"],row["category"]); row["legacy_core"]=True; row["legacy_prompt_sha256"]=sha(item["text"].encode()); rows.append(row)
    # Add 120 non-social, factual continuation controls.
    for object_name in NEW_NEUTRAL_OBJECTS:
        for index,stem in enumerate(NEUTRAL_STEMS,1):
            rows.append({"prompt_id":f"DRIFT-NEUTRAL-{len(rows)-359:03d}","group":"neutral_controls","category":f"neutral_{index:02d}","text":stem.format(object=object_name),"construction_or_source_note":"Study-authored factual continuation control; frozen before capture.","legacy_core":False})
    # Expand ten legacy categories by 48 and create 60 prompts for each new category.
    for category in NATURAL_CATEGORIES:
        wanted=48 if category not in {"risk","personal_reflection"} else 60
        variants=VARIANTS[category]
        made=0
        for variant_index,template in enumerate(variants):
            for context in CONTEXTS:
                if made>=wanted: break
                made += 1
                rows.append({"prompt_id":f"DRIFT-NAT-{category.upper()}-{made:03d}","group":"naturalistic_behavioral","category":category,"text":template.format(context=context),"construction_or_source_note":"Study-authored naturalistic continuation stem; frozen before capture.","legacy_core":False})
        if made!=wanted: raise RuntimeError((category,made,wanted))
    if len(rows)!=1080 or len({r['prompt_id'] for r in rows})!=1080 or len({r['text'] for r in rows})!=1080: raise RuntimeError("prompt count, ID, or text uniqueness failure")
    counts=Counter((r["group"],r["category"]) for r in rows)
    if sum(v for (g,_),v in counts.items() if g=="naturalistic_behavioral")!=720 or any(counts[("naturalistic_behavioral",c)]!=60 for c in NATURAL_CATEGORIES): raise RuntimeError("naturalistic category coverage failure")
    assigned_splits(rows)
    raw={"schema_version":"2.0","legacy_source_raw_sha256":sha(args.legacy_raw.read_bytes()),"records":rows}
    raw_bytes=(json.dumps(raw,indent=2,ensure_ascii=False)+"\n").encode(); raw_sha=sha(raw_bytes)
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    enriched=[]
    for row in rows:
        ids=tokenizer.encode(row["text"],add_special_tokens=True)
        enriched.append(row|{"token_ids":ids,"token_count":len(ids),"prompt_text_sha256":sha(row["text"].encode()),"raw_file_sha256":raw_sha})
    core={"schema_version":"2.0","raw_prompt_file_sha256":raw_sha,"tokenizer_policy":"verified stored adapter tokenizer; add_special_tokens=True","records":enriched}
    fingerprint=sha(json.dumps(core,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode())
    for row in enriched: row["manifest_fingerprint"]=fingerprint
    manifest=core|{"prompt_manifest_fingerprint":fingerprint}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"prompts_raw.json").write_bytes(raw_bytes); (args.output_dir/"prompt_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    frame=pd.DataFrame(enriched); frame.assign(token_ids=frame.token_ids.map(json.dumps)).to_csv(args.output_dir/"prompt_manifest.csv",index=False)
    frame.groupby(["group","category","split"],as_index=False).agg(prompt_count=("prompt_id","size"),mean_token_count=("token_count","mean"),min_token_count=("token_count","min"),max_token_count=("token_count","max"),legacy_core_count=("legacy_core","sum")).to_csv(args.output_dir/"prompt_manifest_summary.csv",index=False)
    print(json.dumps({"records":len(rows),"raw_sha256":raw_sha,"fingerprint":fingerprint,"groups":Counter(r['group'] for r in rows)}),flush=True)


if __name__ == "__main__": main()
