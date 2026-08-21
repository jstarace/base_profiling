"""General, non-personality interpretation of deterministic continuation signatures."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,f1_score,silhouette_score
from sklearn.pipeline import FeatureUnion

from .io import atomic_json

PRONOUNS={"first_singular":{"i","me","my","mine","myself"},"first_plural":{"we","us","our","ours","ourselves"},"second":{"you","your","yours","yourself","yourselves"},"third":{"he","she","him","her","his","hers","they","them","their","theirs"}}


def metrics(text):
    words=re.findall(r"[A-Za-z']+",text.lower()); counts=Counter(words); bigrams=list(zip(words,words[1:])); sentences=[x for x in re.split(r"[.!?]+",text) if x.strip()]
    row={"word_count":len(words),"unique_word_count":len(counts),"type_token_ratio":len(counts)/len(words) if words else 0,"mean_sentence_words":len(words)/len(sentences) if sentences else len(words),"repeated_bigram_fraction":1-len(set(bigrams))/len(bigrams) if bigrams else 0}
    for mark,name in ((".","periods"),(",","commas"),("!","exclamations"),("?","questions"),(";","semicolons"),(':',"colons")): row[name]=text.count(mark)
    for name,vocab in PRONOUNS.items(): row[f"{name}_pronouns"]=sum(counts[w] for w in vocab)
    return row


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project",type=Path,required=True); ap.add_argument("--output-root",type=Path); args=ap.parse_args(); p=args.project; output_root=args.output_root or p; out=output_root/"continuation_outputs"; manifest=json.loads((p/"prompt_manifest/prompt_manifest.json").read_text()); split={r["prompt_id"]:r["split"] for r in manifest["records"]}
    rows=[]
    for key in ["base"]+[f"ptype_{i}" for i in range(32)]:
        data=json.loads((out/"conditions"/f"{key}.json").read_text())
        if len(data["records"])!=120: raise RuntimeError(f"missing continuation condition: {key}")
        for record in data["records"]: rows.append(record|{"split":split[record["prompt_id"]]}|metrics(record["continuation_text"]))
    frame=pd.DataFrame(rows); frame.to_parquet(out/"continuation_records.parquet",index=False)
    metric_cols=[c for c in frame.columns if c in metrics("")]
    frame.groupby("model_key",as_index=False)[metric_cols+["continuation_token_count"]].mean().to_csv(out/"continuation_signature_metrics.csv",index=False)
    adapters=frame[frame.model_key!="base"].copy(); adapters["ptype"]=adapters.model_key.str.removeprefix("ptype_").astype(int); train=adapters.split.isin(["train","validation"]); test=adapters.split.eq("test")
    vectorizer=FeatureUnion([("word",TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=12000,sublinear_tf=True)),("char",TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,max_features=12000,sublinear_tf=True))]); xtrain=vectorizer.fit_transform(adapters.loc[train,"continuation_text"]); xtest=vectorizer.transform(adapters.loc[test,"continuation_text"]); ytrain=adapters.loc[train,"ptype"]; ytest=adapters.loc[test,"ptype"]
    model=LogisticRegression(C=1,max_iter=1000,solver="lbfgs").fit(xtrain,ytrain); pred=model.predict(xtest); confusion=confusion_matrix(ytest,pred,labels=range(32)); np.savetxt(out/"continuation_32way_confusion.csv",confusion,fmt="%d",delimiter=",")
    # Nearest-centroid classifier in the same frozen TF-IDF embedding space.
    centroids=np.stack([np.asarray(xtrain[ytrain.to_numpy()==i].mean(0)).ravel() for i in range(32)]); xte=xtest.toarray(); d=((xte[:,None,:]-centroids[None,:,:])**2).sum(2); nearest=d.argmin(1)
    classification=pd.DataFrame([{"classifier":"regularized_logistic","balanced_accuracy":balanced_accuracy_score(ytest,pred),"macro_f1":f1_score(ytest,pred,average="macro"),"test_rows":len(ytest)},{"classifier":"nearest_tfidf_centroid","balanced_accuracy":balanced_accuracy_score(ytest,nearest),"macro_f1":f1_score(ytest,nearest,average="macro"),"test_rows":len(ytest)}]); classification.to_csv(out/"continuation_adapter_classification.csv",index=False)
    # Top word n-grams from a separate interpretable word-only model.
    word=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=20000,sublinear_tf=True); wx=word.fit_transform(adapters.loc[train,"continuation_text"]); wm=LogisticRegression(C=1,max_iter=1000,solver="lbfgs").fit(wx,ytrain); vocab=np.array(word.get_feature_names_out()); top=[]
    for ptype,coef in enumerate(wm.coef_):
        for rank,index in enumerate(np.argsort(coef)[-20:][::-1],1): top.append({"ptype":ptype,"model_key":f"ptype_{ptype}","rank":rank,"ngram":vocab[index],"coefficient":coef[index]})
    pd.DataFrame(top).to_csv(out/"top_discriminative_ngrams.csv",index=False)
    atomic_json(out/"continuation_analysis_summary.json",{"complete":True,"conditions":33,"prompts_per_condition":120,"generation":"greedy 64-token maximum","embedding_space":"frozen train-only word and character TF-IDF","interpretation_scope":"general continuation signatures only; no personality interpretation","classification":classification.to_dict("records")})
    atomic_json(output_root/"progress.json",{"project":"full_profile_drift_study_v1","stage":"continuation_analysis_complete","integrity_failure":None})

if __name__=="__main__": main()
