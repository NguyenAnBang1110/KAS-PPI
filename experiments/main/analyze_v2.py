"""Summaries for explain_v2 runs: per run and pooled per (pipeline, model) over folds."""
import json, glob, collections, numpy as np
from scipy.stats import wilcoxon
rng=np.random.default_rng(0)
def curve(r,key,lo=0):
    v=r.get(key); return float(np.mean(v[lo:])) if v else None
def ci(a):
    b=[np.mean(rng.choice(a,len(a))) for _ in range(2000)]; return [float(x) for x in np.percentile(b,[2.5,97.5])]
def summarize(R,methods):
    out={"n":len(R),"acc":float(np.mean([r["pred"]==r["y"] for r in R])),
         "k_median":float(np.median([r["k"] for r in R])),"k_zero":int(sum(r["k"]==0 for r in R))}
    dels={};inss={}
    for m in methods+["random"]:
        # empty explanation: nothing is removed (deletion stays at conf), nothing is added back (insertion stays at base)
        dels[m]=np.array([curve(r,f"del_{m}") if r.get(f"del_{m}") else r["conf"] for r in R])
        inss[m]=np.array([curve(r,f"ins_{m}",1) if r.get(f"ins_{m}") else (r["ins_random"][0] if r.get("ins_random") else r["conf"]) for r in R])
        out[f"del_{m}"]=dict(mean=float(dels[m].mean()),ci95=ci(dels[m]))
        out[f"ins_{m}"]=dict(mean=float(inss[m].mean()),ci95=ci(inss[m]))
    for m in methods[1:]+["random"]:
        for nm,arr,sign in [("del",dels,1),("ins",inss,-1)]:
            a,b=arr["D"],arr[m]; d=(a-b)*sign
            try: p=float(wilcoxon(a,b).pvalue)
            except ValueError: p=1.0
            out[f"wil_{nm}_D_vs_{m}"]=dict(diff=float((a-b).mean()),D_better=float((d<0).mean()),ties=float((d==0).mean()),p=p)
    for m in methods:
        out[f"pars_{m}"]=float(np.mean([r[f"size_{m}"] for r in R]))
        out[f"cov_{m}"]=float(np.mean([r[f"cov_{m}"] for r in R]))
        out[f"cons_{m}"]=float(np.mean([r[f"cons_jacc_{m}"] for r in R]))
        c=[r[f"cont_{m}"] for r in R if r[f"cont_{m}"] is not None]; out[f"cont_{m}"]=float(np.mean(c)); out["cont_n"]=len(c)
        out[f"time_{m}"]=float(np.median([r[f"time_{m}"] for r in R]))
    if "LOO" in methods: out["D_vs_LOO_top3_differs"]=float(np.mean([r["top3_D"]!=r["top3_LOO"] for r in R]))
    if "FS0" in methods: out["D_vs_FS0_top3_differs"]=float(np.mean([r["top3_D"]!=r["top3_FS0"] for r in R]))
    return out
runs={}; pooled=collections.defaultdict(list); labels={}
for f in sorted(glob.glob("rec_*_f*_*.json")):
    J=json.load(open(f)); labels=J["labels"]; R=J["records"]
    methods=[k[5:] for k in R[0] if k.startswith("time_")]
    runs[f]=summarize(R,methods); pooled[(J["pipe"],J["model"])]+=R
allout={"runs":runs,"pooled":{}}
for (p,mo),R in pooled.items():
    methods=[k[5:] for k in R[0] if k.startswith("time_")]
    s=summarize(R,methods); allout["pooled"][f"{p}|{mo}"]=s
    # top aspects in D explanations of correctly predicted positives (input for a plausibility case study)
    cnt=collections.Counter(a for r in R if r["y"]==1 and r["pred"]==1 for a in r["top3_D"][:1])
    s["top1_aspect_TP"]=[(labels.get(a,a),c) for a,c in cnt.most_common(8)]
json.dump(allout,open("summary_v2.json","w"),indent=1)
for key,s in allout["pooled"].items():
    print("=====",key,"n",s["n"],"acc %.3f"%s["acc"],"k_med",s["k_median"])
    ms=[k[4:] for k in s if k.startswith("del_")]
    for m in ms: print("  %-16s del %.3f  ins %.3f"%(m,s[f"del_{m}"]["mean"],s[f"ins_{m}"]["mean"]),
                       "" if m=="random" else "pars %.2f cov %.3f cons %.3f cont %.3f t %.3f"%(s[f"pars_{m}"],s[f"cov_{m}"],s[f"cons_{m}"],s[f"cont_{m}"],s[f"time_{m}"]))
    for k,v in s.items():
        if k.startswith("wil_"): print("  ",k,{a:(round(b,4) if isinstance(b,float) else b) for a,b in v.items()})
