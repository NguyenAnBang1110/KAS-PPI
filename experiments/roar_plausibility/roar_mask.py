"""ROAR leakage diagnostic (cf. ROAD, Rong et al. 2022): how well can the label be predicted from WHICH aspects were removed
alone? Features = removed-aspect indicators (51) only; same repeated 5-fold CV as roar.py. usage: python3 roar_mask.py PIPE MODEL"""
import sys, glob, json, random
import numpy as np
sys.path.insert(0, ".")
from kgsim import KG
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
PIPE, MODEL = sys.argv[1], sys.argv[2]
K = np.load("feats_kg.npz"); keys = [tuple(k) for k in K["keys"]]; aspects = list(K["aspects"])
ci = aspects.index("GO_0032991"); keep = [j for j in range(52) if j != ci]; A = [aspects[j] for j in keep]
kg = KG("godata/go-basic.owl", "godata/goa_human.gaf", aspects); sub = [kg.sub[j] for j in keep]
recs = []
for f in sorted(glob.glob(f"rec_{PIPE}_{MODEL}_f*_*.json")): recs += json.load(open(f))["records"]
methods = [k[4:] for k in recs[0] if k.startswith("phi_")] + ["random"]
y = np.array([r["y"] for r in recs])
masks = {(m, k): np.zeros((len(recs), len(A))) for m in methods for k in (1, 2, 3)}
for n, r in enumerate(recs):
    p1, p2 = keys[r["i"]]
    if PIPE == "kg": pl = [i for i in range(len(A)) if kg.ann[p1] & sub[i] and kg.ann[p2] & sub[i]]
    else: pl = sorted({A.index(a) for m in ("D", "LOO") for a in r[f"phi_{m}"]})
    sgn = 1 if r["pred"] == 1 else -1; rr = random.Random(r["i"])
    for m in methods:
        if m == "random": order = rr.sample(pl, len(pl))
        else:
            phi = r[f"phi_{m}"]; tb = {i: rr.random() for i in pl}
            order = sorted(pl, key=lambda i: (-sgn * phi.get(A[i], 0.0), tb[i]))
        for k in (1, 2, 3): masks[(m, k)][n, order[:k]] = 1
splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0).split(y, y))
out = {}
for (m, k), M in masks.items():
    acc = [RandomForestClassifier(100, n_jobs=4, random_state=0).fit(M[tr], y[tr]).score(M[te], y[te]) for tr, te in splits]
    out[f"{m}|{k}"] = float(np.mean(acc)); print(m, k, "mask-only acc %.3f" % np.mean(acc), flush=True)
json.dump(out, open(f"roar_mask_{PIPE}_{MODEL}.json", "w"))
