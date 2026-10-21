"""Full-data ROAR for KGsim2vec: every one of the 47,142 pairs was explained (explain_all.py) by the fold model that did not
train on it. Each pair loses its own top-k players (knowledge-space removal, recompute), then the model is retrained and
evaluated on the repo's 10 partitions (train on 9, test on the held-out partition), exactly like the prediction table.
Also the mask-only leakage check (RF on removed-aspect indicators). usage: python3 roar_full.py MODEL"""
import sys, glob, json, random, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "ExplainablePPI")
from kgsim import KG
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import wilcoxon
MODEL = sys.argv[1]; KS = [1, 2, 3]
K = np.load("feats_kg.npz"); keys = [tuple(k) for k in K["keys"]]; aspects = list(K["aspects"]); y = K["y"]
ci = aspects.index("GO_0032991"); keep = [j for j in range(52) if j != ci]; A = [aspects[j] for j in keep]
kg = KG("godata/go-basic.owl", "godata/goa_human.gaf", aspects); sub = [kg.sub[j] for j in keep]
X0 = K["Xpy"][:, keep]
rec = {}
for f in sorted(glob.glob(f"all_kg_{MODEL}_f*.json")):
    for r in json.load(open(f))["records"]: rec[r["i"]] = r
assert len(rec) == len(y), (len(rec), len(y))
methods = [k[4:] for k in rec[0] if k.startswith("phi_")] + ["random"]
def feats(t1, t2): v = kg.features(t1, t2); return [v[j] for j in keep]
Xs = {(m, k): np.zeros_like(X0) for m in methods for k in KS}; Ms = {(m, k): np.zeros_like(X0) for m in methods for k in KS}
t0 = time.time()
for i in range(len(y)):
    r = rec[i]; p1, p2 = keys[i]; t1, t2 = kg.ann[p1], kg.ann[p2]
    pl = [a for a in range(len(A)) if t1 & sub[a] and t2 & sub[a]]
    sgn = 1 if r["pred"] == 1 else -1; rr = random.Random(i)
    for m in methods:
        if m == "random": order = rr.sample(pl, len(pl))
        else:
            phi = r[f"phi_{m}"]; tb = {a: rr.random() for a in pl}
            order = sorted(pl, key=lambda a: (-sgn * phi.get(A[a], 0.0), tb[a]))
        for k in KS:
            S = order[:k]
            if S:
                U = set().union(*[sub[a] for a in S]); Xs[(m, k)][i] = feats(t1 - U, t2 - U); Ms[(m, k)][i, S] = 1
            else: Xs[(m, k)][i] = X0[i]
    if i % 5000 == 0: print(i, round(time.time() - t0), "s", flush=True)
P = "ExplainablePPI/Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run"
folds = [[int(l) for l in open(P + ("1(complete)" if f == 1 else str(f)) + ".txt") if l.strip()] for f in range(1, 11)]
def make(): return (RandomForestClassifier(100, n_jobs=4, random_state=0) if MODEL == "rf"
                    else XGBClassifier(n_estimators=300, n_jobs=4, random_state=0, eval_metric="logloss"))
def cv(X, mk=make):
    f1s = []
    for te in folds:
        m = np.zeros(len(y), bool); m[te] = True
        c = mk().fit(X[~m], y[~m]); f1s.append(f1_score(y[m], c.predict(X[m]), average="weighted"))
    return f1s
out = dict(model=MODEL, n=len(y), base=cv(X0), res={}, mask={})
print("base WAF %.4f" % np.median(out["base"]), flush=True)
for m in methods:
    for k in KS:
        out["res"][f"{m}|{k}"] = cv(Xs[(m, k)])
        out["mask"][f"{m}|{k}"] = float(np.mean(cv(Ms[(m, k)], lambda: RandomForestClassifier(100, n_jobs=4, random_state=0))))
        a = np.array(out["res"][f"{m}|{k}"])
        print(m, k, "WAF median %.4f drop %.4f mask-only %.3f" % (np.median(a), np.median(out["base"]) - np.median(a), out["mask"][f"{m}|{k}"]), round(time.time() - t0), "s", flush=True)
for k in KS:
    d = np.array(out["res"][f"D|{k}"])
    for m in methods:
        if m != "D":
            o = np.array(out["res"][f"{m}|{k}"]); out[f"p_D_less_{m}_{k}"] = float(wilcoxon(d, o, alternative="less").pvalue)
            print(f"k={k} p(D<{m}) = {out[f'p_D_less_{m}_{k}']:.3g}")
json.dump(out, open(f"roar_full_kg_{MODEL}.json", "w"))
