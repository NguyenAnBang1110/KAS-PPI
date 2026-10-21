"""E2/E3: full-data ROAR in BOTH directions (MoRF = remove the k most relevant aspects; LeRF = remove the k least relevant
players), on all 47,142 pairs, with retraining on the repo's 10 partitions, plus the mask-only leakage check for each.
A good ranking makes MoRF hurt a lot AND LeRF hurt little; reporting both (and their gap) is the usual answer to the
class-information leakage that makes single-direction ROAR unreliable (discussed by Rong et al. 2022, ROAD).
usage: python3 roar_sym.py PIPE MODEL   PIPE in {kg, seek}; reads all_{PIPE}_{MODEL}_f*.json -> roar_sym_{PIPE}_{MODEL}.json
Orders use the signed attribution toward the pair's predicted class; ties broken at random (seeded per pair)."""
import sys, glob, json, random, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "ExplainablePPI")
from kgsim import KG
from seekrep import SEEK
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon
PIPE, MODEL = sys.argv[1], sys.argv[2]; KS = [1, 2, 3]
K = np.load("feats_kg.npz"); keys = [tuple(k) for k in K["keys"]]; aspects = list(K["aspects"]); y = K["y"]
ci = aspects.index("GO_0032991"); keep = [j for j in range(52) if j != ci]; A = [aspects[j] for j in keep]
if PIPE == "kg":
    kg = KG("godata/go-basic.owl", "godata/goa_human.gaf", aspects); sub = [kg.sub[j] for j in keep]
    X0 = K["Xpy"][:, keep]; ann = lambda p: kg.ann[p]
    def feats(t1, t2): v = kg.features(t1, t2); return [v[j] for j in keep]
else:
    S_ = "seek/PPI/"
    sk = SEEK(S_ + "go.owl", S_ + "go_annotations.gaf", S_ + "Embeddings/Emb_classes_maxdepth4_nwalks100.txt")
    CX = sk.desc("GO_0032991") | {"GO_0032991"}; sub = [sk.desc(a) | {a} for a in A]
    X0 = np.load("feats_seek.npz")["Xseek51"]; ann = lambda p: set(sk.ann[p]) - CX
    def feats(t1, t2): return sk.rep(t1, t2)
rec = {}
for f in sorted(glob.glob(f"all_{PIPE}_{MODEL}_f*.json")):
    for r in json.load(open(f))["records"]: rec[r["i"]] = r
assert len(rec) == len(y), (len(rec), len(y))
methods = [k[4:] for k in rec[0] if k.startswith("phi_")]
variants = [(m, d) for m in methods for d in ("MoRF", "LeRF")] + [("random", "MoRF")]
Xs = {(m, d, k): np.zeros_like(X0) for m, d in variants for k in KS}
Ms = {(m, d, k): np.zeros((len(y), len(A)), np.int8) for m, d in variants for k in KS}
t0 = time.time(); cache = {}
for i in range(len(y)):
    r = rec[i]; p1, p2 = keys[i]; t1, t2 = ann(p1), ann(p2)
    pl = [a for a in range(len(A)) if t1 & sub[a] and t2 & sub[a]]
    sgn = 1 if r["pred"] == 1 else -1; rr = random.Random(i); tb = {a: rr.random() for a in pl}
    cache.clear()
    for m, d in variants:
        if m == "random": order = random.Random(i + 7).sample(pl, len(pl))
        else:
            phi = r[f"phi_{m}"]; s = {a: sgn * phi.get(A[a], 0.0) for a in pl}
            order = sorted(pl, key=lambda a: (-s[a], tb[a])) if d == "MoRF" else sorted(pl, key=lambda a: (s[a], tb[a]))
        for k in KS:
            S = tuple(sorted(order[:k]))
            if S:
                if S not in cache:
                    U = set().union(*[sub[a] for a in S]); cache[S] = feats(t1 - U, t2 - U)
                Xs[(m, d, k)][i] = cache[S]; Ms[(m, d, k)][i, list(S)] = 1
            else: Xs[(m, d, k)][i] = X0[i]
    if i % 5000 == 0: print(i, round(time.time() - t0), "s", flush=True)
P = "ExplainablePPI/Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run"
folds = [[int(l) for l in open(P + ("1(complete)" if f == 1 else str(f)) + ".txt") if l.strip()] for f in range(1, 11)]
NJ = int(__import__("os").environ.get("NJOBS", "4"))
def make(): return (RandomForestClassifier(100, n_jobs=NJ, random_state=0) if MODEL == "rf"
                    else XGBClassifier(n_estimators=300, n_jobs=NJ, random_state=0, eval_metric="logloss"))
def cv(X, mk=make):
    f1s = []
    for te in folds:
        m = np.zeros(len(y), bool); m[te] = True
        c = mk().fit(X[~m], y[~m]); f1s.append(f1_score(y[m], c.predict(X[m]), average="weighted"))
    return f1s
out = dict(pipe=PIPE, model=MODEL, n=len(y), base=cv(X0), res={}, mask={})
base = float(np.median(out["base"])); print("base WAF %.4f" % base, flush=True)
for m, d in variants:
    for k in KS:
        key = f"{m}|{d}|{k}"
        out["res"][key] = cv(Xs[(m, d, k)])
        out["mask"][key] = float(np.mean(cv(Ms[(m, d, k)], lambda: RandomForestClassifier(100, n_jobs=NJ, random_state=0))))
        print(key, "WAF %.4f drop %.4f mask-only %.3f" % (np.median(out["res"][key]), base - np.median(out["res"][key]), out["mask"][key]),
              round(time.time() - t0), "s", flush=True)
        json.dump(out, open(f"roar_sym_{PIPE}_{MODEL}.json", "w"))
# per-fold gap = WAF(LeRF) - WAF(MoRF): larger = ranking separates relevant from irrelevant aspects
out["gap"] = {}; out["p"] = {}
for k in KS:
    for m in methods:
        out["gap"][f"{m}|{k}"] = list(np.array(out["res"][f"{m}|LeRF|{k}"]) - np.array(out["res"][f"{m}|MoRF|{k}"]))
    for m in methods:
        if m == "D": continue
        for name, a, b in (("MoRF_less", out["res"][f"D|MoRF|{k}"], out["res"][f"{m}|MoRF|{k}"]),
                           ("LeRF_greater", out["res"][f"D|LeRF|{k}"], out["res"][f"{m}|LeRF|{k}"]),
                           ("gap_greater", out["gap"][f"D|{k}"], out["gap"][f"{m}|{k}"])):
            alt = "less" if name == "MoRF_less" else "greater"
            out["p"][f"D_{name}_{m}_{k}"] = float(wilcoxon(a, b, alternative=alt).pvalue) if np.any(np.array(a) != np.array(b)) else 1.0
json.dump(out, open(f"roar_sym_{PIPE}_{MODEL}.json", "w"))
print("FINISHED")
