"""ROAR (RemOve And Retrain, Hooker et al. 2019) for the aspect explanations already stored in rec_{PIPE}_{MODEL}_f*.json.
usage: python3 roar.py PIPE MODEL      PIPE in {kg,seek,hyb}, MODEL in {rf,xgb}
Pool = every explained pair of that configuration (each pair was explained by the model of the fold where it is a TEST pair,
so explanations never come from a model that saw the pair). For each method and k in KS, each pair loses its own top-k
players; a fresh model of the same class is retrained and evaluated on the modified pool with repeated stratified 5-fold CV
(same splits for every method). A larger accuracy drop = the ranking removed more of the information the task needs.
Ranking: all players of the pair sorted by phi signed toward the original prediction (ties random), so every method removes
exactly min(k, #players) aspects. Removal operators:
  ks   = knowledge-space removal (delete both proteins' annotations under the aspect, recompute features)  [all pipes]
  zero = set the aspect's feature column to 0 (KGsim2vec only)"""
import sys, glob, json, time, random
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "ExplainablePPI")
from kgsim import KG
from seekrep import SEEK
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

PIPE, MODEL = sys.argv[1], sys.argv[2]
KS = [1, 2, 3]; REPEATS = 3

K = np.load("feats_kg.npz"); keys = [tuple(k) for k in K["keys"]]; aspects = list(K["aspects"]); y_all = K["y"]
ci = aspects.index("GO_0032991"); keep = [j for j in range(52) if j != ci]; A = [aspects[j] for j in keep]
kg = KG("godata/go-basic.owl", "godata/goa_human.gaf", aspects); sub_kg = [kg.sub[j] for j in keep]
S_ = "seek/PPI/"
sk = SEEK(S_ + "go.owl", S_ + "go_annotations.gaf", S_ + "Embeddings/Emb_classes_maxdepth4_nwalks100.txt") if PIPE != "kg" else None
CX = (sk.desc("GO_0032991") | {"GO_0032991"}) if sk else set()
sub_sk = [sk.desc(a) | {a} for a in A] if sk else [set() for _ in A]

def feats(st):
    k1, k2, s1, s2 = st; parts = []
    if PIPE in ("kg", "hyb"):
        v = kg.features(k1, k2); parts.append([v[j] for j in keep])
    if PIPE in ("seek", "hyb"): parts.append(list(sk.rep(s1, s2)))
    return np.concatenate(parts)
def players_of(st):
    k1, k2, s1, s2 = st; out = []
    for i in range(len(A)):
        a = (PIPE != "seek") and bool(k1 & sub_kg[i]) and bool(k2 & sub_kg[i])
        b = (PIPE != "kg") and bool(s1 & sub_sk[i]) and bool(s2 & sub_sk[i])
        if a or b: out.append(i)
    return out
def remove(st, S):
    if not S: return st
    rk = set().union(*[sub_kg[i] for i in S]); rs = set().union(*[sub_sk[i] for i in S])
    k1, k2, s1, s2 = st; return (k1 - rk, k2 - rk, s1 - rs, s2 - rs)

recs = []
for f in sorted(glob.glob(f"rec_{PIPE}_{MODEL}_f*_*.json")):
    recs += json.load(open(f))["records"]
methods = [k[4:] for k in recs[0] if k.startswith("phi_")] + ["random"]
print(PIPE, MODEL, "pairs", len(recs), "methods", methods, flush=True)
aidx = {a: j for j, a in enumerate(A)}

rows = {m: {"ks": {k: [] for k in KS}, "zero": {k: [] for k in KS}} for m in methods}
base = []; y = []; npl = []
t0 = time.time()
for n, r in enumerate(recs):
    p1, p2 = keys[r["i"]]
    st = (set(kg.ann[p1]), set(kg.ann[p2]),
          (set(sk.ann[p1]) - CX) if sk else set(), (set(sk.ann[p2]) - CX) if sk else set())
    x0 = feats(st); base.append(x0); y.append(r["y"])
    pl = players_of(st); npl.append(len(pl))
    sgn = 1 if r["pred"] == 1 else -1
    rr = random.Random(r["i"])
    for m in methods:
        if m == "random":
            order = rr.sample(pl, len(pl))
        else:
            phi = r[f"phi_{m}"]; tieb = {i: rr.random() for i in pl}
            order = sorted(pl, key=lambda i: (-sgn * phi.get(A[i], 0.0), tieb[i]))
        for k in KS:
            rows[m]["ks"][k].append(feats(remove(st, order[:k])))
            if PIPE == "kg":
                z = x0.copy(); z[order[:k]] = 0; rows[m]["zero"][k].append(z)
    if n % 500 == 0: print(n, round(time.time() - t0, 1), "s", flush=True)
y = np.array(y); base = np.array(base)

def make():
    return (RandomForestClassifier(100, n_jobs=4, random_state=0) if MODEL == "rf"
            else XGBClassifier(n_estimators=300, n_jobs=4, random_state=0, eval_metric="logloss"))
splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=0).split(base, y))
def cv(X):
    acc, f1 = [], []
    for tr, te in splits:
        c = make().fit(X[tr], y[tr]); p = c.predict(X[te])
        acc.append(accuracy_score(y[te], p)); f1.append(f1_score(y[te], p))
    return acc, f1
out = dict(pipe=PIPE, model=MODEL, n_pairs=len(y), median_players=float(np.median(npl)), ks=KS,
           n_splits=len(splits), methods=methods, base={}, res={})
a, f = cv(base); out["base"] = dict(acc=a, f1=f)
print("base acc %.4f f1 %.4f" % (np.mean(a), np.mean(f)), flush=True)
for op in (["ks", "zero"] if PIPE == "kg" else ["ks"]):
    for m in methods:
        for k in KS:
            a, f = cv(np.array(rows[m][op][k]))
            out["res"][f"{op}|{m}|{k}"] = dict(acc=a, f1=f)
            print(op, m, k, "acc %.4f f1 %.4f" % (np.mean(a), np.mean(f)), round(time.time() - t0), "s", flush=True)
json.dump(out, open(f"roar_{PIPE}_{MODEL}.json", "w"))
print("done", time.time() - t0)
