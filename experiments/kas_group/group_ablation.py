"""Retraining ground truth for KAS-group. usage: python3 group_ablation.py VIEW PART
For every group g: remove g's annotations from every protein (an annotation survives if another group supports it),
recompute all 47,142 pair features, re-run the 10-fold RF100 table on the repo partitions.
Control for g: per protein, remove the same number of randomly chosen annotations among those g does not remove.
excess(g) = drop(g) - drop(control g)."""
import sys, json, time, random
import numpy as np
sys.path.insert(0, ".")
from kgsim import KG
import groups as GR
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon
VIEW, PART = sys.argv[1], sys.argv[2]; NJ = int(sys.argv[3]) if len(sys.argv) > 3 else 2
G = GR.GROUPS[PART]
K = np.load("feats_kg.npz"); y = K["y"]; keys = [tuple(k) for k in K["keys"]]; aspects = list(K["aspects"])
ci = aspects.index("GO_0032991"); keep = [j for j in range(52) if j != ci]
if VIEW == "kg":
    kg = KG("godata/go-basic.owl", "godata/goa_human.gaf", aspects)
    M = GR.load("godata/goa_human.gaf", PART, drop_not=True)
    def feats(a1, a2):
        v = kg.features(a1, a2); return [v[j] for j in keep]
else:
    from seekrep import SEEK
    S_ = "seek/PPI/"
    sk = SEEK(S_ + "go.owl", S_ + "go_annotations.gaf", S_ + "Embeddings/Emb_classes_maxdepth4_nwalks100.txt")
    CX = sk.desc("GO_0032991") | {"GO_0032991"}
    M = GR.load(S_ + "go_annotations.gaf", PART, drop_not=False)
    M = {p: {t: s for t, s in d.items() if t not in CX} for p, d in M.items()}
    def feats(a1, a2): return list(sk.rep(a1, a2))
P = "ExplainablePPI/Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run"
folds = [[int(l) for l in open(P + ("1(complete)" if f == 1 else str(f)) + ".txt") if l.strip()] for f in range(1, 11)]
prots = sorted({p for k in keys for p in k})
def run(annmap):
    X = np.array([feats(annmap[a], annmap[b]) for a, b in keys]); f1s = []
    for te in folds:
        m = np.zeros(len(y), bool); m[te] = True
        c = RandomForestClassifier(100, n_jobs=NJ, random_state=0).fit(X[~m], y[~m])
        f1s.append(f1_score(y[m], c.predict(X[m]), average="weighted"))
    return f1s
out = {}; t0 = time.time()
full = {p: GR.ann(M, p) for p in prots}
out["full"] = dict(f1=run(full), n=sum(map(len, full.values())))
print("full", np.median(out["full"]["f1"]), flush=True)
base = np.array(out["full"]["f1"])
for gi, g in enumerate(G):
    rm = {p: GR.ann(M, p, frozenset([gi])) for p in prots}
    rng = random.Random(gi)
    ctl = {}
    for p in prots:
        nrem = len(full[p]) - len(rm[p]); cand = sorted(rm[p])
        ctl[p] = full[p] - set(rng.sample(cand, min(nrem, len(cand))))
    for name, am in ((g, rm), (g + "_ctl", ctl)):
        f = run(am); d = np.array(f)
        out[name] = dict(f1=f, n=sum(map(len, am.values())), delta=float(np.median(d) - np.median(base)),
                         p=float(wilcoxon(base, d).pvalue) if np.any(base != d) else 1.0)
        print(f"{name:16s} removed {out['full']['n']-out[name]['n']:6d}  delta {out[name]['delta']:+.4f} p {out[name]['p']:.3g} ({time.time()-t0:.0f}s)", flush=True)
    out[g]["excess"] = out[g]["delta"] - out[g + "_ctl"]["delta"]
json.dump(out, open(f"gabl_{VIEW}_{PART}.json", "w"))
print("done")
