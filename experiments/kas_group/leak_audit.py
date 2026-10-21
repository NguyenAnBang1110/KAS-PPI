"""Leakage audit by annotation source. usage: python3 leak_audit.py VIEW   (kg | seek)
Variants recompute the 47,142 pair features with some annotations removed and re-run the 10-fold RF100 table (repo partitions):
 A  full (51 aspects, complex removed as in KAS-PPI)
 B  'cellular anatomical entity' (CAE) annotations assigned_by Reactome removed
 E  control for B: same number of randomly chosen non-Reactome CAE annotations removed (per protein)
 D  all CAE annotations removed (aspect ablation)
 C  every annotation assigned_by Reactome removed
 F  every annotation assigned_by IntAct removed
 G  C + F
 H  control for C: per protein, as many randomly chosen non-Reactome-only annotations removed as C removes
An annotation (protein, term) is kept if at least one GAF line from an allowed source supports it."""
import sys, random, collections, json, time
import numpy as np
sys.path.insert(0, ".")
from kgsim import KG
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from scipy.stats import wilcoxon
VIEW = sys.argv[1]
K = np.load("feats_kg.npz"); asp = list(K["aspects"]); y = K["y"]; keys = [tuple(k) for k in K["keys"]]
ci = asp.index("GO_0032991"); keep = [j for j in range(52) if j != ci]
kg = KG("godata/go-basic.owl", "godata/goa_human.gaf", asp)
CAE = kg.desc("GO_0110165") | {"GO_0110165"}
if VIEW == "kg":
    gaf = "godata/goa_human.gaf"
else:
    from seekrep import SEEK
    S_ = "seek/PPI/"; sk = SEEK(S_ + "go.owl", S_ + "go_annotations.gaf", S_ + "Embeddings/Emb_classes_maxdepth4_nwalks100.txt")
    CX = sk.desc("GO_0032991") | {"GO_0032991"}; CAE = sk.desc("GO_0110165") | {"GO_0110165"}
    gaf = S_ + "go_annotations.gaf"
src = collections.defaultdict(lambda: collections.defaultdict(set))   # prot -> term -> {assigned_by}
for l in open(gaf):
    if l.startswith("!"): continue
    f = l.rstrip("\n").split("\t")
    if len(f) < 15: continue
    if VIEW == "kg" and "NOT" in f[3]: continue
    src[f[1]]["GO_" + f[4].split(":")[1]].add(f[14])
def ann(p, drop_src=(), cae_only=False):
    out = set()
    for t, s in src.get(p, {}).items():
        if cae_only and t not in CAE: out.add(t); continue
        if s - set(drop_src): out.add(t)
    return out
rng = random.Random(0)
def ann_control(p):   # remove as many random non-Reactome-only CAE terms as Reactome-only CAE terms
    full = ann(p); b = ann(p, ("Reactome",), cae_only=True); n = len(full - b)
    cand = sorted(t for t in (full & CAE) & b)
    drop = set(rng.sample(cand, min(n, len(cand))))
    return full - drop
def ann_controlC(p):
    full = ann(p); c = ann(p, ("Reactome",)); n = len(full - c)
    cand = sorted(c)
    return full - set(rng.sample(cand, min(n, len(cand))))
def feats(a1, a2):
    if VIEW == "kg":
        v = kg.features(a1, a2); return [v[j] for j in keep]
    return list(sk.rep(a1 - CX, a2 - CX))
prots = sorted({p for k in keys for p in k})
variants = {
    "A_full": lambda p: ann(p),
    "B_CAE_noReactome": lambda p: ann(p, ("Reactome",), cae_only=True),
    "E_CAE_randomControl": ann_control,
    "D_CAE_removed": lambda p: ann(p) - CAE,
    "C_noReactome": lambda p: ann(p, ("Reactome",)),
    "F_noIntAct": lambda p: ann(p, ("IntAct",)),
    "G_noReactome_noIntAct": lambda p: ann(p, ("Reactome", "IntAct")),
    "H_randomControl_for_C": ann_controlC,
}
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None
if ONLY: variants = {k: v for k, v in variants.items() if k in ONLY or k == "A_full"}
P = "ExplainablePPI/Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run"
folds = [[int(l) for l in open(P + ("1(complete)" if f == 1 else str(f)) + ".txt") if l.strip()] for f in range(1, 11)]
out = {}; t0 = time.time()
for name, fn in variants.items():
    A_ = {p: fn(p) for p in prots}
    X = np.array([feats(A_[a], A_[b]) for a, b in keys])
    f1s = []
    for te in folds:
        m = np.zeros(len(y), bool); m[te] = True
        c = RandomForestClassifier(100, n_jobs=2, random_state=0).fit(X[~m], y[~m])
        f1s.append(f1_score(y[m], c.predict(X[m]), average="weighted"))
    nann = sum(len(v) for v in A_.values())
    out[name] = dict(f1=f1s, median=float(np.median(f1s)), n_annotations=nann)
    extra = ""
    if VIEW == "kg":
        jc = [asp[j] for j in keep].index("GO_0110165"); xc = X[:, jc]
        extra = " CAE-alone AUC %.3f" % (roc_auc_score(y, xc) if xc.std() > 0 else 0.5)
    print(f"{name:24s} WAF {np.median(f1s):.4f}  annotations {nann}{extra}  ({time.time()-t0:.0f}s)", flush=True)
base = np.array(out["A_full"]["f1"])
for name in out:
    if name == "A_full": continue
    d = np.array(out[name]["f1"]); out[name]["delta"] = float(np.median(d) - np.median(base))
    out[name]["p"] = float(wilcoxon(base, d).pvalue) if np.any(base != d) else 1.0
    print(f"  {name:24s} delta {out[name]['delta']:+.4f}  p {out[name]['p']:.3g}")
json.dump(out, open(f"../leak/leak_audit_{VIEW}" + ("_" + "_".join(ONLY) if ONLY else "") + ".json", "w"))
