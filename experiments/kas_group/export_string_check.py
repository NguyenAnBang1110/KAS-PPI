"""Export everything the Colab STRING-channel check needs (no GO/GAF/repo files needed on Colab).
For all 47,142 pairs: accessions, label, repo fold, KG-view and SEEK-view features under
  full | noR (every annotation whose only support is Reactome removed) | ctlR (size-matched random control, same RNG as group_ablation.py)
plus the KAS-group Reactome Shapley value of the 10,000 explained pairs (source partition, both views)."""
import sys, json, glob, random
import numpy as np
sys.path.insert(0, ".")
from kgsim import KG
from seekrep import SEEK
import groups as GR
K = np.load("feats_kg.npz"); y = K["y"]; keys = [tuple(k) for k in K["keys"]]; aspects = list(K["aspects"])
ci = aspects.index("GO_0032991"); keep = [j for j in range(52) if j != ci]
P = "ExplainablePPI/Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run"
fold = np.zeros(len(y), int)
for f in range(1, 11):
    for l in open(P + ("1(complete)" if f == 1 else str(f)) + ".txt"):
        if l.strip(): fold[int(l)] = f
assert (fold > 0).all()
prots = sorted({p for k in keys for p in k})
def variants(M):
    full = {p: GR.ann(M, p) for p in prots}; noR = {p: GR.ann(M, p, frozenset([0])) for p in prots}
    rng = random.Random(0); ctl = {}
    for p in prots:
        nrem = len(full[p]) - len(noR[p]); cand = sorted(noR[p])
        ctl[p] = full[p] - set(rng.sample(cand, min(nrem, len(cand))))
    return full, noR, ctl
out = dict(p1=np.array([a for a, b in keys]), p2=np.array([b for a, b in keys]), y=y, fold=fold)
kg = KG("godata/go-basic.owl", "godata/goa_human.gaf", aspects)
M = GR.load("godata/goa_human.gaf", "source", drop_not=True)
for name, am in zip(["full", "noR", "ctlR"], variants(M)):
    X = np.array([[v[j] for j in keep] for v in (kg.features(am[a], am[b]) for a, b in keys)], dtype=np.float32)
    out["kg_" + name] = X; print("kg", name, flush=True)
assert np.allclose(out["kg_full"], K["Xpy"][:, keep].astype(np.float32))
S_ = "seek/PPI/"
sk = SEEK(S_ + "go.owl", S_ + "go_annotations.gaf", S_ + "Embeddings/Emb_classes_maxdepth4_nwalks100.txt")
CX = sk.desc("GO_0032991") | {"GO_0032991"}
M = GR.load(S_ + "go_annotations.gaf", "source", drop_not=False)
M = {p: {t: s for t, s in d.items() if t not in CX} for p, d in M.items()}
for name, am in zip(["full", "noR", "ctlR"], variants(M)):
    out["seek_" + name] = np.array([sk.rep(am[a], am[b]) for a, b in keys], dtype=np.float32); print("seek", name, flush=True)
assert np.allclose(out["seek_full"], np.load("feats_seek.npz")["Xseek51"].astype(np.float32), atol=1e-5)
for view in ["kg", "seek"]:
    R = [r for f in glob.glob(f"kasg_{view}_source_f*.json") for r in json.load(open(f))["records"]]
    out[f"kas_{view}_idx"] = np.array([r["i"] for r in R]); out[f"kas_{view}_phiReactome"] = np.array([r["KASg"][0] for r in R])
np.savez_compressed("string_check_input.npz", **out)
print("saved")
