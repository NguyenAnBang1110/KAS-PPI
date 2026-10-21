"""G1: per-pair retraining ground truth for KAS-group. usage: python3 gt_retrain.py VIEW PART [NJ]
For EVERY coalition S of annotation groups (2^|G|): remove S's annotations from every protein (an annotation survives
if a group outside S supports it), recompute all 47,142 pair features, retrain RF100 (random_state=0, same as
kas_group.py) on each of the 10 repo folds and record p(interaction) on the 500 pairs per fold that kas_group.py
explained. Retrained Shapley per pair: phi*_g(i) = sum_S w(|S|) [p_S(i) - p_{S+g}(i)] over all S not containing g,
i.e. the same 'help' sign as KASg. Unlike del1 or leave-one-group-out ablation, this ground truth averages over all
coalitions, so it does not favour LOO by construction. Checkpoints after every coalition (resumable)."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, ".")
import groups as GR
from sklearn.ensemble import RandomForestClassifier
VIEW, PART = sys.argv[1], sys.argv[2]; NJ = int(sys.argv[3]) if len(sys.argv) > 3 else 1
G = GR.GROUPS[PART]; NG = len(G)
K = np.load("feats_kg.npz"); y = K["y"]; keys = [tuple(k) for k in K["keys"]]; aspects = list(K["aspects"])
ci = aspects.index("GO_0032991"); keep = [j for j in range(52) if j != ci]
if VIEW == "kg":
    from kgsim import KG
    X0 = K["Xpy"][:, keep]
    kg = KG("godata/go-basic.owl", "godata/goa_human.gaf", aspects)
    M = GR.load("godata/goa_human.gaf", PART, drop_not=True)
    def feats(a1, a2):
        v = kg.features(a1, a2); return [v[j] for j in keep]
else:
    from seekrep import SEEK
    X0 = np.load("feats_seek.npz")["Xseek51"]
    S_ = "seek/PPI/"
    sk = SEEK(S_ + "go.owl", S_ + "go_annotations.gaf", S_ + "Embeddings/Emb_classes_maxdepth4_nwalks100.txt")
    CX = sk.desc("GO_0032991") | {"GO_0032991"}
    M = GR.load(S_ + "go_annotations.gaf", PART, drop_not=False)
    M = {p: {t: s for t, s in d.items() if t not in CX} for p, d in M.items()}
    def feats(a1, a2): return list(sk.rep(a1, a2))
P = "ExplainablePPI/Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run"
folds = [[int(l) for l in open(P + ("1(complete)" if f == 1 else str(f)) + ".txt") if l.strip()] for f in range(1, 11)]
IDX = [[r["i"] for r in json.load(open(f"kasg_{VIEW}_{PART}_f{f}.json"))["records"]] for f in range(1, 11)]
PK = [np.array([r["p"] for r in json.load(open(f"kasg_{VIEW}_{PART}_f{f}.json"))["records"]]) for f in range(1, 11)]
prots = sorted({p for k in keys for p in k})
out = f"gt_{VIEW}_{PART}.npz"
NC = 1 << NG
Pr = np.full((NC, 10, len(IDX[0])), np.nan)
if os.path.exists(out): Pr = np.load(out)["P"]
t0 = time.time()
for mask in range(NC):
    if not np.isnan(Pr[mask]).any(): continue
    S = frozenset(g for g in range(NG) if mask >> g & 1)
    am = {p: GR.ann(M, p, S) for p in prots}
    X = np.array([feats(am[a], am[b]) for a, b in keys])
    if mask == 0: print("full-feature recon maxdiff", float(np.abs(X - X0).max()), flush=True)
    for f, te in enumerate(folds):
        m = np.zeros(len(y), bool); m[te] = True
        c = RandomForestClassifier(100, n_jobs=NJ, random_state=0).fit(X[~m], y[~m])
        Pr[mask, f] = c.predict_proba(X[IDX[f]])[:, 1]
    if mask == 0:
        print("p(full) vs kas_group p maxdiff", max(float(np.abs(Pr[0, f] - PK[f]).max()) for f in range(10)), flush=True)
    np.savez(out, P=Pr, groups=np.array(G), idx=np.array(IDX))
    print(VIEW, PART, "coalition", mask, sorted(G[g] for g in S), round(time.time() - t0), "s", flush=True)
print("done", round(time.time() - t0))
