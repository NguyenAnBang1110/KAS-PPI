"""KAS-group: knowledge-space Shapley with annotation groups (curation source or evidence family) as players.
usage: python3 kas_group.py VIEW PART FOLD NPAIRS      VIEW in {kg,seek}, PART in {source,evidence}
Same model, fold and value function as explain_v2.py (RF100 trained on the other 9 repo partitions; remove the
players' annotations from both proteins and recompute the pair features through the pipeline), only the players change.
Baselines on the same pairs:
  LOOg  leave-one-group-out in knowledge space
  TSg   (kg only) TreeSHAP aspect attributions redistributed to groups by each group's share of the pair's
        annotations inside that aspect (what one can do with feature-space attributions alone)"""
import sys, json, time, itertools, math
import numpy as np
sys.path.insert(0, ".")
from kgsim import KG
from seekrep import SEEK
import groups as GR
from sklearn.ensemble import RandomForestClassifier
import shap
VIEW, PART, FOLD, NP = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
G = GR.GROUPS[PART]; NG = len(G)
K = np.load("feats_kg.npz"); y = K["y"]; keys = [tuple(k) for k in K["keys"]]; aspects = list(K["aspects"])
ci = aspects.index("GO_0032991"); keep = [j for j in range(52) if j != ci]; A = [aspects[j] for j in keep]
if VIEW == "kg":
    X = K["Xpy"][:, keep]
    kg = KG("godata/go-basic.owl", "godata/goa_human.gaf", aspects); sub = [kg.sub[j] for j in keep]
    M = GR.load("godata/goa_human.gaf", PART, drop_not=True)
    def feats(a1, a2):
        v = kg.features(a1, a2); return [v[j] for j in keep]
else:
    X = np.load("feats_seek.npz")["Xseek51"]
    S_ = "seek/PPI/"
    sk = SEEK(S_ + "go.owl", S_ + "go_annotations.gaf", S_ + "Embeddings/Emb_classes_maxdepth4_nwalks100.txt")
    CX = sk.desc("GO_0032991") | {"GO_0032991"}
    M = GR.load(S_ + "go_annotations.gaf", PART, drop_not=False)
    M = {p: {t: s for t, s in d.items() if t not in CX} for p, d in M.items()}
    def feats(a1, a2): return list(sk.rep(a1, a2))
P = "ExplainablePPI/Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run"
test = [int(l) for l in open(P + ("1(complete)" if FOLD == 1 else str(FOLD)) + ".txt") if l.strip()]
m = np.zeros(len(y), bool); m[test] = True
clf = RandomForestClassifier(100, n_jobs=1, random_state=0).fit(X[~m], y[~m])
def proba(Z): return clf.predict_proba(np.atleast_2d(np.asarray(Z, dtype=np.float64)))[:, 1]
if VIEW == "kg":
    bg = X[~m][np.random.default_rng(FOLD).choice((~m).sum(), 100, replace=False)]
    tree = shap.TreeExplainer(clf, data=bg, feature_perturbation="interventional")
rng = np.random.default_rng(100 + FOLD)
pos = [i for i in test if y[i] == 1]; neg = [i for i in test if y[i] == 0]
idx = list(rng.choice(pos, NP // 2, replace=False)) + list(rng.choice(neg, NP // 2, replace=False))
rec = []; t0 = time.time(); nchk = 0; maxdiff = 0.0
for n, i in enumerate(idx):
    p1, p2 = keys[i]
    def val(S):
        S = frozenset(S); return feats(GR.ann(M, p1, S), GR.ann(M, p2, S))
    x0 = np.array(val(())); maxdiff = max(maxdiff, float(np.abs(x0 - X[i]).max()))
    pl = sorted({g for p in (p1, p2) for s in M.get(p, {}).values() for g in s})
    subsets = [frozenset(c) for r in range(len(pl) + 1) for c in itertools.combinations(pl, r)]
    v = dict(zip(subsets, proba([val(S) for S in subsets])))
    k = len(pl); phi = np.zeros(NG)
    for g in pl:
        others = [q for q in pl if q != g]
        for r in range(k):
            w = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
            for c in itertools.combinations(others, r):
                S = frozenset(c); phi[g] += w * (v[S] - v[S | {g}])
    full = frozenset(); loo = np.zeros(NG)
    for g in pl: loo[g] = v[full] - v[frozenset([g])]
    p = float(v[full]); pred = int(p > 0.5)
    r = dict(i=int(i), y=int(y[i]), p=p, pred=pred, players=[G[g] for g in pl],
             KASg=phi.round(6).tolist(), LOOg=loo.round(6).tolist(),
             v_all_removed=float(v[frozenset(pl)]))
    # per-group annotation counts for this pair (an annotation counted once per supporting group, split evenly)
    cnt = np.zeros(NG)
    for pp in (p1, p2):
        for t, s in M.get(pp, {}).items():
            for g in s: cnt[g] += 1 / len(s)
    r["count"] = cnt.round(4).tolist()
    if VIEW == "kg":
        sv = np.array(tree.shap_values(x0[None], check_additivity=False))
        sv = sv[0, :, 1] if sv.ndim == 3 else sv[0]
        tsg = np.zeros(NG)
        for j in range(len(A)):
            if sv[j] == 0: continue
            sh = np.zeros(NG)
            for pp in (p1, p2):
                for t, s in M.get(pp, {}).items():
                    if t in sub[j]:
                        for g in s: sh[g] += 1 / len(s)
            if sh.sum() > 0: tsg += sv[j] * sh / sh.sum()
        r["TSg"] = tsg.round(6).tolist()
    # deletion check in knowledge space: remove the top-1 group of each method (signed toward the prediction)
    sgn = 1 if pred == 1 else -1
    for name in ["KASg", "LOOg"] + (["TSg"] if VIEW == "kg" else []):
        s = sgn * np.array(r[name]) if name != "TSg" else sgn * np.array(r[name])
        cand = [g for g in pl if s[g] > 0]
        if cand:
            top = max(cand, key=lambda g: s[g]); q = v[frozenset([top])]
            r[f"del1_{name}"] = float((p - q) * sgn)
        else:
            r[f"del1_{name}"] = 0.0
    if pl:
        rg = int(rng.choice(pl)); r["del1_random"] = float((p - v[frozenset([rg])]) * sgn)
    rec.append(r)
    if n % 50 == 0: print(VIEW, PART, FOLD, n, round(time.time() - t0, 1), "s k=", k, "maxdiff", maxdiff, flush=True)
json.dump(dict(view=VIEW, part=PART, fold=FOLD, groups=G, maxdiff_recon=maxdiff, records=rec),
          open(f"kasg_{VIEW}_{PART}_f{FOLD}.json", "w"))
print("done", time.time() - t0, "maxdiff", maxdiff)
