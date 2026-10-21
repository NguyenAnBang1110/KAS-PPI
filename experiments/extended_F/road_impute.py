"""F1: ROAD-style evaluation (Rong et al. 2022 idea: remove features by IMPUTATION, keep the trained model, no retraining)
on the KGsim2vec view. Removing aspects S of a pair = replace x_S by a draw from the Gaussian conditional
p(x_S | x_rest) fitted on the fold's training features (noisy linear imputation, clipped at 0), and ask the ORIGINAL
fold model. No retraining -> no class leakage through the mask; imputation keeps the input close to the data manifold
(checked below with a discriminator). This operator is neutral: it is neither KAS's (delete annotations + recompute)
nor FS0's (set to 0). On the 2,200-pair sets a third operator, knn donor imputation (copy the removed values from the
training pair nearest in the kept features), keeps every value a real observed one. Both directions: MoRF (remove most relevant first, lower curve = better) and LeRF (remove least
relevant first, higher curve = better), k = 1..5 aspects, plus the zero operator for reference.
usage: python3 road_impute.py MODEL SRC   SRC=all (47,142 pairs, all_kg_MODEL_f*.json) | rec (2,200 main-run pairs,
       plus LORE scores from f09/lore_kg_MODEL_f*.json for the pairs LORE explained)  -> f09/road_{MODEL}_{SRC}.npz/.json"""
import sys, glob, json, random, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "ExplainablePPI")
from kgsim import KG
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score
MODEL, SRC = sys.argv[1], sys.argv[2]; KMAX = 5; NJ = int(__import__("os").environ.get("NJOBS", "4"))
K_ = np.load("feats_kg.npz"); keys = [tuple(k) for k in K_["keys"]]; aspects = list(K_["aspects"]); y = K_["y"]
ci = aspects.index("GO_0032991"); keep = [j for j in range(52) if j != ci]; A = [aspects[j] for j in keep]
X0 = K_["Xpy"][:, keep]; nA = len(A)
kg = KG("godata/go-basic.owl", "godata/goa_human.gaf", aspects); sub = [kg.sub[j] for j in keep]
P = "ExplainablePPI/Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run"
folds = [[int(l) for l in open(P + ("1(complete)" if f == 1 else str(f)) + ".txt") if l.strip()] for f in range(1, 11)]
rec = {}
if SRC == "all":
    for f in sorted(glob.glob(f"all_kg_{MODEL}_f*.json")):
        for r in json.load(open(f))["records"]: rec[r["i"]] = r
else:
    for f in sorted(glob.glob(f"rec_kg_{MODEL}_f*_*.json")):
        for r in json.load(open(f))["records"]: rec[r["i"]] = {k: v for k, v in r.items() if k in ("i", "y", "pred") or k.startswith("phi_")}
    for f in sorted(glob.glob(f"f09/lore_kg_{MODEL}_f*.json")):
        for r in json.load(open(f))["records"]:
            if r["i"] in rec: rec[r["i"]]["phi_LORE"] = r["phi_LORE"]
methods = sorted({k[4:] for r in rec.values() for k in r if k.startswith("phi_")})
print(SRC, MODEL, len(rec), "pairs, methods", methods, flush=True)
orders = [(m, d) for m in methods for d in ("MoRF", "LeRF")] + [("random", "MoRF")]
OPS = ("imp", "zero", "knn") if SRC == "rec" else ("imp", "zero")   # knn donor imputation only on the 2,200-pair sets (cost)
ids = sorted(rec); pos = {i: n for n, i in enumerate(ids)}
PR = {(m, d, op): np.full((len(ids), KMAX), np.nan, np.float32) for m, d in orders for op in OPS}
PR0 = np.zeros(len(ids), np.float32); FOLD = np.zeros(len(ids), np.int8); PRED = np.zeros(len(ids), np.int8)
HAS = {m: np.zeros(len(ids), bool) for m in methods}; HAS["random"] = np.ones(len(ids), bool)
disc = {"imp": [], "zero": [], "knn": [], "orig": []}
t0 = time.time()
for fi, te in enumerate(folds):
    m_ = np.zeros(len(y), bool); m_[te] = True
    clf = (RandomForestClassifier(100, n_jobs=NJ, random_state=0) if MODEL == "rf"
           else XGBClassifier(n_estimators=300, n_jobs=NJ, random_state=0, eval_metric="logloss")).fit(X0[~m_], y[~m_])
    DON = X0[~m_][np.random.default_rng(fi).choice((~m_).sum(), 5000, replace=False)]   # donor pool for knn imputation
    mu = X0[~m_].mean(0); Sig = np.cov(X0[~m_], rowvar=False) + 1e-6 * np.eye(nA); Lam = np.linalg.inv(Sig)
    rows, slots = [], []
    for i in te:
        if i not in rec: continue
        r = rec[i]; n = pos[i]; FOLD[n] = fi + 1; x = X0[i]; t1, t2 = kg.ann[keys[i][0]], kg.ann[keys[i][1]]
        pl = [a for a in range(nA) if t1 & sub[a] and t2 & sub[a]]; pred = r["pred"]; PRED[n] = pred
        sgn = 1 if pred == 1 else -1; rr = random.Random(i); tb = {a: rr.random() for a in pl}
        rows.append(x); slots.append((n, None, None, None))
        cache = {}
        for m, d in orders:
            if m == "random": order = random.Random(i + 7).sample(pl, len(pl))
            else:
                if f"phi_{m}" not in r: continue
                HAS[m][n] = True; phi = r[f"phi_{m}"]; s = {a: sgn * phi.get(A[a], 0.0) for a in pl}
                order = sorted(pl, key=lambda a: (-s[a], tb[a])) if d == "MoRF" else sorted(pl, key=lambda a: (s[a], tb[a]))
            for k in range(1, KMAX + 1):
                S = tuple(sorted(order[:k]))
                if not S: z = {op: x for op in OPS}
                elif S in cache: z = cache[S]
                else:
                    Sl = list(S); Kl = [j for j in range(nA) if j not in S]
                    C = np.linalg.inv(Lam[np.ix_(Sl, Sl)]); mean = mu[Sl] - C @ Lam[np.ix_(Sl, Kl)] @ (x[Kl] - mu[Kl])
                    g = np.random.default_rng(i * 131 + hash(S) % 100003)
                    zi = x.copy(); zi[Sl] = np.clip(g.multivariate_normal(mean, C, method="cholesky"), 0, None)
                    zz = x.copy(); zz[Sl] = 0; z = cache[S] = {"imp": zi, "zero": zz}
                    if "knn" in OPS:   # copy x_S from the training pair nearest in the kept features (real, on-manifold values)
                        nn = int(np.argmin(((DON[:, Kl] - x[Kl]) ** 2).sum(1))); zk = x.copy(); zk[Sl] = DON[nn, Sl]; z["knn"] = zk
                    if m == "random" and k == 2 and len(disc["imp"]) < 30000:
                        for op in OPS: disc[op].append(z[op])
                        disc["orig"].append(x)
                for op in OPS: rows.append(z[op]); slots.append((n, (m, d, op), k - 1, None))
    p = clf.predict_proba(np.array(rows))[:, 1]
    for (n, key, kk, _), q in zip(slots, p):
        q = q if PRED[n] == 1 else 1 - q
        if key is None: PR0[n] = q
        else: PR[key][n, kk] = q
    print("fold", fi + 1, len(rows), "rows", round(time.time() - t0), "s", flush=True)
# discriminator: can a classifier tell perturbed inputs from real ones? (0.5 = indistinguishable)
dres = {}
for op in OPS:
    Z = np.vstack([np.array(disc["orig"]), np.array(disc[op])]); lab = np.r_[np.zeros(len(disc["orig"])), np.ones(len(disc[op]))]
    idx = np.random.default_rng(0).permutation(len(lab)); h = len(idx) // 2
    dc = RandomForestClassifier(100, n_jobs=NJ, random_state=0, min_samples_leaf=5).fit(Z[idx[:h]], lab[idx[:h]])
    dres[op] = float(roc_auc_score(lab[idx[h:]], dc.predict_proba(Z[idx[h:]])[:, 1]))
print("discriminator AUC", dres, flush=True)
np.savez_compressed(f"f09/road_{MODEL}_{SRC}.npz", ids=np.array(ids), fold=FOLD, pred=PRED, p0=PR0,
                    **{f"{m}|{d}|{op}": v for (m, d, op), v in PR.items()}, **{f"has|{m}": v for m, v in HAS.items()})
json.dump(dict(model=MODEL, src=SRC, n=len(ids), methods=methods, disc_auc=dres), open(f"f09/road_{MODEL}_{SRC}.json", "w"))
print("FINISHED", round(time.time() - t0))
