"""F2: sanity checks for the explainers (Adebayo et al. 2018, 'Sanity checks for saliency maps') + seed stability.
- model randomisation (data randomisation test): RF trained on PERMUTED labels; a faithful explainer must change a lot
  (low rank correlation with the explanation of the real model).
- seed stability: RF retrained with random_state 1..4 (same data); explanations of an equally accurate model should
  stay similar (high rank correlation). This replaces the 'Consistency = 1 by construction' cells of the VXAI table.
Same 300 pairs (150 pos + 150 neg) of fold 1 as the main run; reference explanation = the recorded one (RF seed 0).
usage: python3 sanity.py VIEW   VIEW in {kg, seek}   -> f09/sanity_{VIEW}.json"""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
VIEW = sys.argv[1]
sys.argv = ["x", VIEW, "rf", "1", "1"]
exec(open("explain_v2.py").read().split("pos=[i for i in test")[0])
from seeknat import native
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
R = json.load(open(f"rec_{VIEW}_rf_f1_400.json"))["records"]; R = R[:150] + R[200:350]
if VIEW == "seek":
    SX = {r["i"]: r for r in json.load(open("rec_seekx_rf_f1.json"))["records"]}
names = list(METHODS) + (["SEEKN", "SEEKS"] if VIEW == "seek" else [])
def phis(st):
    out = {nm: f(st, 0) for nm, f in METHODS.items()}
    if VIEW == "seek":
        pN, pS, *_ = native(st, sk, sub_sk, proba, len(A)); out["SEEKN"], out["SEEKS"] = pN, pS
    return out
def ref_phi(r, nm):
    src = SX[r["i"]] if nm in ("SEEKN", "SEEKS") else r
    return np.array([src[f"phi_{nm}"].get(a, 0.0) for a in A])
Ytr = y[~m].copy()
models = {f"seed{s}": (s, Ytr) for s in (1, 2, 3, 4)}
models["randlabel"] = (0, np.random.default_rng(0).permutation(Ytr))
res = {}; t0 = time.time()
for mname, (seed, ytr) in models.items():
    clf = RandomForestClassifier(100, n_jobs=1, random_state=seed).fit(X[~m], ytr)
    tree = shap.TreeExplainer(clf, data=bg, feature_perturbation="interventional")
    waf = float(f1_score(y[m], clf.predict(X[m]), average="weighted"))
    rows = []
    for r in R:
        p1, p2 = keys[r["i"]]
        st = (set(kg.ann[p1]), set(kg.ann[p2]), set(sk.ann[p1]) - CX, set(sk.ann[p2]) - CX)
        pl = players_of(st); pred = int(proba(feats(st))[0] > 0.5)
        ph = phis(st); row = dict(i=r["i"], k=len(pl), pred_same=int(pred == r["pred"]))
        for nm in names:
            a, b = ref_phi(r, nm), ph[nm]
            oa, ob = ranking(a, r["pred"]), ranking(b, pred)
            row[f"top1_{nm}"] = int(bool(oa) and bool(ob) and oa[0] == ob[0])
            row[f"jac3_{nm}"] = jacc(oa, ob)
            if len(pl) >= 3:
                va, vb = a[pl], b[pl]
                row[f"rho_{nm}"] = float(spearmanr(va, vb).correlation) if np.std(va) > 0 and np.std(vb) > 0 else None
        rows.append(row)
    res[mname] = dict(waf=waf, rows=rows)
    print(VIEW, mname, "WAF %.3f" % waf, round(time.time() - t0), "s", flush=True)
    json.dump(dict(view=VIEW, methods=names, base_waf=None, res=res), open(f"f09/sanity_{VIEW}.json", "w"))
print("FINISHED")
