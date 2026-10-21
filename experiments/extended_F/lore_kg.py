"""F3: LORE (Guidotti et al. 2018), the second post-hoc explainer used by the KGsim2vec paper, on the KGsim2vec view.
Uses the LORE authors' library lore_sa (TabularGeneticGeneratorLore: genetic neighbourhood of 1,000 instances + decision-tree
surrogate). Same pairs as the main runs (first N records of rec_kg_{MODEL}_f{FOLD}_*.json).
LORE gives a factual rule + DT feature importances (unsigned). Aspect score phi_LORE = DT importance, signed toward the
predicted class (so ranking() orders by importance). Fidelity is measured with KAS's operator here ('ks');
the neutral operators (zero, imputation) are applied to all methods in road_impute.py.
usage: python3 lore_kg.py MODEL FOLD N   -> f09/lore_kg_{MODEL}_f{FOLD}.json"""
import sys, json, glob, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
MODEL, FOLD, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sys.argv = ["x", "kg", MODEL, str(FOLD), "1"]
exec(open("explain_v2.py").read().split("pos=[i for i in test")[0])
import pandas as pd
from lore_sa.dataset import TabularDataset
from lore_sa.bbox import sklearnBBox
from lore_sa.lore import TabularGeneticGeneratorLore
sel = rng.choice((~m).sum(), 2000, replace=False)
df = pd.DataFrame(X[~m][sel], columns=A); df["y"] = np.where(y[~m][sel] == 1, "1", "0")
class BB(sklearnBBox):
    def __init__(s, c): s.bbox = c
    def predict(s, Z): return np.where(clf.predict(np.asarray(Z, float)) == 1, "1", "0")
    def predict_proba(s, Z): return clf.predict_proba(np.asarray(Z, float))
lo = TabularGeneticGeneratorLore(BB(clf), TabularDataset(df, class_name="y"))
recs = []
for fn in sorted(glob.glob(f"rec_{'kg'}_{MODEL}_f{FOLD}_*.json")): recs += json.load(open(fn))["records"]
h = len(recs) // 2; recs = recs[:N // 2] + recs[h:h + N // 2]; out = []; t0 = time.time()   # records: positives first, then negatives
for n, r in enumerate(recs):
    p1, p2 = keys[r["i"]]; st = (set(kg.ann[p1]), set(kg.ann[p2]), set(), set()); x = feats(st)
    pred = r["pred"]; pl = players_of(st)
    np.random.seed(r["i"]); ts = time.time()
    try: e = lo.explain(x, num_instances=1000)
    except Exception as ex: print("ERR", r["i"], ex, flush=True); continue
    tl = time.time() - ts
    imp = dict(e["feature_importances"] or [])
    phi = np.array([float(imp.get(a, 0.0)) for a in A]) * (1 if pred == 1 else -1)
    prem = [(str(p["attr"]), p["op"], float(p["val"])) for p in e["rule"]["premises"]]
    order = ranking(phi, pred)
    o = dict(i=r["i"], y=r["y"], pred=pred, k=len(pl), time_LORE=tl, fid_LORE=float(e["fidelity"]),
             rule=prem, rule_len=len(prem), rule_aspects=sorted({p[0] for p in prem}),
             rule_players=len({p[0] for p in prem} & {A[j] for j in pl}), n_cf=len(e["counterfactuals"]),
             phi_LORE={A[j]: round(float(phi[j]), 5) for j in np.nonzero(phi)[0]}, cov_LORE=int(len(order) > 0),
             size_LORE=eff_size(phi))
    o["del_LORE"] = deletion_ks(st, order, pred) if order else None
    o["ins_LORE"] = insertion_ks(st, order, pred, pl) if order else None
    out.append(o)
    if n % 10 == 0: print(MODEL, FOLD, n, round(time.time() - t0), "s", flush=True)
json.dump(dict(model=MODEL, fold=FOLD, aspects=A, records=out), open(f"f09/lore_kg_{MODEL}_f{FOLD}.json", "w"))
print("done", MODEL, FOLD, round(time.time() - t0))
