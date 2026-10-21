"""K2/K3: 10-fold median WAF of DT, DT6, RF, XGB on a feature variant, with the model settings the authors actually used.
Authors' pickled fold-1 models (Experimental_Results/Results/{RF,XGB}/Model__Run1.txt, read with pickletools):
  RF  = sklearn 0.21.3 RandomForestClassifier() -> n_estimators = 10 (old default; now 100)
  XGB = XGBClassifier() with n_estimators 100, eta 0.3, max_depth 6, tree_method 'exact' (xgboost now defaults to 'hist')
usage: python3 k_models.py FEATS_NPZ OUT_JSON [SEEDS]"""
import sys, json, numpy as np
from statistics import median
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import xgboost as xgb
V = "/tmp/claude-0/-home-claude/ebe45e83-2762-5fc6-ac9d-6bee5cbe5bce/scratchpad/v2"
FE, OUT = sys.argv[1], sys.argv[2]; SEEDS = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "0,1,2,3,4").split(",")]
X = np.load(FE)["X"]; y = np.load(f"{V}/feats_kg.npz")["y"]
P = f"{V}/ExplainablePPI/Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run"
folds = [[int(l) for l in open(P + ("1(complete)" if f == 1 else str(f)) + ".txt") if l.strip()] for f in range(1, 11)]
M = {"DT": lambda s: DecisionTreeClassifier(random_state=s),
     "DT6": lambda s: DecisionTreeClassifier(max_depth=6, random_state=s),
     "RF100": lambda s: RandomForestClassifier(100, random_state=s, n_jobs=2),
     "RF10_authors": lambda s: RandomForestClassifier(10, random_state=s, n_jobs=2),
     "XGB_hist": lambda s: xgb.XGBClassifier(random_state=s, n_jobs=2),
     "XGB_exact_authors": lambda s: xgb.XGBClassifier(random_state=s, n_jobs=2, tree_method="exact")}
res = {"n_features": int(X.shape[1])}
for name, mk in M.items():
    meds = []; allw = []
    for s in SEEDS:
        w = []
        for te in folds:
            m = np.ones(len(y), bool); m[te] = False
            w.append(f1_score(y[te], mk(s).fit(X[m], y[m]).predict(X[te]), average="weighted"))
        q1, q3 = np.percentile(w, [25, 75]); meds.append(median(w)); allw.append([round(v, 4) for v in w])
    res[name] = dict(median_per_seed=[round(v, 4) for v in meds], mean_median=round(float(np.mean(meds)), 4),
                     IQR_seed0=round(float(np.subtract(*np.percentile(allw[0], [75, 25]))), 4), folds=allw)
    print(FE.split("/")[-1], name, res[name]["median_per_seed"], "IQR", res[name]["IQR_seed0"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
