"""K5: Sousa24 case studies (Tables 3-8). DT6 decision path (repo's prune_duplicate_leaves + processDTs) and LIME
discretisation bins, for the six pairs, with (a) the fold-1 model ('first partition') and (b) the model whose test fold
contains the pair. DT random_state is None in the repo; we try seeds 0-9 and report how many reproduce each paper rule."""
import sys, os, re, json, numpy as np
V = "/tmp/claude-0/-home-claude/ebe45e83-2762-5fc6-ac9d-6bee5cbe5bce/scratchpad/v2"
sys.path.insert(0, f"{V}/ExplainablePPI")
for _a, _t in (('int', int), ('float', float), ('bool', bool)):
    if not hasattr(np, _a): setattr(np, _a, _t)
import six, joblib, sklearn.externals
sklearn.externals.six = six; sys.modules['sklearn.externals.six'] = six
sklearn.externals.joblib = joblib; sys.modules['sklearn.externals.joblib'] = joblib
from Prediction import ML, processDTs
from sklearn.tree import DecisionTreeClassifier
F = np.load(f"{V}/k10/feats_base51.npz"); X = F["X"]; names = [str(n) for n in F["names"]]
K = np.load(f"{V}/feats_kg.npz"); y = K["y"]; keys = [tuple(map(str, k)) for k in K["keys"]]
idx = {k: i for i, k in enumerate(keys)}
P = f"{V}/ExplainablePPI/Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run"
folds = [[int(l) for l in open(P + ("1(complete)" if f == 1 else str(f)) + ".txt") if l.strip()] for f in range(1, 11)]
fold_of = {i: f + 1 for f, te in enumerate(folds) for i in te}
CASES = {"T3 RPS12-RPS10": ("P25398", "P46783", "metabolic_process > 0.5167 AND cellular_process > 0.8032"),
         "T4 S100A10-AHNAK": ("P60903", "Q09666", "binding <= 0.4693 AND cellular_anatomical_entity > 0.1162 AND cellular_process <= 0.0399 AND biological_regulation <= 0.5484"),
         "T5 PRR5L-GNL3L": ("Q6MZQ0", "Q9NVN8", "binding <= 0.4693 AND 0.0399 < cellular_process <= 0.6145 AND biological_regulation <= 0.5484"),
         "T6 TGFA-DLG2": ("P01135", "Q15700", "cellular_anatomical_entity > 0.7589 AND metabolic_process > 0.6023 AND (3rd condition cut in PDF text)"),
         "T7 PXN-ITGA4": ("P49023", "P13612", "metabolic_process <= 0.6048 AND cellular_process > 0.8535"),
         "T8 PSMC1-PSMB3": ("P62191", "P49720", "cellular_anatomical_entity > 0.7973 AND cellular_process > 0.7531")}
def thr(rule): return sorted({round(float(v), 4) for v in re.findall(r"(-?\d+\.\d+)", re.sub(r"\s+", " ", rule))})
def path(clf, i, a, b):
    rules = processDTs.tree_to_code(clf, names)
    return processDTs.get_decision_path(rules, names, {(a, b): list(X[i])}, a, b)
out = {"cases": {}, "lime_quartiles_fold1_train": {}}
models = {}
for which in ("fold1", "own_fold"):
    for name, (a, b, paper) in CASES.items():
        i = idx.get((a, b)); 
        if i is None: i = idx[(b, a)]; a, b = b, a
        f = 1 if which == "fold1" else fold_of[i]
        te = folds[f - 1]; m = np.ones(len(y), bool); m[te] = False
        res = []
        for s in range(10):
            key = (f, s)
            if key not in models:
                c = DecisionTreeClassifier(max_depth=6, random_state=s).fit(X[m], y[m]); ML.prune_duplicate_leaves(c); models[key] = c
            r = path(models[key], i, a, b); res.append(r)
        pt = thr(paper)
        hits = [s for s, r in enumerate(res) if pt and set(pt) <= set(thr(r))]
        out["cases"].setdefault(name, {"pair": [a, b], "label": int(y[i]), "test_fold": fold_of[i], "paper": paper})
        out["cases"][name][which] = {"model_fold": f, "pair_in_training_set": bool(m[i]), "rule_seed0": res[0],
                                     "seeds_reproducing_paper_thresholds": hits}
        print(which, name, "fold", f, "train" if m[i] else "test", "| seeds matching paper:", hits, "\n   ours:", res[0][:220], flush=True)
from sklearn.ensemble import RandomForestClassifier
from Prediction import LIMEexplanations
out["lime_rf10_own_fold"] = {}
for name, (a, b, paper) in CASES.items():
    i = idx.get((a, b), idx.get((b, a))); f = fold_of[i]; te = folds[f - 1]; m = np.ones(len(y), bool); m[te] = False
    rf = RandomForestClassifier(10, random_state=0).fit(X[m], y[m]); j = list(te).index(i)
    l3 = LIMEexplanations.LIME_explanation(list(X[m]), list(X[te]), j, rf, names, 3)
    l8 = LIMEexplanations.LIME_explanation(list(X[m]), list(X[te]), j, rf, names, 8)
    out["lime_rf10_own_fold"][name] = {"LIME3": l3, "LIME8": l8, "RF10_pred": int(rf.predict(X[i:i+1])[0])}
    print("LIME", name, "RF10 pred", int(rf.predict(X[i:i+1])[0]), "| LIME3", l3, "| LIME8", l8[:300], flush=True)
m = np.ones(len(y), bool); m[folds[0]] = False
for n in ("cellular_process", "metabolic_process", "biological_regulation", "cellular_anatomical_entity"):
    j = names.index(n); out["lime_quartiles_fold1_train"][n] = [round(float(q), 2) for q in np.percentile(X[m][:, j], [25, 50, 75])]
print("LIME quartile bins on fold-1 training data:", out["lime_quartiles_fold1_train"])
json.dump(out, open(f"{V}/k10/k5_cases.json", "w"), indent=1)
