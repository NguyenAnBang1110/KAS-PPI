"""K4: Sousa24 Figs. 3-5 (size and average IC of explanations, first partition), with the repo's own code
(ML.performance_DT + prune, LIMEexplanations, evaluation_explanations.process_models/process_rule, IC from ICSeco_GOterms.txt).
usage: python3 k_expl.py VARIANT NLIME     VARIANT in {base51, beta1, gamma001}
Models use the authors' settings (RF 10 trees as in sklearn 0.21.3; XGB tree_method exact). GP6x: the fold-1 models of
k_gp.py. LORE (base51 only): rules from the 09/10 F3 run (lore_sa on the KAS RF100/XGB300 models, fold 1)."""
import sys, os, json, glob, statistics, numpy as np
V = os.environ.get("KGSIM_WORKDIR", "work")   # thư mục làm việc: bản clone ExplainablePPI, feats_kg.npz, k10/
os.chdir(f"{V}/ExplainablePPI"); sys.path.insert(0, ".")
for _a, _t in (('int', int), ('float', float), ('bool', bool)):
    if not hasattr(np, _a): setattr(np, _a, _t)
import six, joblib, sklearn.externals
sklearn.externals.six = six; sys.modules['sklearn.externals.six'] = six
sklearn.externals.joblib = joblib; sys.modules['sklearn.externals.joblib'] = joblib
from Prediction import ML, LIMEexplanations
import evaluation_explanations as EV
from sklearn.tree import export_text as _et
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
ML.export_text = lambda clf, names: _et(clf, feature_names=list(names))
VAR, NL = sys.argv[1], int(sys.argv[2])
F = np.load(f"{V}/k10/feats_{VAR}.npz"); X = F["X"]; names = [str(n) for n in F["names"]]; urls = [str(u) for u in F["urls"]]
K = np.load(f"{V}/feats_kg.npz"); y = K["y"]; keys = [tuple(map(str, k)) for k in K["keys"]]
te = [int(l) for l in open("Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run1(complete).txt") if l.strip()]
m = np.ones(len(y), bool); m[te] = False; tr = np.where(m)[0]
dic_ics = EV.process_IC("Experimental_Results/Data/ICSeco_GOterms.txt")
n2u = {n: "http://purl.obolibrary.org/obo/" + u for n, u in zip(names, urls)}
def summ(v): return dict(n=len(v), median=round(statistics.median(v), 4), q1=round(float(np.percentile(v, 25)), 4), q3=round(float(np.percentile(v, 75)), 4))
out = {"variant": VAR, "n_aspects": len(names)}; od = f"{V}/k10/expl_{VAR}"; os.makedirs(od, exist_ok=True)
dict_SS = {keys[i]: list(X[i]) for i in te}
for alg, d in (("DT6", 6), ("DT", None)):
    p = f"{od}/{alg}_"
    ML.performance_DT(dict_SS, [keys[i] for i in te], list(X[tr]), list(X[te]), list(y[tr]), list(y[te]), names,
                      p + "pred.txt", p + "model.txt", p + "eval.txt", p + "expl.txt", d=d)
    EV.process_models(p + "expl.txt", p + "evalexpl.txt", dic_ics, n2u, names)
    rows = [l.split("\t") for l in open(p + "evalexpl.txt").read().strip().split("\n")[1:-1]]
    out[alg] = dict(size=summ([float(r[4]) for r in rows]), avgIC=summ([float(r[3]) for r in rows]))
    print(VAR, alg, out[alg], flush=True)
for f in sorted(glob.glob(f"{V}/k10/gp_{VAR}/GP6x_f1_s*_model.txt")):
    s, a, n = EV.process_rule(open(f).read(), names, dic_ics, n2u)
    out.setdefault("GP6x", []).append(dict(seed=f.split("_s")[-1].split("_")[0], size=n, avgIC=round(a, 4)))
print(VAR, "GP6x", out.get("GP6x"), flush=True)
rng = np.random.RandomState(0); sel = sorted(rng.choice(len(te), NL, replace=False))
for mname, clf in (("RF", RandomForestClassifier(10, random_state=0)), ("XGB", xgb.XGBClassifier(random_state=0, tree_method="exact"))):
    clf.fit(X[tr], y[tr]); s = {3: [], 8: []}
    for j in sel:
        for k in (3, 8):
            s[k].append(EV.process_rule(LIMEexplanations.LIME_explanation(list(X[tr]), list(X[te]), j, clf, names, k), names, dic_ics, n2u))
    for k in (3, 8):
        out[f"{mname}+LIME{k}"] = dict(size=summ([v[2] for v in s[k]]), avgIC=summ([v[1] for v in s[k]]))
        print(VAR, f"{mname}+LIME{k}", out[f"{mname}+LIME{k}"], flush=True)
if VAR == "base51":
    import ast
    u2n = {u: n for n, u in zip(names, urls)}
    for mname in ("rf", "xgb"):
        R = json.load(open(f"{V}/f09/lore_kg_{mname}_f1.json"))["records"]; sz, ic = [], []
        for r in R:
            rr = ast.literal_eval(r["rule"]) if isinstance(r["rule"], str) else r["rule"]; asp = sorted({c[0] for c in rr})
            if not asp: continue
            rule = " and ".join(u2n.get(a, a) for a in asp)
            _, a_, n_ = EV.process_rule(rule, names, dic_ics, n2u); sz.append(n_); ic.append(a_)
        out[f"{mname.upper()}+LORE(lore_sa,F3)"] = dict(size=summ(sz), avgIC=summ(ic))
        print(VAR, mname, "LORE", out[f"{mname.upper()}+LORE(lore_sa,F3)"], flush=True)
json.dump(out, open(f"{V}/k10/k4_expl_{VAR}.json", "w"), indent=1)
