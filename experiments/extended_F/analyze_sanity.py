"""Summarise sanity.py (F2) + RF-vs-XGB agreement on the same pairs.
usage: python3 analyze_sanity.py VIEW
Per method: mean Spearman rho (players, k>=3), top-1 agreement and top-3 Jaccard between the reference explanation
(RF seed 0) and (a) RF with another seed [want HIGH], (b) RF trained on permuted labels [want LOW]."""
import sys, json
import numpy as np
VIEW = sys.argv[1]
d = json.load(open(f"f09/sanity_{VIEW}.json")); ms = d["methods"]; out = {}
def agg(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None and not np.isnan(r[key])]; return (float(np.mean(v)) if v else float("nan")), len(v)
print(f"== sanity checks, {VIEW} view, RF, fold 1, 300 pairs")
for mn, r in d["res"].items(): print(f"  model {mn}: WAF {r['waf']:.3f}, same predicted class as reference {np.mean([x['pred_same'] for x in r['rows']]):.3f}")
print("  method   | seeds 1-4: rho / top1 / jac3   | permuted labels: rho / top1 / jac3 | drop in rho")
for m in ms:
    seeds = [x for s in ("seed1", "seed2", "seed3", "seed4") if s in d["res"] for x in d["res"][s]["rows"]]
    rnd = d["res"].get("randlabel", {"rows": []})["rows"]
    a = [agg(seeds, f"{k}_{m}")[0] for k in ("rho", "top1", "jac3")]; b = [agg(rnd, f"{k}_{m}")[0] for k in ("rho", "top1", "jac3")]
    out[m] = dict(seed=a, rand=b)
    print(f"  {m:8s} | {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f}          | {b[0]:.3f} / {b[1]:.3f} / {b[2]:.3f}              | {a[0]-b[0]:.3f}")
# RF vs XGB on the same 2,200 pairs (implementation invariance across model families; same data, similar accuracy)
import glob
from scipy.stats import spearmanr
src = "kg" if VIEW == "kg" else "seek"
rf = {r["i"]: r for f in glob.glob(f"rec_{src}_rf_f*_*.json") for r in json.load(open(f))["records"]}
xg = {r["i"]: r for f in glob.glob(f"rec_{src}_xgb_f*_*.json") for r in json.load(open(f))["records"]}
common = sorted(set(rf) & set(xg)); A = json.load(open(glob.glob(f"rec_{src}_rf_f1_*.json")[0]))["aspects"]
print(f"  RF vs XGB, same {len(common)} pairs (both models predict the same class in "
      f"{np.mean([rf[i]['pred'] == xg[i]['pred'] for i in common]):.3f}):")
out["rf_vs_xgb"] = {}
for m in [k[4:] for k in rf[common[0]] if k.startswith("phi_")]:
    rh, t1 = [], []
    for i in common:
        a, b = rf[i][f"phi_{m}"], xg[i].get(f"phi_{m}")
        if b is None: continue
        keys_ = sorted(set(a) | set(b))
        if len(keys_) >= 3:
            va = [a.get(k, 0) for k in keys_]; vb = [b.get(k, 0) for k in keys_]
            if np.std(va) > 0 and np.std(vb) > 0:
                c = spearmanr(va, vb).correlation
                if not np.isnan(c): rh.append(c)
        if rf[i]["pred"] == xg[i]["pred"]:
            ta, tb = rf[i].get(f"top3_{m}") or [], xg[i].get(f"top3_{m}") or []
            t1.append(int(bool(ta) and bool(tb) and ta[0] == tb[0]))
    out["rf_vs_xgb"][m] = (float(np.mean(rh)), float(np.mean(t1)))
    print(f"    {m:8s} rho {np.mean(rh):.3f}  top-1 same {np.mean(t1):.3f}")
json.dump(out, open(f"f09/sanity_{VIEW}_summary.json", "w"))
