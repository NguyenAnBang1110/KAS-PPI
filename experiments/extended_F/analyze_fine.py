"""F4 summary. usage: python3 analyze_fine.py MODEL"""
import sys, json
import numpy as np
from scipy.stats import wilcoxon
MODEL = sys.argv[1]; d = json.load(open(f"f09/fine_kg_{MODEL}_f1.json")); R = d["records"]
print(f"== KAS with finer players (depth-2 GO subtrees + residual unit per aspect; {d['n_units']} units), KGsim2vec view, {MODEL.upper()}, fold 1, {len(R)} pairs")
print(f"  players per pair: aspect level {np.mean([r['k_aspect'] for r in R]):.1f}, fine level {np.mean([r['k_fine'] for r in R]):.1f} (median {np.median([r['k_fine'] for r in R]):.0f}, "
      f"exact Shapley for {np.mean([r['k_fine'] <= 8 for r in R]):.2f} of pairs)")
print(f"  time per pair: fine {np.mean([r['time_fine'] for r in R]):.2f} s, aspect {np.mean([r['time_aspect'] for r in R]):.2f} s")
print(f"  efficiency: max |sum fine phi - sum aspect phi| = {max(abs(r['sum_fine'] - r['sum_aspect']) for r in R):.1e}")
rho = [r["rho_agg_vs_aspect"] for r in R if r["rho_agg_vs_aspect"] is not None]
t1 = [r["top1_fine_in_top1_aspect"] for r in R if r["top1_fine_in_top1_aspect"] is not None]
print(f"  fine phi summed per aspect vs aspect-level phi: Spearman {np.mean(rho):.3f} (n={len(rho)}); top-1 fine unit inside the top-1 aspect {np.mean(t1):.3f}")
auc = lambda c: float(np.mean(c)) if c else np.nan
rows = {m: (np.array([auc(r[f"del_{m}"]) for r in R]), np.array([auc(r[f"ins_{m}"]) for r in R])) for m in ("Dfine", "LOOfine", "randomfine")}
out = {}
print("  method       deletion AUC (down)  insertion AUC (up)   [operator ks, k = 1..5 fine units]")
for m, (de, ins) in rows.items():
    out[m] = (float(np.nanmean(de)), float(np.nanmean(ins))); print(f"  {m:12s} {np.nanmean(de):.3f}                {np.nanmean(ins):.3f}")
for m in ("LOOfine", "randomfine"):
    a, b = rows["Dfine"][0], rows[m][0]; ok = ~np.isnan(a) & ~np.isnan(b)
    a2, b2 = rows["Dfine"][1], rows[m][1]; ok2 = ~np.isnan(a2) & ~np.isnan(b2)
    pd_, pi_ = wilcoxon(a[ok], b[ok]).pvalue, wilcoxon(a2[ok2], b2[ok2]).pvalue
    out[f"p_{m}"] = (float(pd_), float(np.mean(a[ok] < b[ok])), float(pi_), float(np.mean(a2[ok2] > b2[ok2])))
    print(f"  KAS-fine vs {m}: deletion p={pd_:.2g} (KAS lower in {np.mean(a[ok] < b[ok]):.2f}), insertion p={pi_:.2g} (KAS higher in {np.mean(a2[ok2] > b2[ok2]):.2f})")
from collections import Counter
print("  most frequent top-1 fine units:", Counter(r["top1_fine_label"] for r in R if r["top1_fine_label"]).most_common(8))
json.dump(out, open(f"f09/fine_{MODEL}_summary.json", "w"))
