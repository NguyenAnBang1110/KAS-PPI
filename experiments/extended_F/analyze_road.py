"""Summarise road_impute.py output. usage: python3 analyze_road.py MODEL SRC [SUBSET_METHOD]
AUC = mean predicted-class probability over k=1..5 removed aspects (MoRF: lower = better; LeRF: higher = better);
gap = LeRF - MoRF (higher = better); flip@1 = share of pairs whose class flips after removing 1 aspect (MoRF).
Pairs restricted to those explained by every method shown (SUBSET_METHOD, e.g. LORE, restricts further).
Wilcoxon signed-rank, two-sided (paired over pairs), KAS(D) vs each method; direction = folds where KAS is better."""
import sys, json
import numpy as np
from scipy.stats import wilcoxon
MODEL, SRC = sys.argv[1], sys.argv[2]; SUBM = sys.argv[3] if len(sys.argv) > 3 else None
Z = np.load(f"f09/road_{MODEL}_{SRC}.npz"); meta = json.load(open(f"f09/road_{MODEL}_{SRC}.json"))
methods = meta["methods"]
mask = np.ones(len(Z["ids"]), bool)
for m in methods:
    if SUBM is None and m == "LORE": continue
    mask &= Z[f"has|{m}"]
show = [m for m in methods if SUBM is not None or m != "LORE"]
fold = Z["fold"][mask]
print(f"== ROAD-style imputation, KGsim2vec view, {MODEL.upper()}, {mask.sum()} pairs; discriminator AUC " + " ".join(f"{k} {v:.3f}" for k, v in meta['disc_auc'].items()) + "")
out = {}
for op in [o for o in ("imp", "knn", "zero") if f"D|MoRF|{o}" in Z.files]:
    print(f"-- operator {op}")
    print("  method    | MoRF AUC (down) | LeRF AUC (up) | gap (up) | MoRF p@k1 k2 k3 | flip@1")
    stats = {}
    for m in show + ["random"]:
        mo = Z[f"{m}|MoRF|{op}"][mask]; mo = np.where(np.isnan(mo), np.nan, mo)
        mor = np.nanmean(mo, 1)
        if m != "random":
            le = Z[f"{m}|LeRF|{op}"][mask]; ler = np.nanmean(le, 1); gap = ler - mor
        else: ler = gap = np.full_like(mor, np.nan)
        stats[m] = (mor, ler, gap)
        fl = np.mean(mo[:, 0] < 0.5)
        print(f"  {m:9s} | {np.nanmean(mor):.3f}           | {np.nanmean(ler):.3f}         | {np.nanmean(gap):.3f}    | "
              f"{np.nanmean(mo[:,0]):.3f} {np.nanmean(mo[:,1]):.3f} {np.nanmean(mo[:,2]):.3f} | {fl:.3f}")
    for m in show:
        if m == "D": continue
        res = {}
        for j, (nm, alt) in enumerate((("MoRF", "less"), ("LeRF", "greater"), ("gap", "greater"))):
            a, b = stats["D"][j], stats[m][j]; ok = ~np.isnan(a) & ~np.isnan(b)
            p = wilcoxon(a[ok], b[ok]).pvalue if np.any(a[ok] != b[ok]) else 1.0   # two-sided; direction from the fold count
            fw = sum((np.nanmean(a[fold == f]) < np.nanmean(b[fold == f])) if nm == "MoRF" else (np.nanmean(a[fold == f]) > np.nanmean(b[fold == f]))
                     for f in range(1, 11) if np.any(fold == f))
            res[nm] = (float(p), int(fw))
        out[f"{op}|{m}"] = res
        print(f"  KAS vs {m:9s}: " + "  ".join(f"{k} p={v[0]:.2g} folds KAS better {v[1]}/10" for k, v in res.items()))
    out[op] = {m: [float(np.nanmean(s)) for s in stats[m]] for m in stats}
json.dump(dict(n=int(mask.sum()), disc=meta["disc_auc"], res=out), open(f"f09/road_{MODEL}_{SRC}{'_'+SUBM if SUBM else ''}_summary.json", "w"))
