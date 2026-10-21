"""Analysis of KAS-group vs retraining ground truth. Primary global score (fixed before looking at results):
  help_g = mean over explained pairs of phi_g signed toward the TRUE label (how much group g pushes the model to the
  correct answer); compared with the retraining F1 drop of removing g (and with the excess over the size-matched control).
Secondary: mean |phi_g|. Same scores for LOOg, TSg (kg) and the plain annotation count."""
import json, glob, sys
import numpy as np
from scipy.stats import spearmanr, wilcoxon
out = {}; lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)
pooled = {}
for view in ["kg", "seek"]:
    for part in ["source", "evidence"]:
        fs = sorted(glob.glob(f"kasg_{view}_{part}_f*.json"))
        ab = f"gabl_{view}_{part}.json"
        if not fs: continue
        R = [r for f in fs for r in json.load(open(f))["records"]]
        G = json.load(open(fs[0]))["groups"]
        try: A = json.load(open(ab))
        except FileNotFoundError: A = None
        meths = ["KASg", "LOOg"] + (["TSg"] if "TSg" in R[0] else [])
        sy = np.array([1 if r["y"] == 1 else -1 for r in R])
        res = {"n_pairs": len(R), "folds": len(fs), "groups": G}
        P(f"\n== {view} | {part} | {len(R)} pairs from {len(fs)} folds ==")
        cnt = np.array([r["count"] for r in R]).mean(0)
        hdr = f"{'group':12s} {'count':>7s}" + "".join(f" {m+'_help':>11s} {m+'_abs':>10s}" for m in meths)
        if A: hdr += f" {'dropF1':>8s} {'ctl':>8s} {'excess':>8s}"
        P(hdr)
        sc = {}
        for m in meths:
            Phi = np.array([r[m] for r in R])
            sc[m + "_help"] = (Phi * sy[:, None]).mean(0); sc[m + "_abs"] = np.abs(Phi).mean(0)
        sc["count"] = cnt
        drop = exc = None
        if A:
            drop = np.array([-A[g]["delta"] for g in G]); ctl = np.array([-A[g + "_ctl"]["delta"] for g in G])
            exc = np.array([A[g]["excess"] for g in G]) * -1  # positive = drops more than control
        for gi, g in enumerate(G):
            s = f"{g:12s} {cnt[gi]:7.2f}" + "".join(f" {sc[m+'_help'][gi]:11.4f} {sc[m+'_abs'][gi]:10.4f}" for m in meths)
            if A: s += f" {drop[gi]*100:8.2f} {ctl[gi]*100:8.2f} {exc[gi]*100:8.2f}"
            P(s)
        res["scores"] = {k: v.tolist() for k, v in sc.items()}
        if A:
            res["drop"] = drop.tolist(); res["excess"] = exc.tolist()
            P("Spearman across groups (score vs retraining dropF1 | vs excess):")
            for k, v in sc.items():
                r1 = spearmanr(v, drop).correlation; r2 = spearmanr(v, exc).correlation
                P(f"  {k:12s} {r1:+.2f} | {r2:+.2f}"); res[f"rho_{k}"] = [float(r1), float(r2)]
                pooled.setdefault(k, []).append((v, drop, exc))
            # per-annotation reliance: help per unit of annotation count, vs excess
            for m in meths:
                pa = sc[m + "_help"] / np.maximum(cnt, 1e-9)
                P(f"  {m}_help/count top groups: {[G[j] for j in np.argsort(-pa)[:3]]}  excess top: {[G[j] for j in np.argsort(-exc)[:3]]}")
        # deletion of the top-1 group (knowledge space, no retraining)
        P("del1 (prob drop of the predicted class when the top-1 group is removed):")
        rnd = np.array([r.get("del1_random", 0.0) for r in R])
        for m in meths:
            d = np.array([r[f"del1_{m}"] for r in R])
            P(f"  {m:5s} {d.mean():.4f}  vs random {rnd.mean():.4f}  p={wilcoxon(d, rnd).pvalue:.2g}")
            res[f"del1_{m}"] = float(d.mean())
        res["del1_random"] = float(rnd.mean())
        # label-conditional: mean phi toward class 1 for true positives vs true negatives (circularity probe)
        Phi = np.array([r["KASg"] for r in R]); yy = np.array([r["y"] for r in R])
        P("KASg mean phi (toward interaction) in positives | negatives:")
        for gi, g in enumerate(G):
            P(f"  {g:12s} {Phi[yy==1,gi].mean():+.4f} | {Phi[yy==0,gi].mean():+.4f}")
        res["phi_pos"] = Phi[yy == 1].mean(0).tolist(); res["phi_neg"] = Phi[yy == 0].mean(0).tolist()
        out[f"{view}_{part}"] = res
P("\n== pooled over view x partition (rank within each block, then Spearman on all points) ==")
for k, L in pooled.items():
    a = np.concatenate([np.argsort(np.argsort(v)) / (len(v) - 1) for v, _, _ in L])
    b = np.concatenate([np.argsort(np.argsort(d)) / (len(d) - 1) for _, d, _ in L])
    c = np.concatenate([np.argsort(np.argsort(e)) / (len(e) - 1) for _, _, e in L])
    P(f"  {k:12s} n={len(a)}  rho_drop {spearmanr(a,b).correlation:+.2f}  rho_excess {spearmanr(a,c).correlation:+.2f}")
json.dump(out, open("kasg_summary.json", "w"), indent=1)
open("kasg_summary.txt", "w").write("\n".join(lines))
