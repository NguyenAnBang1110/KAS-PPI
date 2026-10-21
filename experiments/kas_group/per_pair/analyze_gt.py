"""G1 analysis: per-pair agreement of KAS-group (no retraining) with the retrained-Shapley ground truth.
usage: python3 analyze_gt.py VIEW PART [PREF]   (PREF = kasg_ for RF explanations; gt files are always RF100)
Rule fixed BEFORE looking at the results:
  primary metric  = per-pair Spearman between attribution and phi* over the pair's player groups (pairs with >= 3
                    players), both signed toward the predicted class.
  KAS-group passes per pair iff mean rho(KASg) > mean rho(LOOg), two-sided Wilcoxon over pairs p < 0.05,
                    and KASg better in >= 8/10 folds (fold means).
Secondary: top-1 agreement (argmax signed attribution == argmax signed phi*); Pearson; the same against the
retrained leave-one-group-out (rLOO) to show what LOO-style ground truth would say; efficiency of phi*."""
import sys, json, math, itertools
import numpy as np
from scipy.stats import spearmanr, pearsonr, wilcoxon
VIEW, PART = sys.argv[1], sys.argv[2]; PREF = sys.argv[3] if len(sys.argv) > 3 else "kasg_"
Z = np.load(f"gt_{VIEW}_{PART}.npz"); Pr = Z["P"]; G = list(Z["groups"]); NG = len(G); NC = 1 << NG
assert not np.isnan(Pr).any(), "ground truth incomplete"
W = [math.factorial(r) * math.factorial(NG - r - 1) / math.factorial(NG) for r in range(NG)]
pc = [bin(s).count("1") for s in range(NC)]
def gt_shapley(p):  # p: (NC,) predictions under each removed-coalition; returns help-signed phi* over all groups
    phi = np.zeros(NG)
    for g in range(NG):
        b = 1 << g
        for s in range(NC):
            if s & b: continue
            phi[g] += W[pc[s]] * (p[s] - p[s | b])
    return phi
rng = np.random.default_rng(0)
meths = None; per = {}; fold_mean = {}; eff = []
for f in range(10):
    R = json.load(open(f"{PREF}{VIEW}_{PART}_f{f+1}.json"))["records"]
    if meths is None:
        meths = ["KASg", "LOOg"] + (["TSg"] if "TSg" in R[0] else []) + ["count", "random"]
        for m in meths:
            for k in ("rho", "top1", "pear", "rho_rloo"): per[(m, k)] = []
    fm = {m: [] for m in meths}
    for n, r in enumerate(R):
        p = Pr[:, f, n]; phi = gt_shapley(p); rloo = p[0] - np.array([p[1 << g] for g in range(NG)])
        eff.append(abs(phi.sum() - (p[0] - p[NC - 1])))
        pl = [G.index(g) for g in r["players"]]
        if len(pl) < 3: continue
        sgn = 1 if r["pred"] == 1 else -1
        gt = sgn * phi[pl]; gl = sgn * rloo[pl]
        for m in meths:
            if m == "count": a = np.array(r["count"])[pl]
            elif m == "random": a = rng.standard_normal(len(pl))
            else: a = sgn * np.array(r[m])[pl]
            rho = spearmanr(a, gt).correlation if np.ptp(a) > 0 and np.ptp(gt) > 0 else 0.0
            rl = spearmanr(a, gl).correlation if np.ptp(a) > 0 and np.ptp(gl) > 0 else 0.0
            pe = pearsonr(a, gt)[0] if np.ptp(a) > 0 and np.ptp(gt) > 0 else 0.0
            per[(m, "rho")].append(rho); per[(m, "rho_rloo")].append(rl); per[(m, "pear")].append(pe)
            per[(m, "top1")].append(float(np.argmax(a) == np.argmax(gt))); fm[m].append(rho)
    for m in meths: fold_mean.setdefault(m, []).append(np.mean(fm[m]))
res = dict(view=VIEW, part=PART, pref=PREF, groups=G, n_pairs=len(per[("KASg", "rho")]),
           phi_eff_maxerr=float(max(eff)))
print(f"== G1 {VIEW} | {PART} | explanations {PREF} | {res['n_pairs']} pairs with >=3 players | phi* efficiency max err {max(eff):.1e}")
print(f"{'method':8s} {'rho(phi*)':>10s} {'top1':>7s} {'pearson':>8s} {'rho(rLOO)':>10s}")
for m in meths:
    row = {k: float(np.nanmean(per[(m, k)])) for k in ("rho", "top1", "pear", "rho_rloo")}
    res[m] = row; res[m]["fold_rho"] = [float(x) for x in fold_mean[m]]
    print(f"{m:8s} {row['rho']:10.3f} {row['top1']:7.3f} {row['pear']:8.3f} {row['rho_rloo']:10.3f}")
a = np.array(per[("KASg", "rho")]); b = np.array(per[("LOOg", "rho")])
p = float(wilcoxon(a, b).pvalue); wins = int(sum(x > y for x, y in zip(fold_mean["KASg"], fold_mean["LOOg"])))
res["KAS_vs_LOO"] = dict(p=p, fold_wins=wins, diff=float(a.mean() - b.mean()))
res["verdict"] = "PASS" if (a.mean() > b.mean() and p < 0.05 and wins >= 8) else "FAIL"
print(f"KASg vs LOOg on rho(phi*): diff {a.mean()-b.mean():+.3f}, p={p:.2g}, KAS better in {wins}/10 folds -> {res['verdict']}")
for m in meths:
    if m in ("KASg", "LOOg"): continue
    c = np.array(per[(m, "rho")]); q = float(wilcoxon(a, c).pvalue)
    w = int(sum(x > y for x, y in zip(fold_mean["KASg"], fold_mean[m])))
    res[f"KAS_vs_{m}"] = dict(p=q, fold_wins=w, diff=float(a.mean() - c.mean()))
    print(f"KASg vs {m}: diff {a.mean()-c.mean():+.3f}, p={q:.2g}, KAS better in {w}/10 folds")
# global: mean phi* toward the true label per group (model-level reliance, all groups, all explained pairs)
Ph = []; Yv = []
for f in range(10):
    R = json.load(open(f"{PREF}{VIEW}_{PART}_f{f+1}.json"))["records"]
    for n, r in enumerate(R): Ph.append(gt_shapley(Pr[:, f, n])); Yv.append(1 if r["y"] == 1 else -1)
Ph = np.array(Ph); Yv = np.array(Yv)
res["gt_help"] = dict(zip(G, (Ph * Yv[:, None]).mean(0).round(4).tolist()))
print("phi* help toward true label, per group:", res["gt_help"])
json.dump(res, open(f"g10/gt_{VIEW}_{PART}_{PREF.strip('_').split('/')[-1]}_summary.json", "w"), indent=1)
