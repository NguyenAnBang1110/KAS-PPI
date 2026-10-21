"""G1 extra: for each group, correlation ACROSS pairs between the method's phi_g and the retrained phi*_g
(pairs where g is a player); and the retrained-Shapley agreement split by redundancy (LOO share of total effect)."""
import sys, json, math, numpy as np
from scipy.stats import spearmanr, wilcoxon
VIEW, PART = sys.argv[1], sys.argv[2]
Z = np.load(f"gt_{VIEW}_{PART}.npz"); Pr = Z["P"]; G = list(Z["groups"]); NG = len(G); NC = 1 << NG
W = [math.factorial(r) * math.factorial(NG - r - 1) / math.factorial(NG) for r in range(NG)]
pc = [bin(s).count("1") for s in range(NC)]
def gts(p):
    return np.array([sum(W[pc[s]] * (p[s] - p[s | 1 << g]) for s in range(NC) if not s >> g & 1) for g in range(NG)])
rows = []
for f in range(10):
    R = json.load(open(f"kasg_{VIEW}_{PART}_f{f+1}.json"))["records"]
    for n, r in enumerate(R): rows.append((r, gts(Pr[:, f, n])))
meths = ["KASg", "LOOg"] + (["TSg"] if "TSg" in rows[0][0] else [])
out = {"across_pairs": {}, "by_redundancy": {}}
print(f"== {VIEW} {PART}: Spearman across pairs, phi_g vs phi*_g (pairs where g is a player)")
for gi, g in enumerate(G):
    sel = [(r, ph) for r, ph in rows if g in r["players"]]
    gt = np.array([ph[gi] for _, ph in sel])
    s = {m: float(spearmanr(np.array([r[m][gi] for r, _ in sel]), gt).correlation) for m in meths}
    out["across_pairs"][g] = dict(n=len(sel), **s); print(f"  {g:11s} n={len(sel):5d} " + " ".join(f"{m} {v:+.3f}" for m, v in s.items()))
# redundancy: LOO share = sum|LOOg| / sum|KASg| (KAS sums to the total effect)
print("per-pair Spearman with phi* by LOO share of total effect (pairs >=3 players, |total|>0.1)")
bins = [(0, .25), (.25, .5), (.5, 1e9)]
for lo, hi in bins:
    a = {m: [] for m in meths}
    for r, ph in rows:
        pl = [G.index(g) for g in r["players"]]
        tot = abs(sum(r["KASg"]))
        if len(pl) < 3 or tot <= 0.1: continue
        sh = np.abs(np.array(r["LOOg"])).sum() / max(np.abs(np.array(r["KASg"])).sum(), 1e-12)
        if not lo <= sh < hi: continue
        sg = 1 if r["pred"] == 1 else -1; gt = sg * ph[pl]
        for m in meths:
            x = sg * np.array(r[m])[pl]
            a[m].append(spearmanr(x, gt).correlation if np.ptp(x) > 0 and np.ptp(gt) > 0 else 0.0)
    k = np.array(a["KASg"]); l = np.array(a["LOOg"])
    out["by_redundancy"][f"{lo}-{hi}"] = dict(n=len(k), **{m: float(np.mean(v)) for m, v in a.items()}, p=float(wilcoxon(k, l).pvalue) if len(k) > 10 else None)
    print(f"  LOO share [{lo},{hi}) n={len(k):5d} " + " ".join(f"{m} {np.mean(v):.3f}" for m, v in a.items()) + f"  p(KAS vs LOO)={out['by_redundancy'][f'{lo}-{hi}']['p']}")
json.dump(out, open(f"g10/gt_extra_{VIEW}_{PART}.json", "w"), indent=1)
