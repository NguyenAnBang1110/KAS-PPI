"""STRING channel check for the Reactome-circularity hypothesis.
usage: python3 string_check.py INPUT_NPZ LINKS_GZ ALIASES_GZ VERSION_LABEL
Pre-registered (written 2026-10-06 before any STRING data was seen):
  Groups of positive pairs (matched in STRING):
    DB-dependent   : combined score recomputed WITHOUT the database channel < 0.950 (pair would not be a positive
                     without curated-database evidence)
    DB-independent : recomputed-without-database >= 0.950
  T1 rate of Reactome-touched pairs (features change when Reactome-only annotations are removed) in the two groups
  T2 (main) per fold, recall drop on each group when Reactome-only annotations are removed (noR) minus the drop under the
     size-matched random control (ctlR); effect = excess(DB-dependent) - excess(DB-independent); Wilcoxon over 10 folds
  T3 KAS-group Reactome Shapley value on explained positives: DB-dependent vs DB-independent (Mann-Whitney)
  Verdict: SUPPORTED if T2 effect > 0 with p < 0.05 AND T3 higher in DB-dependent with p < 0.05, in both views;
           NOT SUPPORTED if T2 effect <= 0 or p >= 0.05 in both views; otherwise MIXED.
The STRING combination rule (prior 0.041 removed from each channel, 1 - prod(1 - s), prior added back) is checked
against the file's own combined_score before it is used; if it does not reproduce the file, the script says so and
falls back to a formula-free grouping (database >= 0.900 and experiments < 0.400 vs experiments >= 0.700)."""
import sys, gzip, json, collections
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score
INP, LINKS, ALIASES, VER = sys.argv[1:5]
D = np.load(INP)
p1, p2, y, fold = D["p1"], D["p2"], D["y"], D["fold"]
acc = set(p1) | set(p2)
out = {"string_version": VER}
def P(*a): print(*a, flush=True)
# ---- 1. UniProt accession -> STRING ids
a2s = collections.defaultdict(set)
with gzip.open(ALIASES, "rt") as f:
    for l in f:
        if l.startswith("#"): continue
        s, al, src = l.rstrip("\n").split("\t")[:3]
        if "UniProt_AC" in src and al in acc: a2s[al].add(s)
out["acc_mapped"] = len(a2s); out["acc_total"] = len(acc)
P(f"accessions mapped to STRING: {len(a2s)}/{len(acc)}")
need = {}
for i, (a, b) in enumerate(zip(p1, p2)):
    for s in a2s.get(a, ()):
        for t in a2s.get(b, ()):
            need[(s, t)] = i; need[(t, s)] = i
# ---- 2. channel scores of the dataset pairs
CH = None; sc = {}
with gzip.open(LINKS, "rt") as f:
    hdr = f.readline().split()
    CH = hdr[2:]
    for l in f:
        x = l.split()
        k = (x[0], x[1])
        if k in need:
            i = need[k]; v = np.array(x[2:], float) / 1000
            if i not in sc or v[-1] > sc[i][-1]: sc[i] = v
P("channels:", CH)
ci = {c: j for j, c in enumerate(CH)}
cdb = ci["database"]; cex = ci.get("experiments", ci.get("experimental")); ccomb = ci["combined_score"]
chan = [j for c, j in ci.items() if c != "combined_score"]
pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
found = np.array([i in sc for i in range(len(y))])
S = np.zeros((len(y), len(CH))); S[found] = np.array([sc[i] for i in range(len(y)) if found[i]])
out["pos_found"] = float(found[pos].mean()); out["pos_ge950"] = float((S[pos, ccomb] >= 0.95).mean())
out["neg_found"] = float(found[neg].mean()); out["neg_ge950"] = float((S[neg, ccomb] >= 0.95).mean())
P(f"positives found in STRING {VER}: {found[pos].mean():.3f}; combined>=0.950: {(S[pos,ccomb]>=0.95).mean():.3f}")
P(f"negatives found: {found[neg].mean():.3f}; combined>=0.950: {(S[neg,ccomb]>=0.95).mean():.4f}")
P(f"positives with database>0: {(S[pos,cdb]>0).mean():.3f}; experiments>0: {(S[pos,cex]>0).mean():.3f}")
# ---- 3. combination rule check
PRIOR = 0.041
def combine(M, cols):
    s = np.clip((M[:, cols] - PRIOR) / (1 - PRIOR), 0, 1)
    c = 1 - np.prod(1 - s, axis=1)
    return c * (1 - PRIOR) + PRIOR
fp = found & (y == 1)
rec = combine(S[fp], chan); err = np.abs(rec - S[fp, ccomb])
out["formula_median_abs_err"] = float(np.median(err)); out["formula_p95_abs_err"] = float(np.percentile(err, 95))
P(f"combination rule vs file combined_score: median |err| {np.median(err):.4f}, 95th pct {np.percentile(err,95):.4f}")
use_formula = np.percentile(err, 95) < 0.02
if use_formula:
    nodb = combine(S, [j for j in chan if j != cdb])
    dbdep = fp & (nodb < 0.95); dbind = fp & (nodb >= 0.95); out["grouping"] = "formula: combined without database channel"
else:
    dbdep = fp & (S[:, cdb] >= 0.9) & (S[:, cex] < 0.4); dbind = fp & (S[:, cex] >= 0.7)
    out["grouping"] = "fallback: database>=0.9 & experiments<0.4  vs  experiments>=0.7"
out["n_dbdep"] = int(dbdep.sum()); out["n_dbind"] = int(dbind.sum())
P(f"grouping = {out['grouping']}: DB-dependent {dbdep.sum()}, DB-independent {dbind.sum()} of {fp.sum()} matched positives")
# ---- 4. tests per view
for view in ["kg", "seek"]:
    Xf, Xn, Xc = D[view + "_full"], D[view + "_noR"], D[view + "_ctlR"]
    touched = np.abs(Xf - Xn).max(1) > 1e-9
    t1 = [touched[dbdep].mean(), touched[dbind].mean()]
    tab = [[touched[dbdep].sum(), (~touched[dbdep]).sum()], [touched[dbind].sum(), (~touched[dbind]).sum()]]
    p_t1 = chi2_contingency(tab)[1]
    P(f"\n[{view}] T1 Reactome-touched: DB-dependent {t1[0]:.3f} vs DB-independent {t1[1]:.3f} (chi2 p={p_t1:.2g})")
    R = {k: {"dep": [], "ind": [], "f1": []} for k in ("full", "noR", "ctlR")}
    for f in range(1, 11):
        te = fold == f
        for k, X in (("full", Xf), ("noR", Xn), ("ctlR", Xc)):
            c = RandomForestClassifier(100, n_jobs=-1, random_state=0).fit(X[~te], y[~te])
            pr = c.predict(X[te]); idx = np.where(te)[0]
            pred = np.zeros(len(y), int); pred[idx] = pr
            R[k]["dep"].append(pred[te & dbdep].mean()); R[k]["ind"].append(pred[te & dbind].mean())
            R[k]["f1"].append(f1_score(y[te], pr, average="weighted"))
    A = {k: {g: np.array(v) for g, v in d.items()} for k, d in R.items()}
    exc_dep = (A["full"]["dep"] - A["noR"]["dep"]) - (A["full"]["dep"] - A["ctlR"]["dep"])
    exc_ind = (A["full"]["ind"] - A["noR"]["ind"]) - (A["full"]["ind"] - A["ctlR"]["ind"])
    eff = exc_dep - exc_ind
    p_t2 = float(wilcoxon(eff, alternative="greater").pvalue) if np.any(eff != 0) else 1.0
    P(f"[{view}] recall full: DB-dep {np.median(A['full']['dep']):.3f}  DB-ind {np.median(A['full']['ind']):.3f}")
    P(f"[{view}] recall drop noR: DB-dep {np.median(A['full']['dep']-A['noR']['dep'])*100:.2f} pt, DB-ind {np.median(A['full']['ind']-A['noR']['ind'])*100:.2f} pt")
    P(f"[{view}] recall drop ctlR: DB-dep {np.median(A['full']['dep']-A['ctlR']['dep'])*100:.2f} pt, DB-ind {np.median(A['full']['ind']-A['ctlR']['ind'])*100:.2f} pt")
    P(f"[{view}] T2 effect (excess DB-dep - excess DB-ind): median {np.median(eff)*100:+.2f} pt, folds>0 {int((eff>0).sum())}/10, p={p_t2:.3g}")
    P(f"[{view}] overall WAF full {np.median(A['full']['f1']):.4f} noR {np.median(A['noR']['f1']):.4f} ctlR {np.median(A['ctlR']['f1']):.4f}")
    ki, kp = D[f"kas_{view}_idx"], D[f"kas_{view}_phiReactome"]
    md, mi = kp[dbdep[ki]], kp[dbind[ki]]
    p_t3 = float(mannwhitneyu(md, mi, alternative="greater").pvalue) if len(md) and len(mi) else 1.0
    P(f"[{view}] T3 KAS Reactome phi on positives: DB-dep mean {md.mean():+.4f} (n={len(md)}) vs DB-ind {mi.mean():+.4f} (n={len(mi)}), p={p_t3:.3g}")
    out[view] = dict(T1=[float(v) for v in t1], T1_p=float(p_t1),
                     recall={k: {g: v.tolist() for g, v in d.items()} for k, d in A.items()},
                     T2_effect_median=float(np.median(eff)), T2_folds_pos=int((eff > 0).sum()), T2_p=p_t2,
                     T3_mean=[float(md.mean()) if len(md) else None, float(mi.mean()) if len(mi) else None],
                     T3_n=[int(len(md)), int(len(mi))], T3_p=p_t3)
ok = [out[v]["T2_effect_median"] > 0 and out[v]["T2_p"] < 0.05 and out[v]["T3_p"] < 0.05 for v in ("kg", "seek")]
no = [out[v]["T2_effect_median"] <= 0 or out[v]["T2_p"] >= 0.05 for v in ("kg", "seek")]
out["verdict"] = "SUPPORTED" if all(ok) else "NOT SUPPORTED" if all(no) else "MIXED"
P("\nVERDICT (pre-registered rule):", out["verdict"])
if out["pos_ge950"] < 0.8:
    P(f"WARNING: only {out['pos_ge950']:.0%} of dataset positives reach 0.950 in this STRING version -> version mismatch weakens the check")
json.dump(out, open("string_check_result.json", "w"), indent=1)
print("saved string_check_result.json")
