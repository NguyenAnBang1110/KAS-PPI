"""F3 summary: LORE vs KAS and the other explainers on the same pairs (KGsim2vec view), KAS's operator ('ks').
Also writes rec_kglore_{MODEL}_f{F}.json (phi of every method on the LORE pairs) for plaus2.py.
usage: python3 analyze_lore.py MODEL"""
import sys, glob, json
import numpy as np
from scipy.stats import wilcoxon
MODEL = sys.argv[1]
L = {}; folds = {}
for f in sorted(glob.glob(f"f09/lore_kg_{MODEL}_f*.json")):
    d = json.load(open(f))
    for r in d["records"]: L[r["i"]] = r; folds[r["i"]] = d["fold"]
R = {r["i"]: r for f in glob.glob(f"rec_kg_{MODEL}_f*_*.json") for r in json.load(open(f))["records"]}
ids = [i for i in L if i in R]
meths = ["D", "LOO", "FS0", "TreeSHAP", "LIME"]
print(f"== LORE (lore_sa, genetic neighbourhood 1,000) vs others, KGsim2vec view, {MODEL.upper()}, {len(ids)} pairs")
print(f"  LORE surrogate fidelity (local DT vs model on neighbourhood): mean {np.mean([L[i]['fid_LORE'] for i in ids]):.3f}")
print(f"  rule length mean {np.mean([L[i]['rule_len'] for i in ids]):.2f}; empty rule {np.mean([L[i]['rule_len']==0 for i in ids]):.3f}; "
      f"rule mentions >=1 player aspect {np.mean([L[i]['rule_players']>0 for i in ids]):.3f}; "
      f"share of rule aspects that are non-players {np.mean([1-L[i]['rule_players']/max(1,len(L[i]['rule_aspects'])) for i in ids if L[i]['rule_aspects']]):.3f}")
print(f"  coverage (>=1 ranked aspect) {np.mean([L[i]['cov_LORE'] for i in ids]):.3f}; time/pair {np.mean([L[i]['time_LORE'] for i in ids]):.1f} s "
      f"(KAS {np.mean([R[i]['time_D'] for i in ids]):.2f} s)")
def auc(c): return float(np.mean(c)) if c else np.nan
rows = {m: ([auc(R[i][f"del_{m}"]) for i in ids], [auc(R[i][f"ins_{m}"]) for i in ids]) for m in meths}
rows["LORE"] = ([auc(L[i]["del_LORE"]) for i in ids], [auc(L[i]["ins_LORE"]) for i in ids])
rows["random"] = ([auc(R[i]["del_random"]) for i in ids], [auc(R[i]["ins_random"]) for i in ids])
print("  method    deletion AUC (down)  insertion AUC (up)   [operator ks, NaN pairs = no ranked aspect, excluded]")
summ = {}
for m, (de, ins) in rows.items():
    summ[m] = (float(np.nanmean(de)), float(np.nanmean(ins))); print(f"  {m:9s} {np.nanmean(de):.3f}                {np.nanmean(ins):.3f}")
pv = {}
for m in rows:
    if m == "D": continue
    a, b = np.array(rows["D"][0]), np.array(rows[m][0]); ok = ~np.isnan(a) & ~np.isnan(b)
    a2, b2 = np.array(rows["D"][1]), np.array(rows[m][1]); ok2 = ~np.isnan(a2) & ~np.isnan(b2)
    pv[m] = (float(wilcoxon(a[ok], b[ok]).pvalue), float(np.mean(a[ok] < b[ok])), float(wilcoxon(a2[ok2], b2[ok2]).pvalue), float(np.mean(a2[ok2] > b2[ok2])))
    print(f"  KAS vs {m:9s}: deletion p={pv[m][0]:.2g} KAS lower in {pv[m][1]:.2f} of pairs; insertion p={pv[m][2]:.2g} KAS higher in {pv[m][3]:.2f}")
t1 = np.mean([int(bool(R[i]['top3_D']) and bool(L[i]['phi_LORE']) and R[i]['top3_D'][0] == max(L[i]['phi_LORE'], key=lambda a: abs(L[i]['phi_LORE'][a]))) for i in ids])
print(f"  top-1 aspect same as KAS: {t1:.3f}")
json.dump(dict(n=len(ids), summ=summ, p=pv, top1_same_D=float(t1)), open(f"f09/lore_{MODEL}_summary.json", "w"))
# merged records for plausibility
by = {}
for i in ids:
    o = {k: R[i][k] for k in ("i", "pair", "y", "pred")}; o.update({f"phi_{m}": R[i][f"phi_{m}"] for m in meths}); o["phi_LORE"] = L[i]["phi_LORE"]
    by.setdefault(folds[i], []).append(o)
A = json.load(open(glob.glob(f"rec_kg_{MODEL}_f1_*.json")[0]))["aspects"]
for f, recs in by.items(): json.dump(dict(aspects=A, records=recs), open(f"rec_kglore_{MODEL}_f{f}.json", "w"))
