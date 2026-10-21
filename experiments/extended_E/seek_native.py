"""E1: SEEK's own explainer vs KAS on the SEEK view, same 2,200 pairs per model as the main runs (rec_seek_{MODEL}_f*.json).
usage: python3 seek_native.py MODEL FOLD      -> rec_seekx_{MODEL}_f{FOLD}.json
Two removal operators: 'ks' = delete annotations + recompute (KAS's value function, same code as explain_v2),
'dca' = drop DCA vectors of the removed aspects, no recompute (SEEK's own operator)."""
import sys, json, glob, time
import numpy as np
MODEL, FOLD = sys.argv[1], int(sys.argv[2])
sys.argv = ["x", "seek", MODEL, str(FOLD), "1"]
exec(open("explain_v2.py").read().split("pos=[i for i in test")[0])
from seeknat import native, dca_curves
recs = []
for fn in glob.glob(f"rec_seek_{MODEL}_f{FOLD}_*.json"): recs += json.load(open(fn))["records"]
out = []; t0 = time.time()
for n, r in enumerate(recs):
    i = r["i"]; p1, p2 = keys[i]
    st = (set(kg.ann[p1]), set(kg.ann[p2]), set(sk.ann[p1]) - CX, set(sk.ann[p2]) - CX)
    pred = r["pred"]; pl = players_of(st)
    ts = time.time(); phiN, phiS, stats, D, asp_of = native(st, sk, sub_sk, proba, len(A)); tn = time.time() - ts
    phis = {"D": np.array([r["phi_D"].get(a, 0.0) for a in A]), "LOO": np.array([r["phi_LOO"].get(a, 0.0) for a in A]),
            "SEEKN": phiN, "SEEKS": phiS}
    o = dict(i=i, pair=r["pair"], y=r["y"], pred=pred, k=len(pl), time_SEEK=tn, **stats)
    rr = np.random.default_rng(i); orders = {m: ranking(f, pred) for m, f in phis.items()}
    orders["random"] = [int(v) for v in rr.permutation(pl)]
    for m, f in phis.items():
        o[f"phi_{m}"] = {A[j]: round(float(f[j]), 5) for j in np.nonzero(f)[0]}
        o[f"size_{m}"] = eff_size(f); o[f"cov_{m}"] = int(len(orders[m]) > 0)
    for m, od in orders.items():
        if not od: continue
        if m in ("D", "LOO"): o[f"del_ks_{m}"], o[f"ins_ks_{m}"] = r[f"del_{m}"], r[f"ins_{m}"]
        else: o[f"del_ks_{m}"], o[f"ins_ks_{m}"] = deletion_ks(st, od, pred), insertion_ks(st, od, pred, pl)
        o[f"del_dca_{m}"], o[f"ins_dca_{m}"] = dca_curves(D, asp_of, od, pl, pred, sk, proba)
    o["top1_D"] = A[orders["D"][0]] if orders["D"] else None
    for m in ("LOO", "SEEKN", "SEEKS"):
        a, b = orders["D"][:3], orders[m][:3]
        o[f"jac3_D_{m}"] = len(set(a) & set(b)) / len(set(a) | set(b)) if (a or b) else 1.0
        o[f"top1same_D_{m}"] = int(bool(a) and bool(b) and a[0] == b[0])
    out.append({k: (float(v) if isinstance(v, np.floating) else v) for k, v in o.items()})
    if n % 100 == 0: print(MODEL, FOLD, n, round(time.time() - t0), "s", flush=True)
json.dump(dict(pipe="seekx", model=MODEL, fold=FOLD, aspects=A, records=out), open(f"rec_seekx_{MODEL}_f{FOLD}.json", "w"))
print("done", MODEL, FOLD, round(time.time() - t0))
