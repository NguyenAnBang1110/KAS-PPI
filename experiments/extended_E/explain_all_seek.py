"""E2 input: explain EVERY test pair of one fold on the SEEK view with KAS (D), LOO and SEEK's own explainer (SEEKN, SEEKS),
for full-data ROAR on SEEK. usage: python3 explain_all_seek.py MODEL FOLD [PART NPARTS] -> all_seek_{MODEL}_f{FOLD}[_p{PART}].json
(each pair explained by the fold model that did not train on it)."""
import sys, json, time
import numpy as np
MODEL, FOLD = sys.argv[1], int(sys.argv[2])
PART, NPARTS = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
sys.argv = ["x", "seek", MODEL, str(FOLD), "1"]
exec(open("explain_v2.py").read().split("pos=[i for i in test")[0])
from seeknat import native
out = []; t0 = time.time()
for n, i in enumerate(test[PART::NPARTS]):
    p1, p2 = keys[i]
    st = (set(kg.ann[p1]), set(kg.ann[p2]), set(sk.ann[p1]) - CX, set(sk.ann[p2]) - CX)
    p = proba(feats(st))[0]; r = dict(i=int(i), y=int(y[i]), pred=int(p > 0.5))
    phiN, phiS, stats, _, _ = native(st, sk, sub_sk, proba, len(A))
    for name, phi in (("D", METHODS["D"](st, 0)), ("LOO", METHODS["LOO"](st, 0)), ("SEEKN", phiN), ("SEEKS", phiS)):
        r[f"phi_{name}"] = {A[j]: round(float(phi[j]), 5) for j in np.nonzero(phi)[0]}
    r["n_nec"], r["n_suf"], r["n_dca"] = stats["n_nec"], stats["n_suf"], stats["n_dca"]
    out.append(r)
    if n % 500 == 0: print(FOLD, PART, n, round(time.time() - t0), "s", flush=True)
json.dump(dict(model=MODEL, fold=FOLD, aspects=A, records=out),
          open(f"all_seek_{MODEL}_f{FOLD}" + (f"_p{PART}" if NPARTS > 1 else "") + ".json", "w"))
print("done", FOLD, PART, round(time.time() - t0))
