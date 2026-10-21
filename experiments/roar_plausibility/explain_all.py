"""Explain EVERY test pair of one fold (KGsim2vec view) with KAS, LOO, FS0, TreeSHAP, LIME, for full-data ROAR.
usage: python3 explain_all.py MODEL FOLD [PART NPARTS]   -> all_kg_{MODEL}_f{FOLD}[_p{PART}].json
(each pair explained by the fold model that did not train on it; PART/NPARTS split a fold across CPUs)"""
import sys, json, time
import numpy as np
MODEL, FOLD = sys.argv[1], int(sys.argv[2])
PART, NPARTS = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
sys.argv = ["x", "kg", MODEL, str(FOLD), "1"]
exec(open("explain_v2.py").read().split("pos=[i for i in test")[0])   # setup: data, model, feats, explainers
out = []; t0 = time.time()
for n, i in enumerate(test[PART::NPARTS]):
    p1, p2 = keys[i]
    st = (set(kg.ann[p1]), set(kg.ann[p2]), set(), set())
    p = proba(feats(st))[0]; r = dict(i=int(i), y=int(y[i]), pred=int(p > 0.5))
    for name, f in METHODS.items():
        phi = f(st, 0); r[f"phi_{name}"] = {A[j]: round(float(phi[j]), 5) for j in np.nonzero(phi)[0]}
    out.append(r)
    if n % 500 == 0: print(FOLD, n, round(time.time() - t0), "s", flush=True)
json.dump(dict(model=MODEL, fold=FOLD, aspects=A, records=out), open(f"all_kg_{MODEL}_f{FOLD}" + (f"_p{PART}" if NPARTS > 1 else "") + ".json", "w"))
print("done", FOLD, time.time() - t0)
