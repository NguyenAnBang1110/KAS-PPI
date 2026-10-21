"""Summarise roar_*.json (+ roar_mask_*.json): mean accuracy after retraining, drop vs base, Wilcoxon of KAS (D) vs each
method over the same CV splits (one-sided alternative: D gives LOWER accuracy = more faithful)."""
import json, glob, numpy as np
from scipy.stats import wilcoxon
out = {}
for f in sorted(glob.glob("roar_kg_*.json") + glob.glob("roar_seek_*.json") + glob.glob("roar_hyb_*.json")):
    if "mask" in f: continue
    d = json.load(open(f)); cfg = f"{d['pipe']}|{d['model']}"
    mf = f"roar_mask_{d['pipe']}_{d['model']}.json"; mask = json.load(open(mf)) if glob.glob(mf) else {}
    b = np.mean(d["base"]["acc"]); print(f"\n{cfg}  n={d['n_pairs']}  base acc {b:.3f}")
    out[cfg] = dict(base=b, rows={})
    for key, v in d["res"].items():
        op, m, k = key.split("|"); a = np.array(v["acc"])
        dd = np.array(d["res"][f"{op}|D|{k}"]["acc"])
        p = float(wilcoxon(dd, a, alternative="less").pvalue) if m != "D" and np.any(dd != a) else None
        out[cfg]["rows"][key] = dict(acc=float(a.mean()), drop=float(b - a.mean()), p_D_less=p, mask_acc=mask.get(f"{m}|{k}"))
        print(f"  {op:4s} {m:9s} k={k}  acc {a.mean():.3f}  drop {b-a.mean():.3f}" + (f"  p(D<{m})={p:.2g}" if p is not None else "") +
              (f"  mask-only {mask[f'{m}|{k}']:.3f}" if f"{m}|{k}" in mask else ""))
json.dump(out, open("roar_summary.json", "w"), indent=1)
