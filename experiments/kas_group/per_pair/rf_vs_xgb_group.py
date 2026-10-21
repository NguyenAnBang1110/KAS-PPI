"""G2b: per-pair agreement of group attributions between RF100 and XGB300 (same pairs), players only, >=3 players."""
import json, numpy as np
from scipy.stats import spearmanr, wilcoxon
out = {}
for v in ["kg", "seek"]:
    for p in ["source", "evidence"]:
        rho = {"KASg": [], "LOOg": []}; fm = {"KASg": [], "LOOg": []}
        for f in range(1, 11):
            A = json.load(open(f"kasg_{v}_{p}_f{f}.json")); B = json.load(open(f"g10/kasgx_{v}_{p}_f{f}.json"))
            G = A["groups"]; fr = {"KASg": [], "LOOg": []}
            for a, b in zip(A["records"], B["records"]):
                assert a["i"] == b["i"]
                pl = [G.index(g) for g in a["players"]]
                if len(pl) < 3: continue
                for m in rho:
                    x = np.array(a[m])[pl]; z = np.array(b[m])[pl]
                    r = spearmanr(x, z).correlation if np.ptp(x) > 0 and np.ptp(z) > 0 else 0.0
                    rho[m].append(r); fr[m].append(r)
            for m in rho: fm[m].append(np.mean(fr[m]))
        k, l = np.array(rho["KASg"]), np.array(rho["LOOg"])
        w = int(sum(x > y for x, y in zip(fm["KASg"], fm["LOOg"])))
        out[f"{v}_{p}"] = dict(n=len(k), KASg=float(k.mean()), LOOg=float(l.mean()), p=float(wilcoxon(k, l).pvalue), fold_wins=w)
        print(v, p, out[f"{v}_{p}"])
json.dump(out, open("g10/rf_vs_xgb_group.json", "w"), indent=1)
