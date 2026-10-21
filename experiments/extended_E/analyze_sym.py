"""Summary of roar_sym_{PIPE}_{MODEL}.json: WAF drop (base median - variant median) for MoRF and LeRF, k = 1..3,
gap = WAF(LeRF) - WAF(MoRF) (median over folds), mask-only accuracy, one-sided Wilcoxon of KAS (D) vs each method.
usage: python3 analyze_sym.py PIPE MODEL"""
import sys, json
import numpy as np
PIPE, MODEL = sys.argv[1], sys.argv[2]
o = json.load(open(f"roar_sym_{PIPE}_{MODEL}.json")); base = np.median(o["base"])
methods = sorted({k.split("|")[0] for k in o["res"]} - {"random"}, key=lambda m: (m != "D", m)) + ["random"]
print(f"== two-direction ROAR, {PIPE} + {MODEL.upper()}, {o['n']} pairs, base WAF {base:.4f} ==")
print("  %-9s | %-23s | %-23s | %-23s | %s" % ("method", "MoRF drop k=1/2/3 (up)", "LeRF drop k=1/2/3 (down)", "gap k=1/2/3 (up)", "mask-only MoRF/LeRF k=1"))
for m in methods:
    mo = [base - np.median(o["res"][f"{m}|MoRF|{k}"]) for k in (1, 2, 3) if f"{m}|MoRF|{k}" in o["res"]]
    le = [base - np.median(o["res"][f"{m}|LeRF|{k}"]) for k in (1, 2, 3) if f"{m}|LeRF|{k}" in o["res"]]
    gp = [np.median(np.array(o["res"][f"{m}|LeRF|{k}"]) - np.array(o["res"][f"{m}|MoRF|{k}"])) for k in (1, 2, 3) if f"{m}|LeRF|{k}" in o["res"]]
    mk = "%.3f / %s" % (o["mask"].get(f"{m}|MoRF|1", float("nan")), "%.3f" % o["mask"][f"{m}|LeRF|1"] if f"{m}|LeRF|1" in o["mask"] else "-")
    f = lambda v: " ".join("%.3f" % x for x in v) if v else "-"
    print("  %-9s | %-23s | %-23s | %-23s | %s" % (m, f(mo), f(le), f(gp), mk))
for k, v in sorted(o.get("p", {}).items()):
    if v < 0.05 or k.endswith("_1"): print("  p", k, "%.3g" % v)
