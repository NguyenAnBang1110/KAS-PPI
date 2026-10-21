"""E1 summary: KAS (D) vs SEEK's own explainer mapped to aspects (SEEKN graded-necessary, SEEKS graded-sufficient), LOO,
random, on the SEEK view; deletion/insertion under both operators ('ks' = delete annotations + recompute, KAS's value
function; 'dca' = drop DCA vectors, SEEK's operator). Same aggregation as analyze_v2.py (deletion = mean of k=1..5,
insertion = mean of k=1..5; empty explanation -> deletion stays at the original confidence, insertion at the all-removed
state). usage: python3 analyze_e1.py  -> e1_summary.json, prints a table"""
import json, glob, collections
import numpy as np
from scipy.stats import wilcoxon
M = ["D", "LOO", "SEEKN", "SEEKS", "random"]; out = {}
for model in ("rf", "xgb"):
    conf = {}
    for fn in glob.glob(f"rec_seek_{model}_f*_*.json"):
        for r in json.load(open(fn))["records"]: conf[r["i"]] = r["conf"]
    R = []
    for fn in sorted(glob.glob(f"rec_seekx_{model}_f*.json")): R += json.load(open(fn))["records"]
    if not R: continue
    s = dict(n=len(R), folds=len(glob.glob(f"rec_seekx_{model}_f*.json")))
    for sp in ("ks", "dca"):
        dels, inss = {}, {}
        for m in M:
            dels[m] = np.array([np.mean(r[f"del_{sp}_{m}"]) if r.get(f"del_{sp}_{m}") else conf[r["i"]] for r in R])
            inss[m] = np.array([np.mean(r[f"ins_{sp}_{m}"][1:]) if r.get(f"ins_{sp}_{m}") else
                                (r[f"ins_{sp}_random"][0] if r.get(f"ins_{sp}_random") else conf[r["i"]]) for r in R])
            s[f"del_{sp}_{m}"] = float(dels[m].mean()); s[f"ins_{sp}_{m}"] = float(inss[m].mean())
        for m in M[1:]:
            for nm, arr, alt in (("del", dels, "less"), ("ins", inss, "greater")):
                a, b = arr["D"], arr[m]
                s[f"p_{nm}_{sp}_D_vs_{m}"] = float(wilcoxon(a, b, alternative=alt).pvalue) if np.any(a != b) else 1.0
                s[f"Dbetter_{nm}_{sp}_{m}"] = float(np.mean((a < b) if nm == "del" else (a > b)))
    for m in M[:-1]:
        s[f"pars_{m}"] = float(np.mean([r[f"size_{m}"] for r in R])); s[f"cov_{m}"] = float(np.mean([r[f"cov_{m}"] for r in R]))
    for m in ("LOO", "SEEKN", "SEEKS"):
        s[f"top1same_D_{m}"] = float(np.mean([r[f"top1same_D_{m}"] for r in R])); s[f"jac3_D_{m}"] = float(np.mean([r[f"jac3_D_{m}"] for r in R]))
    s["dca_mean"] = float(np.mean([r["n_dca"] for r in R])); s["dca_median"] = float(np.median([r["n_dca"] for r in R]))
    s["dca_outside_aspects_frac"] = float(sum(r["n_dca_noasp"] for r in R) / max(1, sum(r["n_dca"] for r in R)))
    s["pairs_with_necessary"] = float(np.mean([r["n_nec"] > 0 for r in R])); s["necessary_mean"] = float(np.mean([r["n_nec"] for r in R]))
    s["sufficient_mean"] = float(np.mean([r["n_suf"] for r in R]))
    s["time_SEEK_median"] = float(np.median([r["time_SEEK"] for r in R]))
    A = json.load(open(sorted(glob.glob(f"rec_seekx_{model}_f*.json"))[0]))["aspects"]
    hit = [int(r["top1_D"] is not None and A.index(r["top1_D"]) in r["nec_asp"]) for r in R if r["nec_asp"]]
    s["KAS_top1_in_SEEK_necessary_aspects"] = float(np.mean(hit)) if hit else None
    out[model] = s
    print(f"== SEEK view, {model.upper()}: {s['n']} pairs ({s['folds']} folds) ==")
    print("  %-8s %8s %8s %8s %8s %6s %6s" % ("method", "del_ks", "ins_ks", "del_dca", "ins_dca", "pars", "cov"))
    for m in M:
        print("  %-8s %8.3f %8.3f %8.3f %8.3f %6s %6s" % (m, s[f"del_ks_{m}"], s[f"ins_ks_{m}"], s[f"del_dca_{m}"], s[f"ins_dca_{m}"],
              "%.2f" % s[f"pars_{m}"] if m != "random" else "", "%.3f" % s[f"cov_{m}"] if m != "random" else ""))
    for m in M[1:]:
        print("  D vs %-6s p(del_ks) %.2g p(ins_ks) %.2g p(del_dca) %.2g p(ins_dca) %.2g | D better del_ks %.2f del_dca %.2f" % (
            m, s[f"p_del_ks_D_vs_{m}"], s[f"p_ins_ks_D_vs_{m}"], s[f"p_del_dca_D_vs_{m}"], s[f"p_ins_dca_D_vs_{m}"],
            s[f"Dbetter_del_ks_{m}"], s[f"Dbetter_del_dca_{m}"]))
    print("  agreement with KAS: top-1 same LOO %.2f SEEKN %.2f SEEKS %.2f | Jaccard@3 LOO %.2f SEEKN %.2f SEEKS %.2f" % tuple(
        [s[f"top1same_D_{m}"] for m in ("LOO", "SEEKN", "SEEKS")] + [s[f"jac3_D_{m}"] for m in ("LOO", "SEEKN", "SEEKS")]))
    print("  SEEK native: DCAs/pair mean %.1f (median %.0f), outside every aspect %.1f%%; pairs with >=1 necessary DCA %.1f%%; "
          "necessary %.2f, sufficient %.2f per pair; KAS top-1 inside a SEEK-necessary aspect %.2f; time %.3fs" % (
          s["dca_mean"], s["dca_median"], 100 * s["dca_outside_aspects_frac"], 100 * s["pairs_with_necessary"], s["necessary_mean"],
          s["sufficient_mean"], s["KAS_top1_in_SEEK_necessary_aspects"] or float("nan"), s["time_SEEK_median"]))
json.dump(out, open("e1_summary.json", "w"), indent=1)
