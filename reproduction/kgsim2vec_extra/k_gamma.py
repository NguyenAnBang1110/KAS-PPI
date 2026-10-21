"""K3 gamma: the repo's get_semantic_aspects(gamma>0) crashes on this GO/GAF (KeyError in filter_classes_by_coverage for
classes that no annotation reaches, e.g. GO_0019012) and its coverage loop is O(classes x proteins) (>1 h).
Same logic, faster: coverage(c) = % of GAF proteins (all lines, as process_GAFversions) whose annotations or their
subClassOf-ancestors include c; a class no annotation reaches gets 0 (instead of KeyError). Starting from the 3 roots
and expanding once gives the 52 base aspects, so the gamma variant = base aspects with coverage >= gamma.
The code compares percentages with gamma directly, so gamma=0.01 means 0.01 %; the paper's text says 1 %. Both reported."""
import sys, os, json, collections, numpy as np
V = os.environ.get("KGSIM_WORKDIR", "work")   # thư mục làm việc: bản clone ExplainablePPI, feats_kg.npz, k10/
sys.path.insert(0, f"{V}/k10"); import kgsim_fix as KF
parents, label = KF.parse_go(f"{V}/godata/go-basic.owl")
anc_cache = {}
def anc(c):
    if c in anc_cache: return anc_cache[c]
    out = {c}; st = [c]
    while st:
        x = st.pop()
        for p in parents.get(x, ()):
            if p not in out: out.add(p); st.append(p)
    anc_cache[c] = out; return out
cov = collections.defaultdict(set)
for l in open(f"{V}/godata/goa_human.gaf"):
    if l.startswith("!"): continue
    f = l.split("\t"); t = "GO_" + f[4].split(":")[1]
    for a in anc(t): cov[a].add(f[1])
nprot = len({p for s in cov.values() for p in s} | set())
prots = set()
for l in open(f"{V}/godata/goa_human.gaf"):
    if not l.startswith("!"): prots.add(l.split("\t")[1])
base = list(np.load(f"{V}/feats_kg.npz")["aspects"])
pct = {a: 100 * len(cov.get(a, ())) / len(prots) for a in base}
res = {"n_proteins": len(prots), "coverage_pct": {f"{a} {label.get(a,'')}": round(v, 4) for a, v in sorted(pct.items(), key=lambda x: -x[1])}}
for g in (0.01, 1.0):
    sel = [a for a in base if pct[a] >= g]
    res[f"gamma_{g}"] = dict(n_selected=len(sel), complex_selected="GO_0032991" in sel, n_without_complex=len([a for a in sel if a != "GO_0032991"]), aspects=sel)
    print("gamma", g, "->", len(sel), "selected,", res[f"gamma_{g}"]["n_without_complex"], "without complex")
json.dump(res, open(f"{V}/k10/gamma_selection.json", "w"), indent=1)
