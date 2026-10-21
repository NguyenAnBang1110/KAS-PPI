"""K2/K3: KGsim2vec feature variants with kgsim_fix.py (Python re-implementation that matches SML 100% of cells).
usage: python3 k_feats.py VARIANT      VARIANT in {check, nocx, beta1, gamma001}
  check    : base 52 aspects on the full graph -> must equal the authors' SML file (Xref)
  nocx     : paper's stated procedure (Sousa24 sec 4.1): remove the 'protein-containing complex' branch (GO_0032991 and all
             its subclasses) from the GO graph AND the annotations to those classes; 51 base aspects
  beta1    : aspects from the repo's get_semantic_aspects(alpha=10, gamma=0, beta=1)   (Table 10 row 2, paper: 42 SAs)
  gamma001 : aspects from get_semantic_aspects(alpha=10, gamma=0.01, beta=0)             (Table 10 row 3, paper: 24 SAs)
  For beta1/gamma001 the complex aspect, if selected, is dropped (same convention as the 51-aspect base)."""
import sys, os, json, time, numpy as np
V = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(V, "k10"))
import kgsim_fix as KF
VAR = sys.argv[1]
K = np.load(f"{V}/feats_kg.npz"); keys = [tuple(k) for k in K["keys"]]; base = list(K["aspects"])
CX = "GO_0032991"; t0 = time.time(); info = {"variant": VAR}
owl, gaf = f"{V}/godata/go-basic.owl", f"{V}/godata/goa_human.gaf"
if VAR in ("check", "nocx"):
    asp = base if VAR == "check" else [a for a in base if a != CX]
elif VAR == "gamma001":   # repo function crashes for gamma>0 on this GO; selection from k_gamma.py (coverage >= 1 %, = paper's 24)
    G_ = json.load(open(f"{V}/k10/gamma_selection.json"))["gamma_1.0"]
    info["n_selected"] = G_["n_selected"]; info["complex_selected"] = G_["complex_selected"]; info["note"] = "selection by k_gamma.py, coverage >= 1 %"
    asp = [a for a in G_["aspects"] if a != CX]
else:
    sys.path.insert(0, f"{V}/ExplainablePPI"); cwd = os.getcwd(); os.chdir(f"{V}/ExplainablePPI")
    import ss_calculation_sas as S
    beta, gamma = (1, 0) if VAR == "beta1" else (0, 0.01)
    names, s = S.get_semantic_aspects(f"{V}/k10/sas_{VAR}.txt", owl, gaf, 10, gamma, beta)
    os.chdir(cwd)
    urls = [u.strip('"').split("/")[-1] for u in s.split()]
    info["n_selected"] = len(urls); info["complex_selected"] = CX in urls
    asp = [a for a in urls if a != CX]
info["n_aspects"] = len(asp)
kg = KF.KG(owl, gaf, asp)
if VAR == "nocx":
    drop = kg.desc(CX) | {CX}
    info["n_removed_classes"] = len(drop)
    parents = {c: {p for p in ps if p not in drop} for c, ps in kg.parents.items() if c not in drop}
    kg2 = KF.KG.__new__(KF.KG); kg2.parents = parents; kg2.label = kg.label
    import collections
    kg2.children = collections.defaultdict(set)
    for c, ps in parents.items():
        for p in ps: kg2.children[p].add(c)
    ann = {p: {t for t in ts if t not in drop} for p, ts in kg.ann.items()}
    info["n_removed_annotations"] = sum(len(ts) for ts in kg.ann.values()) - sum(len(ts) for ts in ann.values())
    kg2.ann = collections.defaultdict(set, ann); kg2.aspects = asp; kg2._anc = {}; kg2.sub = []; kg2.ic = []
    KF.KG.desc.__defaults__[0].clear()   # desc() caches across instances
    import math
    for a in asp:
        members = kg2.desc(a) | {a}
        nprot = sum(1 for p, ts in kg2.ann.items() if ts & members)
        N = len(members) + nprot
        kg2.sub.append(members); kg2.ic.append({c: 1 - math.log(len(kg2.desc(c)) + 1) / math.log(N) if N > 1 else 1.0 for c in members})
    kg = kg2
X = np.array([kg.features(kg.ann.get(a, set()), kg.ann.get(b, set())) for a, b in keys])
if VAR == "check":
    info["maxdiff_vs_SML"] = float(np.abs(X - K["Xref"]).max())
if VAR == "nocx":
    keep = [j for j, a in enumerate(base) if a != CX]
    d = np.abs(X - K["Xref"][:, keep]); info["cells_changed_vs_base51"] = float((d > 1e-12).mean()); info["maxdiff_vs_base51"] = float(d.max())
names = [kg.label.get(a, a).replace(" ", "_").replace("-", "_") for a in asp]
info["names"] = names; info["seconds"] = round(time.time() - t0)
np.savez(f"{V}/k10/feats_{VAR}.npz", X=X, names=np.array(names), urls=np.array(asp))
json.dump(info, open(f"{V}/k10/feats_{VAR}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "names"}))
