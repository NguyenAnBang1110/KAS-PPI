"""Plausibility case study table (KGsim2vec + RF, records from the 10-fold runs): top-3 aspects of KAS / TreeSHAP / LIME
vs the aspects over-represented in the pair's shared Reactome pathway (plaus.py reference). Output: case_study.md"""
import json, collections
d = json.load(open("plaus_REACTOME.json")); cases = {c["pair"]: c for c in d["cases"]}
sym = {}
for l in open("godata/goa_human.gaf"):
    if l.startswith("!"): continue
    f = l.split("\t"); sym.setdefault(f[1], f[2])
PAIRS = ["P00533-P15514", "Q05086-Q16763", "P62917-P05388", "Q9BYM8-P21580", "P19447-P20226", "Q9Y6K5-P29728",
         "O60231-Q7L014", "Q9Y5J6-O14925", "Q15788-Q13133", "Q9UJX6-O95997", "Q9Y676-P82650", "Q93034-P46934"]
def fmt(lst, rel): return "; ".join(("**%s** ✓" % a) if a in rel else a for a in lst)
L = ["| Cặp (gene) | Pathway Reactome chung (số gene) | Aspect được làm giàu | KAS top-3 | TreeSHAP top-3 | LIME top-3 |", "|---|---|---|---|---|---|"]
hits = collections.Counter()
for p in PAIRS:
    c = cases[p]; a, b = p.split("-"); rel = set(c["relevant"])
    L.append(f"| {sym.get(a,a)}–{sym.get(b,b)} | {c['pathway'].replace('REACTOME_','').replace('_',' ').lower()} ({c['size']}) | {', '.join(c['relevant'])} | "
             f"{fmt(c['top3']['D'],rel)} | {fmt(c['top3']['TreeSHAP'],rel)} | {fmt(c['top3']['LIME'],rel)} |")
    for m in ("D", "TreeSHAP", "LIME"): hits[m] += c["top3"][m][0] in rel
L.append(""); L.append("Top-1 nằm trong aspect được làm giàu: " + ", ".join(f"{'KAS' if m=='D' else m} {hits[m]}/{len(PAIRS)}" for m in ("D", "TreeSHAP", "LIME")))
open("case_study.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
