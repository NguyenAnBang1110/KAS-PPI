"""E5: KGsim2vec Table 9 (most frequent DT6 rules) reproduction.
Protocol of Sousa et al. 2024 (sec. 4.6 + Prediction/ML.py, processDTs.py): DT with max_depth=6 on the SML similarity
features of 51 aspects, one model per repo partition (10 models); sibling leaves with the same decision are pruned as in
prune_duplicate_leaves; each leaf gives a rule (per feature: max of '>' cut-offs, min of '<=' cut-offs); cut-offs rounded
to one decimal; a rule's frequency = number of models containing it; #pairs = test pairs reaching that rule, summed over
the models that contain it. Rules in >= 5 models are compared with the paper's Table 9.
usage: python3 dt6_rules.py [SEEDS]   (default 0; e.g. 0,1,2,3,4 to check stability) -> dt6_rules.json"""
import sys, json, collections, re
import numpy as np
sys.path.insert(0, ".")
from kgsim import parse_go
from sklearn.tree import DecisionTreeClassifier
SEEDS = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0").split(",")]
K = np.load("feats_kg.npz"); y = K["y"]; aspects = list(K["aspects"])
ci = aspects.index("GO_0032991"); keep = [j for j in range(52) if j != ci]
_, label = parse_go("godata/go-basic.owl")
names = [label[aspects[j]].replace(" ", "_") for j in keep]
X = K["Xref"][:, keep]
P = "ExplainablePPI/Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run"
folds = [[int(l) for l in open(P + ("1(complete)" if f == 1 else str(f)) + ".txt") if l.strip()] for f in range(1, 11)]
PAPER = """1|binding <= 0.5 & 0.0 < cellular_process <= 0.6 & biological_regulation <= 0.5|0|10|17216
2|metabolic_process > 0.6 & cellular_process > 0.8|1|5|4837
3|metabolic_process <= 0.6 & cellular_process > 0.9|1|5|2359
4|cellular_anatomical_entity > 0.8 & 0.6 < cellular_process <= 0.8|1|6|2082
5|binding <= 0.5 & cellular_anatomical_entity > 0.1 & cellular_process <= 0.0 & biological_regulation <= 0.5|0|10|1403
6|binding <= 0.7 & cellular_anatomical_entity > 0.7 & cellular_process <= 0.5 & 0.5 < biological_regulation <= 0.9|0|10|841
7|0.5 < binding <= 0.5 & cellular_anatomical_entity > 0.8 & cellular_process <= 0.6 & biological_regulation <= 0.5|1|10|574
8|binding <= 0.7 & cellular_anatomical_entity > 0.7 & 0.5 < cellular_process <= 0.6 & 0.5 < biological_regulation <= 0.9|1|10|515
9|binding > 0.7 & cellular_anatomical_entity > 0.7 & cellular_process <= 0.6 & 0.5 < biological_regulation <= 0.9|1|8|472
10|0.5 < binding <= 0.7 & cellular_anatomical_entity > 0.9 & cellular_process <= 0.6 & biological_regulation <= 0.5|0|6|350
11|binding > 0.7 & cellular_anatomical_entity > 0.9 & cellular_process <= 0.6 & biological_regulation <= 0.5|1|6|268
12|binding <= 0.1 & cellular_anatomical_entity <= 0.1 & cellular_process <= 0.0 & biological_regulation <= 0.5|1|6|248
13|0.5 < binding <= 0.5 & 0.1 < cellular_anatomical_entity <= 0.8 & cellular_process <= 0.6 & biological_regulation <= 0.5|0|10|217
14|binding <= 0.5 & 0.1 < cellular_anatomical_entity <= 0.8 & 0.6 < cellular_process <= 0.8 & biological_regulation <= 0.7|0|7|210
15|0.1 < binding <= 0.5 & cellular_anatomical_entity <= 0.1 & cellular_process <= 0.0 & biological_regulation <= 0.5|0|6|157
16|cellular_anatomical_entity <= 0.8 & 0.6 < cellular_process <= 0.8 & biological_regulation > 0.7|1|6|101
17|0.5 < binding <= 0.5 & cellular_anatomical_entity <= 0.1 & cellular_process <= 0.1 & biological_regulation <= 0.5|1|7|93
18|cellular_anatomical_entity <= 0.7 & 0.0 < cellular_process <= 0.5 & biological_regulation > 0.7|0|5|86"""
def parse_rule(txt):
    d = {}
    for c in txt.split(" & "):
        m = re.fullmatch(r"([\d.]+) < (\w+) <= ([\d.]+)", c)
        if m: d[m[2]] = (float(m[1]), float(m[3])); continue
        m = re.fullmatch(r"(\w+) > ([\d.]+)", c)
        if m: d[m[1]] = (float(m[2]), None); continue
        m = re.fullmatch(r"(\w+) <= ([\d.]+)", c); d[m[1]] = (None, float(m[2]))
    return frozenset(d.items())
paper = [dict(n=int(a), rule=b, key=(parse_rule(b), int(c)), models=int(d), pairs=int(e))
         for a, b, c, d, e in (l.split("|") for l in PAPER.splitlines())]
def fmt(key):
    conds, cls = key; parts = []
    for f, (lo, hi) in sorted(conds):
        parts.append(f"{lo} < {f} <= {hi}" if lo is not None and hi is not None else f"{f} > {lo}" if lo is not None else f"{f} <= {hi}")
    return " & ".join(parts) + f"  => {'+' if cls else '-'}"
def rules_of(clf):
    t = clf.tree_; L, R = t.children_left.copy(), t.children_right.copy(); feat = t.feature.copy()
    dec = t.value.argmax(axis=2).flatten()
    leaf = lambda i: L[i] == -1 and R[i] == -1
    def prune(i):          # prune_index of the authors' ML.py, on copies of the arrays
        if not leaf(L[i]): prune(L[i])
        if not leaf(R[i]): prune(R[i])
        if leaf(L[i]) and leaf(R[i]) and dec[i] == dec[L[i]] == dec[R[i]]: L[i] = R[i] = -1; feat[i] = -2
    if not leaf(0): prune(0)
    out = {}
    def rec(i, path):
        if L[i] == -1:
            iv = {}
            for f, op, th in path:
                lo, hi = iv.get(f, (None, None))
                if op == ">": lo = th if lo is None else max(lo, th)
                else: hi = th if hi is None else min(hi, th)
                iv[f] = (lo, hi)
            r = lambda v: None if v is None else round(v, 1)
            out[i] = (frozenset((names[f], (r(lo), r(hi))) for f, (lo, hi) in iv.items()), int(dec[i])); return
        rec(L[i], path + [(feat[i], "<=", t.threshold[i])]); rec(R[i], path + [(feat[i], ">", t.threshold[i])])
    rec(0, [])
    def node_of(x):
        i = 0
        while L[i] != -1: i = L[i] if x[feat[i]] <= t.threshold[i] else R[i]
        return i
    return out, node_of
result = {}
for seed in SEEDS:
    freq = collections.Counter(); npairs = collections.Counter(); waf = []
    for te in folds:
        m = np.zeros(len(y), bool); m[te] = True
        clf = DecisionTreeClassifier(max_depth=6, random_state=seed).fit(X[~m], y[~m])
        rules, node_of = rules_of(clf)
        for key in set(rules.values()): freq[key] += 1
        for i in te: npairs[rules[node_of(X[i])]] += 1
    frequent = sorted([k for k, c in freq.items() if c >= 5], key=lambda k: -npairs[k])
    def close(a, b):       # same features, same class, every bound within 0.1
        if a[1] != b[1]: return False
        da, db = dict(a[0]), dict(b[0])
        if set(da) != set(db): return False
        for f in da:
            for u, v in zip(da[f], db[f]):
                if (u is None) != (v is None) or (u is not None and abs(u - v) > 0.1 + 1e-9): return False
        return True
    comp = []
    for p in paper:
        exact = freq.get(p["key"], 0)
        near = [k for k in freq if close(k, p["key"])]
        comp.append(dict(n=p["n"], rule=fmt(p["key"]), paper_models=p["models"], paper_pairs=p["pairs"],
                         ours_models_exact=exact, ours_pairs_exact=npairs.get(p["key"], 0),
                         ours_models_near=max([freq[k] for k in near], default=0)))
    result[seed] = dict(n_frequent=len(frequent), n_paper=len(paper),
                        n_paper_exact_found=sum(c["ours_models_exact"] > 0 for c in comp),
                        n_paper_exact_frequent=sum(c["ours_models_exact"] >= 5 for c in comp),
                        n_paper_near_frequent=sum(c["ours_models_near"] >= 5 for c in comp),
                        n_ours_frequent_in_paper=sum(any(close(k, p["key"]) for p in paper) for k in frequent),
                        comparison=comp, ours=[dict(rule=fmt(k), models=freq[k], pairs=npairs[k]) for k in frequent])
    r = result[seed]
    print(f"seed {seed}: our frequent rules {r['n_frequent']}; paper rules found exactly in >=1 model {r['n_paper_exact_found']}/18, "
          f"in >=5 models {r['n_paper_exact_frequent']}/18, near (bounds +-0.1) in >=5 models {r['n_paper_near_frequent']}/18; "
          f"our frequent rules matching a paper rule (near) {r['n_ours_frequent_in_paper']}/{r['n_frequent']}")
for c in result[SEEDS[0]]["comparison"]:
    print("%2d %-110s paper %2d/%5d  ours exact %2d/%5d near %2d" % (c["n"], c["rule"], c["paper_models"], c["paper_pairs"],
          c["ours_models_exact"], c["ours_pairs_exact"], c["ours_models_near"]))
print("our frequent rules (seed %d):" % SEEDS[0])
for o in result[SEEDS[0]]["ours"]: print("  %2d models %5d pairs  %s" % (o["models"], o["pairs"], o["rule"]))
json.dump(result, open("dt6_rules.json", "w"), indent=1)
