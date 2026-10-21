"""Quantitative Plausibility proxy: does an explanation's top aspect agree with external pathway knowledge?
usage: python3 plaus.py GMT_NAME     (reads {GMT_NAME}_msigdb751.gmt: REACTOME / KEGG from MSigDB 7.5.1, or REACTOME_LIVE
       written by get_reactome_live.py from the current reactome.org UniProt2Reactome file)
For every explained pair with y=1 and pred=1 (all rec_*_{rf,xgb}_f*.json), take the smallest gene set (10..300 members in the
GAF universe) that contains both proteins. Reference relevance of an aspect = over-representation (one-sided hypergeometric,
BH over the 51 aspects) of the OTHER members of that pathway (the two proteins excluded) among proteins annotated under the
aspect, versus all GAF proteins. Annotations used for the reference exclude evidence IEA and anything assigned_by Reactome,
so the reference does not reuse Reactome-derived GO annotations.
A pair is informative when it has >=2 players and at least one relevant and one non-relevant player.
Metrics over the pair's players: hit@1, precision@3, NDCG@3 (gain = -log10 p, capped at 10).
Baselines: random order (exact expectation for hit@1, 200 permutations otherwise) and, for KGsim2vec, ranking players by
their raw similarity feature (model-free)."""
import sys, glob, json, math, random, collections
import numpy as np
from scipy.stats import hypergeom, wilcoxon
sys.path.insert(0, ".")
from kgsim import KG

GMT = sys.argv[1]
K = np.load("feats_kg.npz"); keys = [tuple(k) for k in K["keys"]]; aspects = list(K["aspects"])
ci = aspects.index("GO_0032991"); keep = [j for j in range(52) if j != ci]; A = [aspects[j] for j in keep]
kg = KG("godata/go-basic.owl", "godata/goa_human.gaf", aspects); sub = [kg.sub[j] for j in keep]
Xpy = K["Xpy"][:, keep]

# filtered annotations (no IEA, not assigned by Reactome) and symbol -> accession
filt = collections.defaultdict(set); sym2acc = collections.defaultdict(set)
for l in open("godata/goa_human.gaf"):
    if l.startswith("!"): continue
    f = l.rstrip("\n").split("\t")
    if len(f) < 15: continue
    sym2acc[f[2]].add(f[1])
    if "NOT" in f[3] or f[6] == "IEA" or f[14] == "Reactome": continue
    filt[f[1]].add("GO_" + f[4].split(":")[1])
U = sorted(filt); NU = len(U)
asp_has = np.array([[bool(filt[p] & sub[a]) for a in range(len(A))] for p in U])  # NU x 51
uidx = {p: i for i, p in enumerate(U)}
bg = asp_has.sum(0)

sets = []
for l in open(f"{GMT}_msigdb751.gmt"):
    f = l.rstrip("\n").split("\t")
    mem = {a for s in f[2:] for a in (sym2acc.get(s) or {s})} & set(uidx)   # members may be gene symbols or UniProt accessions
    if 10 <= len(mem) <= 300: sets.append((f[0], mem))
member_of = collections.defaultdict(list)
for j, (_, mem) in enumerate(sets):
    for p in mem: member_of[p].append(j)
print(GMT, "gene sets kept", len(sets), "universe", NU, flush=True)

relcache = {}
def relevance(j, p1, p2):
    key = (j, p1, p2)
    if key in relcache: return relcache[key]
    M = [uidx[p] for p in sets[j][1] if p not in (p1, p2)]
    cnt = asp_has[M].sum(0); n = len(M)
    pv = np.array([hypergeom.sf(cnt[a] - 1, NU, bg[a], n) if bg[a] > 0 else 1.0 for a in range(len(A))])
    order = np.argsort(pv); q = np.empty_like(pv); m = len(pv); prev = 1.0
    for rank, a in reversed(list(enumerate(order))):
        prev = min(prev, pv[a] * m / (rank + 1)); q[a] = prev
    relcache[key] = (q < 0.05, np.minimum(-np.log10(np.maximum(pv, 1e-300)), 10.0))
    return relcache[key]

def players(p1, p2):
    t1, t2 = kg.ann[p1], kg.ann[p2]
    return [i for i in range(len(A)) if t1 & sub[i] and t2 & sub[i]]
def ndcg3(order, gain):
    dcg = sum(gain[a] / math.log2(r + 2) for r, a in enumerate(order[:3]))
    ideal = sorted((gain[a] for a in order), reverse=True)
    idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal[:3]))
    return dcg / idcg if idcg > 0 else 0.0

out = {}; cases = []
for cfg in ["kg|rf", "kg|xgb", "seek|rf", "seek|xgb", "hyb|rf", "hyb|xgb"]:
    pipe, model = cfg.split("|"); recs = []
    for fn in sorted(glob.glob(f"rec_{pipe}_{model}_f*_*.json")): recs += json.load(open(fn))["records"]
    methods = [k[4:] for k in recs[0] if k.startswith("phi_")]
    res = collections.defaultdict(lambda: collections.defaultdict(list)); n_tp = n_shared = 0
    for r in recs:
        if not (r["y"] == 1 and r["pred"] == 1): continue
        n_tp += 1
        p1, p2 = keys[r["i"]]
        shared = set(member_of.get(p1, ())) & set(member_of.get(p2, ()))
        if not shared: continue
        n_shared += 1
        j = min(shared, key=lambda j: len(sets[j][1]))
        rel, gain = relevance(j, p1, p2)
        # players: KGsim2vec players for kg; for seek/hyb, aspects with non-zero KAS or LOO attribution (= SEEK players)
        pl = players(p1, p2) if pipe == "kg" else sorted({A.index(a) for m in ("D", "LOO") for a in r[f"phi_{m}"]})
        nrel = int(sum(rel[i] for i in pl))
        if len(pl) < 2 or nrel == 0 or nrel == len(pl): continue
        rr = random.Random(r["i"])
        orders = {}
        for m in methods:
            phi = r[f"phi_{m}"]; tb = {i: rr.random() for i in pl}
            orders[m] = sorted(pl, key=lambda i: (-phi.get(A[i], 0.0), tb[i]))
        if pipe == "kg":
            tb = {i: rr.random() for i in pl}
            orders["Sim"] = sorted(pl, key=lambda i: (-Xpy[r["i"], i], tb[i]))
        for m, o in orders.items():
            res[m]["hit1"].append(float(rel[o[0]])); res[m]["p3"].append(float(np.mean([rel[a] for a in o[:3]])))
            res[m]["ndcg3"].append(ndcg3(o, gain))
        res["random"]["hit1"].append(nrel / len(pl))
        perms = [rr.sample(pl, len(pl)) for _ in range(200)]
        res["random"]["p3"].append(float(np.mean([np.mean([rel[a] for a in o[:3]]) for o in perms])))
        res["random"]["ndcg3"].append(float(np.mean([ndcg3(o, gain) for o in perms])))
        if cfg == "kg|rf":
            cases.append(dict(pair=r["pair"], pathway=sets[j][0], size=len(sets[j][1]), players=[kg.label.get(A[i], A[i]) for i in pl],
                              relevant=[kg.label.get(A[i], A[i]) for i in pl if rel[i]],
                              top3={m: [kg.label.get(A[i], A[i]) for i in o[:3]] for m, o in orders.items()}))
    summ = dict(n_tp=n_tp, n_shared=n_shared, n_informative=len(res["random"]["hit1"]), metrics={}, wilcoxon_vs_D={})
    for m, d in res.items():
        summ["metrics"][m] = {k: dict(mean=float(np.mean(v)),
                                      ci95=[float(x) for x in np.percentile([np.mean(np.random.default_rng(b).choice(v, len(v))) for b in range(1000)], [2.5, 97.5])])
                              for k, v in d.items()}
    for m in res:
        if m == "D": continue
        for k in ("hit1", "ndcg3"):
            a, b = np.array(res["D"][k]), np.array(res[m][k])
            summ["wilcoxon_vs_D"][f"{m}|{k}"] = float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
    out[cfg] = summ
    print(cfg, "TP", n_tp, "shared", n_shared, "informative", summ["n_informative"])
    for m in summ["metrics"]:
        mm = summ["metrics"][m]
        print("   %-16s hit@1 %.3f  p@3 %.3f  ndcg@3 %.3f" % (m, mm["hit1"]["mean"], mm["p3"]["mean"], mm["ndcg3"]["mean"]),
              "" if m == "D" else "p(hit1)=%.2g p(ndcg)=%.2g" % (summ["wilcoxon_vs_D"].get(f"{m}|hit1", 1), summ["wilcoxon_vs_D"].get(f"{m}|ndcg3", 1)))
json.dump(dict(gmt=GMT, results=out, cases=cases), open(f"plaus_{GMT}.json", "w"))
