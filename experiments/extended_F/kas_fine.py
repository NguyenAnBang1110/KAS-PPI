"""F4: KAS at a finer granularity, with the trained model unchanged (KAS = one method parameterised by the partition).
Players = the direct-child subtrees of each of the 51 aspects (GO depth 2) + one residual unit per aspect holding the
aspect term itself. A unit is a player of a pair when it lies in an aspect-level player and holds >=1 direct annotation of either
protein (so removing every fine player removes exactly what removing every aspect player removes, and the fine
attributions sum to the same total as aspect-level KAS). Value function as KAS: delete both proteins' annotations in the removed units, recompute the 51 KGsim2vec
features, read the model. Exact Shapley for k<=8, else 100 permutations. Compared with LOO and random at the same
granularity (deletion/insertion, operator ks), and checked for consistency with aspect-level KAS (sum of fine phi
inside an aspect vs the aspect's phi).
usage: python3 kas_fine.py MODEL FOLD N   -> f09/fine_kg_{MODEL}_f{FOLD}.json"""
import sys, json, time, random, itertools, math
import numpy as np
MODEL, FOLD, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sys.argv = ["x", "kg", MODEL, str(FOLD), "1"]
exec(open("explain_v2.py").read().split("pos=[i for i in test")[0])
from scipy.stats import spearmanr
units, unit_asp, unit_name = [], [], []
for ai, a in enumerate(A):
    for c in sorted(kg.children.get(a, ())):
        if c in sub_kg[ai]: units.append(kg.desc(c) | {c}); unit_asp.append(ai); unit_name.append(c)
    units.append({a}); unit_asp.append(ai); unit_name.append(a + "(self)")
U = len(units); print("fine units", U, flush=True)
def players_f(st):   # units inside an aspect-level player that hold >=1 direct annotation of either protein
    k1, k2 = st[0], st[1]; pa = set(players_of(st)); return [u for u in range(U) if unit_asp[u] in pa and ((k1 | k2) & units[u])]
def remove_f(st, S):
    if not S: return st
    r = set().union(*[units[u] for u in S]); return (st[0] - r, st[1] - r, st[2], st[3])
def shapley_f(st, pl, seed):
    k = len(pl); phi = np.zeros(U)
    if k == 0: return phi
    vb = lambda subs: proba(np.array([feats(remove_f(st, S)) for S in subs]))
    if k <= EXACT_MAX:
        subsets = [frozenset(c) for r in range(k + 1) for c in itertools.combinations(pl, r)]; vals = dict(zip(subsets, vb(subsets)))
        for i in pl:
            others = [p for p in pl if p != i]
            for r in range(k):
                w = math.factorial(r) * math.factorial(k - r - 1) / math.factorial(k)
                for c in itertools.combinations(others, r): S = frozenset(c); phi[i] += w * (vals[S] - vals[S | {i}])
        return phi
    rr = random.Random(seed); perms = [rr.sample(pl, k) for _ in range(M_PERM)]; subsets = set()
    for p in perms:
        for t in range(k + 1): subsets.add(frozenset(p[:t]))
    subsets = list(subsets); vals = dict(zip(subsets, vb(subsets)))
    for p in perms:
        for t, i in enumerate(p): phi[i] += vals[frozenset(p[:t])] - vals[frozenset(p[:t + 1])]
    return phi / M_PERM
def curves(st, order, pred, pl):
    Zd = [feats(remove_f(st, order[:k])) for k in range(1, TOPK + 1)]
    Zi = [feats(remove_f(st, [u for u in pl if u not in order[:k]])) for k in range(0, TOPK + 1)]
    p = proba(np.array(Zd + Zi)); p = p if pred == 1 else 1 - p
    return list(p[:TOPK]), list(p[TOPK:])
R = json.load(open(f"rec_kg_{MODEL}_f{FOLD}_" + ("400" if FOLD == 1 else "200") + ".json"))["records"]
h = len(R) // 2; R = R[:N // 2] + R[h:h + N // 2]; out = []; t0 = time.time()
for n, r in enumerate(R):
    p1, p2 = keys[r["i"]]; st = (set(kg.ann[p1]), set(kg.ann[p2]), set(), set()); pred = r["pred"]
    pl = players_f(st); ts = time.time(); phi = shapley_f(st, pl, 0); tk = time.time() - ts
    v = proba(np.array([feats(st)] + [feats(remove_f(st, [u])) for u in pl])); loo = np.zeros(U)
    for j, u in enumerate(pl): loo[u] = v[0] - v[j + 1]
    o = dict(i=r["i"], y=r["y"], pred=pred, k_aspect=r["k"], k_fine=len(pl), time_fine=tk, time_aspect=r["time_D"])
    for nm, f in (("Dfine", phi), ("LOOfine", loo)):
        od = ranking(f, pred); o[f"del_{nm}"], o[f"ins_{nm}"] = curves(st, od, pred, pl) if od else (None, None)
        o[f"phi_{nm}"] = {unit_name[u]: round(float(f[u]), 5) for u in np.nonzero(f)[0]}
    ro = [int(x) for x in np.random.default_rng(r["i"]).permutation(pl)]
    o["del_randomfine"], o["ins_randomfine"] = curves(st, ro, pred, pl) if ro else (None, None)
    # consistency with aspect-level KAS: aggregate fine phi to aspects
    agg = np.zeros(len(A))
    for u in pl: agg[unit_asp[u]] += phi[u]
    da = np.array([r["phi_D"].get(a, 0.0) for a in A]); asp_pl = sorted({unit_asp[u] for u in pl})
    o["rho_agg_vs_aspect"] = float(spearmanr(agg[asp_pl], da[asp_pl]).correlation) if len(asp_pl) >= 3 and np.std(agg[asp_pl]) > 0 and np.std(da[asp_pl]) > 0 else None
    oa, of = ranking(da, pred), ranking(phi, pred)
    o["top1_fine_in_top1_aspect"] = int(bool(oa) and bool(of) and unit_asp[of[0]] == oa[0]) if oa and of else None
    o["top1_fine"] = unit_name[of[0]] if of else None; o["top1_fine_label"] = kg.label.get(unit_name[of[0]].split("(")[0], "") if of else None
    o["sum_fine"], o["sum_aspect"] = float(phi.sum()), float(da.sum())
    out.append(o)
    if n % 10 == 0: print(MODEL, FOLD, n, "k_fine", len(pl), round(time.time() - t0), "s", flush=True)
json.dump(dict(model=MODEL, fold=FOLD, n_units=U, records=out), open(f"f09/fine_kg_{MODEL}_f{FOLD}.json", "w"))
print("done", round(time.time() - t0))
