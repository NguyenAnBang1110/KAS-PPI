"""E4: accuracy of the 100-permutation Shapley estimate used by KAS when k > 8. On pairs with 6 <= k <= 8 (exact is
affordable) compare exact KAS with permutation estimates (M = 25, 50, 100, 200; 3 seeds each), same value function.
usage: python3 perm_check.py PIPE MODEL FOLD N    -> perm_{PIPE}_{MODEL}_f{FOLD}.json"""
import sys, json, time
import numpy as np
from scipy.stats import spearmanr
PIPE_, MODEL_, FOLD_, N_ = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sys.argv = ["x", PIPE_, MODEL_, str(FOLD_), "1"]
exec(open("explain_v2.py").read().split("pos=[i for i in test")[0])
rs = np.random.default_rng(1000 + FOLD_); cand = list(rs.permutation(test)); res = []; t0 = time.time()
for i in cand:
    if len(res) >= N_: break
    p1, p2 = keys[i]
    st = (set(kg.ann[p1]), set(kg.ann[p2]), set(sk.ann[p1]) - CX, set(sk.ann[p2]) - CX)
    pl = players_of(st)
    if not 6 <= len(pl) <= 8: continue
    cache = {}
    def vb(subs):
        need = [S for S in subs if S not in cache]
        if need:
            for S, v in zip(need, proba(np.array([feats(remove(st, S)) for S in need]))): cache[S] = v
        return [cache[S] for S in subs]
    EXACT_MAX = 8; ex = shapley(vb, pl, 0)
    pred = int(cache[frozenset()] > 0.5); sgn = 1 if pred else -1
    top = lambda f: [int(j) for j in np.argsort(-sgn * f) if j in pl]
    o_ex = top(ex); scale = np.abs(ex[pl]).max() or 1.0
    row = dict(i=int(i), k=len(pl))
    for M in (25, 50, 100, 200):
        for sd in range(3):
            EXACT_MAX = 0; M_PERM = M; est = shapley(vb, pl, 17 * sd + M)
            o = top(est)
            row[f"M{M}_s{sd}"] = dict(rel_mae=float(np.abs(est[pl] - ex[pl]).mean() / scale),
                                      rho=float(spearmanr(est[pl], ex[pl]).correlation) if np.std(ex[pl]) > 0 and np.std(est[pl]) > 0 else 1.0,
                                      top1=int(o[0] == o_ex[0]), jac3=len(set(o[:3]) & set(o_ex[:3])) / len(set(o[:3]) | set(o_ex[:3])),
                                      sum_err=float(abs(est.sum() - ex.sum())))
    EXACT_MAX = 8; M_PERM = 100
    res.append(row)
    if len(res) % 50 == 0: print(len(res), round(time.time() - t0), "s", flush=True)
summ = {}
for M in (25, 50, 100, 200):
    vals = [r[f"M{M}_s{sd}"] for r in res for sd in range(3)]
    summ[M] = {k: float(np.mean([v[k] for v in vals])) for k in ("rel_mae", "rho", "top1", "jac3", "sum_err")}
    print(PIPE_, MODEL_, "M=%d" % M, " ".join("%s %.3f" % kv for kv in summ[M].items()))
json.dump(dict(pipe=PIPE_, model=MODEL_, fold=FOLD_, n=len(res), summary=summ, rows=res), open(f"perm_{PIPE_}_{MODEL_}_f{FOLD_}.json", "w"))
