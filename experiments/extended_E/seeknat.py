"""SEEK's own explainer (Sousa et al. 2023, generate_seek_explanation.py), made graded and mapped to the 51 fixed aspects.
SEEK explains in DCA space: for each disjoint common ancestor c of the pair it asks the model
  'without c'  (average embedding of the other DCAs; zero vector if none)  -> c is NECESSARY if the predicted class flips
  'only c'     (embedding of c alone)                                        -> c is SUFFICIENT if the predicted class stays.
Graded versions used here (toward interaction, i.e. class-1 probability):
  SEEKN(c) = p(all DCAs) - p(without c)      SEEKS(c) = p(only c) - 0.5
Aspect score: SEEKN(a) = sum of SEEKN(c) over the DCAs c lying in aspect a's subtree;
              SEEKS(a) = SEEKS(c*) for the DCA c* in a that most supports the predicted class.
Each DCA of a pair lies only in player aspects (both proteins have descendants of c), so the mapping never leaves the players.
DCAs outside every aspect (e.g. the BP/MF/CC roots) are counted but cannot be attributed."""
import numpy as np

def dca_vec(sk, D):
    return np.mean([sk.emb[c] for c in D], 0) if D else np.zeros(sk.dim)

def native(st, sk, sub_sk, proba, nA):
    s1, s2 = st[2], st[3]
    D = sk.dca(s1, s2)
    Z = [dca_vec(sk, D)] + [dca_vec(sk, [d for d in D if d != c]) for c in D] + [sk.emb[c] for c in D]
    p = proba(np.array(Z)); p0 = p[0]; m = len(D)
    pw, po = p[1:1 + m], p[1 + m:]
    pred = int(p0 > 0.5); sgn = 1 if pred else -1
    nec = [c for c, q in zip(D, pw) if int(q > 0.5) != pred]
    suf = [c for c, q in zip(D, po) if int(q > 0.5) == pred]
    asp_of = {c: [a for a in range(nA) if c in sub_sk[a]] for c in D}
    phiN = np.zeros(nA); phiS = np.zeros(nA); best = np.full(nA, -np.inf)
    for c, q_w, q_o in zip(D, pw, po):
        for a in asp_of[c]:
            phiN[a] += p0 - q_w
            v = q_o - 0.5
            if sgn * v > best[a]: best[a] = sgn * v; phiS[a] = v
    stats = dict(n_dca=m, n_dca_noasp=sum(1 for c in D if not asp_of[c]), n_nec=len(nec), n_suf=len(suf),
                 nec_asp=sorted({a for c in nec for a in asp_of[c]}), suf_asp=sorted({a for c in suf for a in asp_of[c]}))
    return phiN, phiS, stats, D, asp_of

def dca_curves(D, asp_of, order, pl, pred, sk, proba, topk=5):
    """Deletion / insertion in SEEK's own (DCA) space: drop the DCAs that lie in the removed aspects and average the
    rest, without recomputing annotations. Returns predicted-class probabilities (deletion k=1..topk, insertion k=0..topk)."""
    def keep_without(S):
        S = set(S); return [c for c in D if not (set(asp_of[c]) & S)]
    Zd = [dca_vec(sk, keep_without(order[:k])) for k in range(1, topk + 1)]
    Zi = [dca_vec(sk, keep_without([a for a in pl if a not in order[:k]])) for k in range(0, topk + 1)]
    p = proba(np.array(Zd + Zi)); p = p if pred == 1 else 1 - p
    return list(p[:topk]), list(p[topk:])
