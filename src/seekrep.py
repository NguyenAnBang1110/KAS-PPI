"""Python re-implementation of SEEK pair representations (Sousa et al., KR 2023):
pair vector = average RDF2Vec embedding of the disjoint common ancestors (DCA) of the two proteins' GO annotations.
Class embeddings are the authors' precomputed ones (fixed); only the DCA set changes when annotations are removed."""
import re, json, collections, numpy as np
from kgsim import parse_go
class SEEK:
    def __init__(s, owl, gaf, emb_file):
        s.parents, s.label = parse_go(owl)
        s.children = collections.defaultdict(set)
        for c, ps in s.parents.items():
            for p in ps: s.children[p].add(c)
        # SEEK's build_kg reads every GAF line (qualifier column ignored), protein id = column 2
        s.ann = collections.defaultdict(set)
        with open(gaf) as f:
            f.readline()
            for l in f:
                if l.startswith("!"): continue
                c = l.split("\t")
                s.ann[c[1]].add("GO_" + c[4].split(":")[1])
        txt = open(emb_file).read().replace("'", '"')
        raw = json.loads(re.sub(r'"http://purl.obolibrary.org/obo/', '"', txt))
        s.emb = {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}
        s.dim = len(next(iter(s.emb.values())))
        s._anc = {}
    def anc(s, c):
        if c in s._anc: return s._anc[c]
        out = {c}; st = [c]
        while st:
            x = st.pop()
            for y in s.parents.get(x, ()):
                if y not in out: out.add(y); st.append(y)
        s._anc[c] = frozenset(out); return s._anc[c]
    def desc(s, c, _cache={}):
        if c in _cache: return _cache[c]
        out = set(); st = [c]
        while st:
            x = st.pop()
            for y in s.children.get(x, ()):
                if y not in out: out.add(y); st.append(y)
        _cache[c] = out; return out
    def dca(s, t1, t2):
        A1 = set().union(*[s.anc(t) for t in t1]) if t1 else set()
        A2 = set().union(*[s.anc(t) for t in t2]) if t2 else set()
        common = A1 & A2
        strict = set()
        for d in common: strict |= (s.anc(d) - {d})
        return [c for c in common - strict if c in s.emb]
    def rep(s, t1, t2):
        d = s.dca(t1, t2)
        if not d: return np.zeros(s.dim)
        return np.mean([s.emb[c] for c in d], 0)
