"""Python re-implementation of KGsim2vec features (Resnik-max, IC Seco computed inside each aspect subgraph)."""
import re, math, collections, pickle, os
OBO="http://purl.obolibrary.org/obo/GO_"
def parse_go(owl):
    parents=collections.defaultdict(set); label={}; cur=None
    pat_cls=re.compile(r'<owl:Class rdf:about="http://purl.obolibrary.org/obo/(GO_\d+)">')
    pat_sub=re.compile(r'<rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/(GO_\d+)"/>')
    pat_lab=re.compile(r'<rdfs:label[^>]*>([^<]*)</rdfs:label>')
    depth=0
    for line in open(owl,encoding="utf-8"):
        m=pat_cls.search(line)
        if m: cur=m.group(1); continue
        if cur is None: continue
        if "</owl:Class>" in line and line.strip()=="</owl:Class>": cur=None; continue
        m=pat_sub.search(line)
        if m: parents[cur].add(m.group(1)); continue
        m=pat_lab.search(line)
        if m and cur not in label: label[cur]=m.group(1)
    return parents,label
def load_gaf(gaf):
    ann=collections.defaultdict(set)
    for l in open(gaf):
        if l.startswith("!"): continue
        f=l.split("\t")
        if len(f)<5 or "NOT" in f[3]: continue
        ann[f[1]].add("GO_"+f[4].split(":")[1])
    return ann
class KG:
    def __init__(s,owl,gaf,aspects):
        s.parents,s.label=parse_go(owl)
        s.children=collections.defaultdict(set)
        for c,ps in s.parents.items():
            for p in ps: s.children[p].add(c)
        s.ann=load_gaf(gaf)
        s.aspects=aspects
        s._anc={}
        s.sub=[]; s.ic=[]
        for a in aspects:
            members=s.desc(a)|{a}
            nprot=sum(1 for p,ts in s.ann.items() if ts & members)
            N=len(members)+nprot   # SML counts annotated instances as vertices of the reduced graph
            ic={}
            for c in members:
                d=len(s.desc(c))
                ic[c]=1-math.log(d+1)/math.log(N) if N>1 else 1.0
            s.sub.append(members); s.ic.append(ic)
    def desc(s,c,_cache={}):
        if c in _cache: return _cache[c]
        out=set(); st=[c]
        while st:
            x=st.pop()
            for y in s.children.get(x,()):
                if y not in out: out.add(y); st.append(y)
        _cache[c]=out; return out
    def anc(s,c):
        if c in s._anc: return s._anc[c]
        out={c}; st=[c]
        while st:
            x=st.pop()
            for y in s.parents.get(x,()):
                if y not in out: out.add(y); st.append(y)
        s._anc[c]=out; return out
    def closure(s,terms):
        out=set()
        for t in terms: out|=s.anc(t)
        return out
    def features(s,t1,t2):
        """t1,t2: sets of direct annotation terms."""
        A1=s.closure(t1); A2=s.closure(t2)
        common=A1&A2
        v=[]
        for i,a in enumerate(s.aspects):
            m=s.sub[i]
            # protein needs >=1 direct annotation in aspect
            if not (t1 & m) or not (t2 & m): v.append(0.0); continue
            cm=common & m
            v.append(max((s.ic[i][c] for c in cm),default=0.0))
        return v
