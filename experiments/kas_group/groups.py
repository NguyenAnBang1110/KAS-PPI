"""Annotation-group players for KAS-group.
Each GAF line gets one group under a partition; an annotation (protein, term) survives the removal of a set S of
groups iff at least one of its supporting GAF lines belongs to a group outside S.
Partitions:
  source   : column 15 assigned_by -> Reactome, IntAct, UniProt, GO_Central, Ensembl, BHF-UCL, Other
  evidence : column 7 evidence code -> EXP (experimental incl. high-throughput), PHYLO (IBA/IBD/IKR/IRD),
             COMP (ISS/ISO/ISA/ISM/IGC/RCA), STATEMENT (TAS/NAS/IC/ND), IEA"""
import collections
SOURCES = ["Reactome", "IntAct", "UniProt", "GO_Central", "Ensembl", "BHF-UCL", "Other"]
EVID = {"EXP": "EXP IDA IPI IMP IGI IEP HTP HDA HMP HGI HEP".split(),
        "PHYLO": "IBA IBD IKR IRD".split(),
        "COMP": "ISS ISO ISA ISM IGC RCA".split(),
        "STATEMENT": "TAS NAS IC ND".split(),
        "IEA": ["IEA"]}
EV2G = {c: g for g, cs in EVID.items() for c in cs}
GROUPS = {"source": SOURCES, "evidence": list(EVID)}
def group_of(part, f):
    if part == "source":
        return f[14] if f[14] in SOURCES else "Other"
    return EV2G.get(f[6], "COMP")
def load(gaf, part, drop_not):
    """prot -> term -> frozenset(group index)"""
    G = GROUPS[part]; gi = {g: i for i, g in enumerate(G)}
    out = collections.defaultdict(lambda: collections.defaultdict(set))
    for l in open(gaf):
        if l.startswith("!"): continue
        f = l.rstrip("\n").split("\t")
        if len(f) < 15: continue
        if drop_not and "NOT" in f[3]: continue
        out[f[1]]["GO_" + f[4].split(":")[1]].add(gi[group_of(part, f)])
    return {p: {t: frozenset(s) for t, s in d.items()} for p, d in out.items()}
def ann(m, p, S=frozenset()):
    return {t for t, s in m.get(p, {}).items() if s - S}
