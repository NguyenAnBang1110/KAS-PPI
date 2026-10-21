"""So file SML vua chay voi file tac gia commit (Experimental_Results/Results/), can cot theo URL aspect.
python kg_compare_sml.py <new_ss> <new_aspect_urls> <authors_ss> <authors_sas>"""
import sys, numpy as np
def load(p):
    d = {}
    for l in open(p):
        t = l.rstrip('\n').split('\t'); d[(t[0].split('/')[-1], t[1].split('/')[-1])] = np.array(t[2:], float)
    return d
new, old = load(sys.argv[1]), load(sys.argv[3])
un = [l.split('\t')[0].strip() for l in open(sys.argv[2]) if l.strip()]
uo = [l.split('\t')[0].strip() for l in open(sys.argv[4]) if l.strip()]
print('so aspect: moi', len(un), '| tac gia', len(uo), '| cung tap aspect:', set(un) == set(uo))
perm = [un.index(u) for u in uo]
ks = [k for k in old if k in new]
A = np.array([new[k][perm] for k in ks]); B = np.array([old[k] for k in ks])
d = np.abs(A - B)
print('so cap: moi', len(new), '| tac gia', len(old), '| chung', len(ks))
print('ty le o lech > 1e-9: %.6f | lech lon nhat: %.3g' % ((d > 1e-9).mean(), d.max()))
