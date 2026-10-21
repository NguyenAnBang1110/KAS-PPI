"""K1: KGsim2vec GP / GP6x (Sousa24 Table 1, Table 10) with the repo's own code (Prediction/ML.performance_GP[6x],
vendored gplearn_variations, parameters of ML.py: population 500, 50 generations, ...).
usage (from ExplainablePPI/): python3 ../k10/k_gp.py ALG FOLD SEED FEATS OUTDIR
FEATS: npz with X (47142 x n, pair order of feats_kg.npz) and names (aspect labels as get_semantic_aspects returns them).
The repo leaves random_state=None (non-deterministic); we set ML.random_state_value=SEED so runs are repeatable."""
import sys, os, json, time, numpy as np
# shims for the repo's numpy<1.20 code (no algorithm change)
for _a, _t in (('int', int), ('float', float), ('bool', bool), ('object', object)):
    if not hasattr(np, _a): setattr(np, _a, _t)
sys.path.insert(0, '.')
import six, joblib, sklearn.externals
sklearn.externals.six = six; sys.modules['sklearn.externals.six'] = six
sklearn.externals.joblib = joblib; sys.modules['sklearn.externals.joblib'] = joblib
from Prediction import ML
ALG, FOLD, SEED, FEATS, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
ML.random_state_value = SEED; ML.verbose_value = 0
F = np.load(FEATS); X = F['X']; names = [str(n) for n in F['names']]
K = np.load(os.path.join(os.environ.get("KGSIM_WORKDIR", "work"), 'feats_kg.npz')); y = K['y']; keys = [tuple(k) for k in K['keys']]
D = 'Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run'
te = [int(l) for l in open(D + ('1(complete)' if FOLD == 1 else str(FOLD)) + '.txt') if l.strip()]
m = np.zeros(len(y), bool); m[te] = True
os.makedirs(OUT, exist_ok=True); p = f'{OUT}/{ALG}_f{FOLD}_s{SEED}_'
t0 = time.time()
fn = ML.performance_GP if ALG == 'GP' else ML.performance_GP6x
fn([keys[i] for i in te], list(X[~m]), list(X[m]), list(y[~m]), list(y[m]), names,
   p + 'pred.txt', p + 'model.txt', p + 'eval.txt', p + 'expl.txt')
waf = float(open(p + 'eval.txt').readline().split('\t')[1])
json.dump(dict(alg=ALG, fold=FOLD, seed=SEED, feats=FEATS, WAF=waf, seconds=round(time.time() - t0),
               model=open(p + 'model.txt').read()), open(p + 'summary.json', 'w'), indent=1)
os.remove(p + 'expl.txt') if FOLD != 1 else None
print(ALG, FOLD, SEED, 'WAF', round(waf, 4), round(time.time() - t0), 's', flush=True)
