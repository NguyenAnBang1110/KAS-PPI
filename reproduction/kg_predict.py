"""Tai hien bang 1 / bang 10 Sousa24: 10-fold, DT/DT6/RF/XGB (tham so mac dinh nhu Prediction/ML.py).
Dung: python kg_predict.py <repo_dir> <ss_file> <aspect_file> <out_json> [algs] [n_folds]
aspect_file: moi dong 'url<TAB>ten', cung thu tu cot voi ss_file."""
import sys, json, time, numpy as np
from statistics import median
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import xgboost as xgb
repo, ss_file, asp_file, out = sys.argv[1:5]
algs = sys.argv[5].split(',') if len(sys.argv) > 5 else ['DT6', 'DT', 'RF', 'XGB']
nf = int(sys.argv[6]) if len(sys.argv) > 6 else 10
D = repo + '/Experimental_Results/Data/'
pairs, y = [], []
for l in open(D + 'v11(score950).txt'):
    t = l.rstrip('\n').split('\t'); pairs.append((t[0], t[1])); y.append(int(t[2]))
y = np.array(y)
ss = {}
for l in open(ss_file):
    t = l.rstrip('\n').split('\t'); ss[(t[0].split('/')[-1], t[1].split('/')[-1])] = [float(v) for v in t[2:]]
X = np.array([ss[p] for p in pairs])
asp = [l.split('\t')[0].strip() for l in open(asp_file) if l.strip()]
assert len(asp) == X.shape[1], (len(asp), X.shape)
CX = 'http://purl.obolibrary.org/obo/GO_0032991'
keep51 = [i for i, a in enumerate(asp) if a != CX]
def fold_idx(r):
    f = D + 'StratifiedPartitions/Indexes__crossvalidationTest__Run' + ('1(complete)' if r == 1 else str(r)) + '.txt'
    return [int(l) for l in open(f) if l.strip()]
def model(a, seed):
    if a == 'DT': return DecisionTreeClassifier(random_state=seed)
    if a == 'DT6': return DecisionTreeClassifier(max_depth=6, random_state=seed)
    if a == 'RF': return RandomForestClassifier(random_state=seed, n_jobs=-1)
    if a == 'XGB': return xgb.XGBClassifier(random_state=seed, n_jobs=-1)
res = {}
for cfg, cols in [('52_aspects_repo', list(range(X.shape[1]))), ('51_aspects_no_complex', keep51)]:
    for a in algs:
        w = []
        for r in range(1, nf + 1):
            te = fold_idx(r); m = np.ones(len(y), bool); m[te] = False
            clf = model(a, 0).fit(X[m][:, cols], y[m])
            w.append(f1_score(y[te], clf.predict(X[te][:, cols]), average='weighted'))
        q1, q3 = np.percentile(w, [25, 75])
        res[cfg + '/' + a] = {'median_WAF': round(median(w), 4), 'IQR': round(q3 - q1, 4), 'folds': [round(v, 4) for v in w]}
        print(cfg, a, res[cfg + '/' + a]['median_WAF'], 'IQR', res[cfg + '/' + a]['IQR'], flush=True)
json.dump(res, open(out, 'w'), indent=1)
