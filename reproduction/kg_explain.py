"""Giai thich + danh gia size/informativeness cua KGsim2vec (Sousa24, Fig. 4) tren fold 1, dung code goc cua repo.
Chay trong thu muc repo ExplainablePPI: python kg_explain.py <ss_file> <names_file> <N_LIME> <out_dir>
names_file: ten aspect (dang get_semantic_aspects tra ve), moi dong mot ten, dung thu tu cot cua ss_file."""
import sys, os, json, time, statistics, numpy as np
sys.path.insert(0, '.')
# gplearn_variations (vendored, viet cho sklearn 0.21) con import sklearn.externals.six/joblib da bi go khoi sklearn moi
import six, joblib, sklearn.externals
sklearn.externals.six = six; sys.modules['sklearn.externals.six'] = six
sklearn.externals.joblib = joblib; sys.modules['sklearn.externals.joblib'] = joblib
import supervised_learning as SL, evaluation_explanations as EV
from Prediction import ML, LIMEexplanations
from sklearn.tree import export_text as _et
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
ML.export_text = lambda clf, names: _et(clf, feature_names=list(names))  # sklearn moi: feature_names la keyword-only
ss_file, names_file, N, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
os.makedirs(out, exist_ok=True)
D = 'Experimental_Results/Data/'
aspects = [l.rstrip('\n') for l in open(names_file) if l.strip()]
labels, pairs = SL.process_dataset(D + 'v11(score950).txt')
dict_SS = SL.process_SS_file(ss_file)
X = np.array([dict_SS[(a, b)] for a, b in pairs]); y = np.array(labels); pairs = np.array(pairs)
te = SL.process_indexes_partition(D + 'StratifiedPartitions/Indexes__crossvalidationTest__Run1(complete).txt')
tr = sorted(set(range(len(y))) - set(te))
Xtr, Xte, ytr, yte = list(X[tr]), list(X[te]), list(y[tr]), list(y[te]); pte = list(pairs[te])
dic_ics = EV.process_IC(D + 'ICSeco_GOterms.txt'); n2u = EV.GOurl2GOname(D + 'go-basic.owl', aspects)
res = {}; t0 = time.time()
for alg, d in [('DT6', 6), ('DT', None)]:
    f = out + f'/{alg}_'
    ML.performance_DT(dict_SS, pte, Xtr, Xte, ytr, yte, aspects, f + 'pred.txt', f + 'model.txt', f + 'eval.txt', f + 'expl.txt', d=d)
    EV.process_models(f + 'expl.txt', f + 'evalexpl.txt', dic_ics, n2u, aspects)
    last = open(f + 'evalexpl.txt').read().strip().split('\n')[-1].split('\t')
    res[alg] = {'median_SumIC': float(last[2]), 'median_AvgIC': float(last[3]), 'median_size': float(last[4]),
                'WAF_fold1': float(open(f + 'eval.txt').readline().split('\t')[1])}
    print(alg, res[alg], round(time.time() - t0), 's', flush=True)
rng = np.random.RandomState(0); idx = sorted(rng.choice(len(yte), min(N, len(yte)), replace=False))
for alg, clf in [('RF', RandomForestClassifier(random_state=0)), ('XGB', xgb.XGBClassifier(random_state=0))]:
    clf.fit(np.array(Xtr), np.array(ytr))
    s = {3: [], 8: []}
    for i in idx:
        for k in (3, 8):
            rule = LIMEexplanations.LIME_explanation(Xtr, Xte, i, clf, aspects, k)
            s[k].append(EV.process_rule(rule, aspects, dic_ics, n2u))
    res[alg] = {f'LIME{k}': {'median_SumIC': statistics.median(v[0] for v in s[k]), 'median_AvgIC': statistics.median(v[1] for v in s[k]),
                             'median_size': statistics.median(v[2] for v in s[k])} for k in (3, 8)}
    res[alg]['n_pairs'] = len(idx)
    print(alg, res[alg], round(time.time() - t0), 's', flush=True)
json.dump(res, open(out + '/kg_explain_summary.json', 'w'), indent=1)
