"""10-fold prediction (repo partitions) with KGsim2vec features from kgsim.py (current) vs kgsim_fix.py (= SML exactly), 51 aspects."""
import numpy as np, json
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon
K=np.load("feats_kg.npz"); y=K["y"]; asp=list(K["aspects"]); keep=[j for j in range(52) if asp[j]!="GO_0032991"]
Xs={"current":K["Xpy"][:,keep],"fixed":np.load("../leak/Xpy_fix.npy")[:,keep]}
P="ExplainablePPI/Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run"
folds=[[int(l) for l in open(P+("1(complete)" if f==1 else str(f))+".txt") if l.strip()] for f in range(1,11)]
out={}
for mn,mk in [("RF100",lambda:RandomForestClassifier(100,n_jobs=2,random_state=0)),("XGB300",lambda:XGBClassifier(n_estimators=300,n_jobs=2,random_state=0,eval_metric="logloss"))]:
    for xn,X in Xs.items():
        f1=[]
        for te in folds:
            m=np.zeros(len(y),bool); m[te]=True
            f1.append(f1_score(y[m],mk().fit(X[~m],y[~m]).predict(X[m]),average="weighted"))
        out[f"{mn}|{xn}"]=f1; print(mn,xn,"median WAF %.4f"%np.median(f1),flush=True)
    a,b=np.array(out[f"{mn}|current"]),np.array(out[f"{mn}|fixed"]); print("  diff %.4f p %.3g"%(np.median(b)-np.median(a),wilcoxon(a,b).pvalue if np.any(a!=b) else 1))
json.dump(out,open("../leak/nfix_pred.json","w"))
