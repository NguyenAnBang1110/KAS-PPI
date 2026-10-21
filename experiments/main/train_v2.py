"""10-fold prediction table for the three pipelines (repo partitions), with and without the circular aspect."""
import numpy as np, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier
from scipy.stats import wilcoxon
K=np.load("feats_kg.npz"); S=np.load("feats_seek.npz"); y=K["y"]; aspects=list(K["aspects"])
ci=aspects.index("GO_0032991"); keep=[j for j in range(52) if j!=ci]
P="ExplainablePPI/Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run"
folds=[[int(l) for l in open(P+("1(complete)" if r==1 else str(r))+".txt") if l.strip()] for r in range(1,11)]
feats={"KG_52":K["Xpy"],"KG_51":K["Xpy"][:,keep],"SEEK_52":S["Xseek"],"SEEK_51":S["Xseek51"],
       "HYB_52":np.hstack([K["Xpy"],S["Xseek"]]),"HYB_51":np.hstack([K["Xpy"][:,keep],S["Xseek51"]])}
out={}
for mname in ["RF100","XGB300"]:
    for fname,X in feats.items():
        f1s=[];aucs=[]
        for test in folds:
            m=np.zeros(len(y),bool); m[test]=True
            clf=RandomForestClassifier(100,n_jobs=-1,random_state=0) if mname=="RF100" else XGBClassifier(n_estimators=300,n_jobs=-1,random_state=0,eval_metric="logloss")
            clf.fit(X[~m],y[~m]); p=clf.predict_proba(X[m])[:,1]
            f1s.append(f1_score(y[m],(p>0.5).astype(int),average="weighted")); aucs.append(roc_auc_score(y[m],p))
        out[f"{fname}|{mname}"]=dict(median_waf=float(np.median(f1s)),iqr=float(np.percentile(f1s,75)-np.percentile(f1s,25)),median_auc=float(np.median(aucs)),folds=[round(x,4) for x in f1s],aucs=[round(x,4) for x in aucs])
        print(fname,mname,"WAF %.4f IQR %.4f AUC %.4f"%(np.median(f1s),out[f"{fname}|{mname}"]["iqr"],np.median(aucs)),flush=True)
    for a,b in [("HYB_51","KG_51"),("KG_51","SEEK_51"),("HYB_51","SEEK_51"),("SEEK_52","SEEK_51")]:
        fa=np.array(out[f"{a}|{mname}"]["folds"]); fb=np.array(out[f"{b}|{mname}"]["folds"])
        out[f"wilcoxon|{mname}|{a}_vs_{b}"]=dict(mean_diff=float((fa-fb).mean()),wins=int((fa>fb).sum()),p=float(wilcoxon(fa,fb).pvalue))
        print(mname,a,"vs",b,out[f"wilcoxon|{mname}|{a}_vs_{b}"],flush=True)
json.dump(out,open("results_prediction_v2.json","w"),indent=1)
