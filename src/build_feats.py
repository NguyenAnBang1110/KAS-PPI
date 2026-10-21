import sys,time,numpy as np,pickle
sys.path.insert(0,".")
from kgsim import KG
R="ExplainablePPI/Experimental_Results/"
aspects=[l.split("\t")[0].split("/")[-1] for l in open(R+"Results/kgsim2vec_sas.txt") if l.strip()]
kg=KG("godata/go-basic.owl","godata/goa_human.gaf",aspects)
rows=[l.rstrip("\n").split("\t") for l in open(R+"Results/ss_resnik_max_ICseco.txt")]
lab={}
for l in open(R+"Data/v11(score950).txt"):
    f=l.split(); lab[(f[0],f[1])]=int(f[2])
keys=[(r[0].replace("http://",""),r[1].replace("http://","")) for r in rows]
Xref=np.array([[float(x) for x in r[2:]] for r in rows])
t=time.time()
Xpy=np.array([kg.features(kg.ann[a],kg.ann[b]) for a,b in keys])
print("built",time.time()-t)
y=np.array([lab[k] for k in keys])
np.savez("feats.npz",Xref=Xref,Xpy=Xpy,y=y,keys=np.array(keys),aspects=np.array(aspects))
d=np.abs(Xref-Xpy)
print("exact pairs",(d.max(1)<1e-6).mean(),"cells within 1e-6",(d<1e-6).mean(),"within 0.05",(d<0.05).mean())
print("zero-pattern agreement",((Xref>0)==(Xpy>0)).mean())
cors=[np.corrcoef(Xref[:,j],Xpy[:,j])[0,1] for j in range(52) if Xref[:,j].std()>0 and Xpy[:,j].std()>0]
print("per-aspect pearson median %.4f min %.4f"%(np.median(cors),np.min(cors)))
