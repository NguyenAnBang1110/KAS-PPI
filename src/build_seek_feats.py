import numpy as np, time
from seekrep import SEEK
S="seek/PPI/"
sk=SEEK(S+"go.owl",S+"go_annotations.gaf",S+"Embeddings/Emb_classes_maxdepth4_nwalks100.txt")
K=np.load("feats_kg.npz"); aspects=list(K["aspects"]); keys=[tuple(k) for k in K["keys"]]
roots=["GO_0008150","GO_0003674","GO_0005575"]
ch=set().union(*[sk.children[r] for r in roots])
print("aspects in SEEK GO:",len(ch),"overlap with KG aspects",len(ch&set(aspects)),"missing",set(aspects)-ch)
cx=sk.desc("GO_0032991")|{"GO_0032991"}
Xs=np.zeros((len(keys),sk.dim)); Xs51=np.zeros_like(Xs); t=time.time()
for i,(a,b) in enumerate(keys):
    t1,t2=sk.ann[a],sk.ann[b]
    Xs[i]=sk.rep(t1,t2); Xs51[i]=sk.rep(t1-cx,t2-cx)
print("built",time.time()-t, "rows changed by complex removal",(np.abs(Xs-Xs51).max(1)>0).mean())
np.savez("feats_seek.npz",Xseek=Xs,Xseek51=Xs51)
