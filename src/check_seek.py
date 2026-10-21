import numpy as np, time
from seekrep import SEEK
S="seek/PPI/"
t=time.time(); sk=SEEK(S+"go.owl",S+"go_annotations.gaf",S+"Embeddings/Emb_classes_maxdepth4_nwalks100.txt"); print("load",time.time()-t, len(sk.emb))
ref={}
for l in open(S+"Embeddings/Emb_pair_maxdepth4_nwalks100_Avg_disjointcommonancestor.txt"):
    f=l.rstrip("\n").split("\t"); ref[(f[0],f[1])]=np.array(f[2:],float)
pairs=[l.split("\t")[:2] for l in open(S+"STRING_pairs.txt")]
rng=np.random.default_rng(0); idx=rng.choice(len(pairs),1000,replace=False)
err=[];t=time.time()
for i in idx:
    a,b=pairs[i]; v=sk.rep(sk.ann[a],sk.ann[b]); err.append(np.abs(v-ref[(a,b)]).max())
err=np.array(err); print("time/pair",(time.time()-t)/1000, "exact(1e-6)",(err<1e-6).mean(), "max",err.max(), "p99",np.quantile(err,.99))
