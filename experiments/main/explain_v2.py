"""Direction D, full basic experiment: knowledge-space aspect Shapley (D) on three pipelines.
usage: python3 explain_v2.py PIPE MODEL FOLD N    PIPE in {kg,seek,hyb}, MODEL in {rf,xgb}
Players = GO aspects (51 direct children of the 3 GO roots, 'protein-containing complex' removed) in which
both proteins have >=1 direct annotation. Removing a player = deleting both proteins' annotations under that
aspect and recomputing the features through the pipeline (KGsim2vec: Resnik-max/IC-Seco; SEEK: DCA + mean
RDF2Vec embedding with the authors' fixed class embeddings)."""
import sys, time, json, itertools, math, random
import numpy as np
sys.path.insert(0,"."); sys.path.insert(0,"ExplainablePPI")
from kgsim import KG
from seekrep import SEEK
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import spearmanr
import shap
from lime import lime_tabular

PIPE,MODEL,FOLD,N_PAIRS=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
M_PERM=100; EXACT_MAX=8; TOPK=5
rng=np.random.default_rng(FOLD); random.seed(FOLD)

K=np.load("feats_kg.npz"); y=K["y"]; keys=[tuple(k) for k in K["keys"]]; aspects=list(K["aspects"])
ci=aspects.index("GO_0032991"); keep=[j for j in range(52) if j!=ci]; A=[aspects[j] for j in keep]
SF=np.load("feats_seek.npz")
X={"kg":K["Xpy"][:,keep],"seek":SF["Xseek51"],"hyb":np.hstack([K["Xpy"][:,keep],SF["Xseek51"]])}[PIPE]
kg=KG("godata/go-basic.owl","godata/goa_human.gaf",aspects); sub_kg=[kg.sub[j] for j in keep]
S_="seek/PPI/"
sk=SEEK(S_+"go.owl",S_+"go_annotations.gaf",S_+"Embeddings/Emb_classes_maxdepth4_nwalks100.txt")
CX=sk.desc("GO_0032991")|{"GO_0032991"}
sub_sk=[sk.desc(a)|{a} for a in A]

P="ExplainablePPI/Experimental_Results/Data/StratifiedPartitions/Indexes__crossvalidationTest__Run"
test=[int(l) for l in open(P+("1(complete)" if FOLD==1 else str(FOLD))+".txt") if l.strip()]
m=np.zeros(len(y),bool); m[test]=True
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
clf=(RandomForestClassifier(100,n_jobs=1,random_state=0) if MODEL=="rf"
     else XGBClassifier(n_estimators=300,n_jobs=1,random_state=0,eval_metric="logloss") if MODEL=="xgb"
     else make_pipeline(StandardScaler(),MLPClassifier((100,),max_iter=300,early_stopping=True,random_state=0))).fit(X[~m],y[~m])
def proba(Z): return clf.predict_proba(np.atleast_2d(Z))[:,1].astype(np.float64)

# state = (kg annotations p1, kg annotations p2, seek annotations p1, seek annotations p2)
def feats(st):
    k1,k2,s1,s2=st; parts=[]
    if PIPE in("kg","hyb"):
        v=kg.features(k1,k2); parts.append([v[j] for j in keep])
    if PIPE in("seek","hyb"): parts.append(list(sk.rep(s1,s2)))
    return np.concatenate(parts)
def players_of(st):
    k1,k2,s1,s2=st; out=[]
    for i in range(len(A)):
        a=(PIPE!="seek") and bool(k1&sub_kg[i]) and bool(k2&sub_kg[i])
        b=(PIPE!="kg") and bool(s1&sub_sk[i]) and bool(s2&sub_sk[i])
        if a or b: out.append(i)
    return out
def remove(st,S):
    if not S: return st
    rk=set().union(*[sub_kg[i] for i in S]); rs=set().union(*[sub_sk[i] for i in S])
    k1,k2,s1,s2=st; return (k1-rk,k2-rk,s1-rs,s2-rs)

def shapley(value_batch, players, seed):
    k=len(players); phi=np.zeros(len(A))
    if k==0: return phi
    if k<=EXACT_MAX:
        subsets=[frozenset(c) for r in range(k+1) for c in itertools.combinations(players,r)]
        vals=dict(zip(subsets,value_batch(subsets)))
        for i in players:
            others=[p for p in players if p!=i]
            for r in range(k):
                w=math.factorial(r)*math.factorial(k-r-1)/math.factorial(k)
                for c in itertools.combinations(others,r):
                    S=frozenset(c); phi[i]+=w*(vals[S]-vals[S|{i}])
        return phi
    r=random.Random(seed); perms=[r.sample(players,k) for _ in range(M_PERM)]
    subsets=set()
    for p in perms:
        for t in range(k+1): subsets.add(frozenset(p[:t]))
    subsets=list(subsets); vals=dict(zip(subsets,value_batch(subsets)))
    for p in perms:
        for t,i in enumerate(p): phi[i]+=vals[frozenset(p[:t])]-vals[frozenset(p[:t+1])]
    return phi/M_PERM

def explain_D(st,seed=0):
    pl=players_of(st)
    return shapley(lambda subs: proba(np.array([feats(remove(st,S)) for S in subs])),pl,seed)
def explain_LOO(st,seed=0):   # leave-one-aspect-out occlusion in knowledge space (aspect-level analogue of SEEK 'necessary')
    pl=players_of(st); phi=np.zeros(len(A))
    if not pl: return phi
    v=proba(np.array([feats(st)]+[feats(remove(st,[i])) for i in pl]))
    for j,i in enumerate(pl): phi[i]=v[0]-v[j+1]
    return phi
def explain_FS0(st,seed=0):   # KGsim2vec only: same players and sampler, feature set to 0 instead of recomputed
    pl=players_of(st); x=feats(st)
    def vb(subs):
        Z=np.repeat(x[None],len(subs),0)
        for r,S in enumerate(subs): Z[r,list(S)]=0
        return proba(Z)
    return shapley(vb,pl,seed)
bg=X[~m][rng.choice((~m).sum(),100,replace=False)]
tree=shap.TreeExplainer(clf,data=bg,feature_perturbation="interventional") if MODEL!="mlp" else None
def explain_TS(st,seed=0):    # kg: per aspect feature; hyb: only the 51 named KGsim2vec columns map to aspects
    x=feats(st); sv=tree.shap_values(x[None],check_additivity=False)
    sv=sv[1] if isinstance(sv,list) else sv
    sv=np.array(sv); sv=sv[0,:,1] if sv.ndim==3 else sv[0]
    return np.asarray(sv[:len(A)],dtype=np.float64)
if PIPE=="kg":
    lime_exp=lime_tabular.LimeTabularExplainer(X[~m][:2000],feature_names=A,class_names=["0","1"],discretize_continuous=True,random_state=0)
def explain_LIME(st,seed=0):
    x=feats(st); lime_exp.random_state=np.random.RandomState(seed)
    e=lime_exp.explain_instance(x,lambda Z: clf.predict_proba(Z).astype(np.float64),num_features=10,num_samples=2000,labels=(1,))
    phi=np.zeros(len(A))
    for j,w in e.as_map()[1]: phi[j]=w
    return phi
METHODS={"D":explain_D,"LOO":explain_LOO}
if PIPE=="kg": METHODS.update({"FS0":explain_FS0,"TreeSHAP":explain_TS,"LIME":explain_LIME})
if PIPE=="hyb": METHODS.update({"TreeSHAP_KGpart":explain_TS})
if MODEL=="mlp": METHODS={k:v for k,v in METHODS.items() if not k.startswith("TreeSHAP")}
DETERMINISTIC={"LOO","TreeSHAP","TreeSHAP_KGpart"}

def ranking(phi,pred):
    s=phi if pred==1 else -phi
    return [int(j) for j in np.argsort(-s) if s[j]>0]
def deletion_ks(st,order,pred):
    Z=np.array([feats(remove(st,order[:k])) for k in range(1,TOPK+1)]); p=proba(Z)
    return list(p if pred==1 else 1-p)
def insertion_ks(st,order,pred,pl):  # start with every player removed, add back the top-k
    Z=np.array([feats(remove(st,[i for i in pl if i not in order[:k]])) for k in range(0,TOPK+1)]); p=proba(Z)
    return list(p if pred==1 else 1-p)
def eff_size(phi):
    a=np.sort(np.abs(phi))[::-1]; tot=a.sum()
    return int(np.searchsorted(np.cumsum(a)/tot,0.8)+1) if tot>0 else 0
def jacc(a,b,k=3):
    A_,B_=set(a[:k]),set(b[:k]); return len(A_&B_)/len(A_|B_) if (A_|B_) else 1.0

pos=[i for i in test if y[i]==1]; neg=[i for i in test if y[i]==0]
idx=list(rng.choice(pos,N_PAIRS//2,replace=False))+list(rng.choice(neg,N_PAIRS//2,replace=False))
rec=[]; t0=time.time()
for n,i in enumerate(idx):
    p1,p2=keys[i]
    st=(set(kg.ann[p1]),set(kg.ann[p2]),set(sk.ann[p1])-CX,set(sk.ann[p2])-CX)
    p=proba(feats(st))[0]; pred=int(p>0.5); conf=p if pred else 1-p
    pl=players_of(st)
    r=dict(i=int(i),pair=f"{p1}-{p2}",y=int(y[i]),pred=pred,conf=float(conf),k=len(pl))
    # continuity: drop ONE random direct annotation (same GO term in both annotation sets) lying in a player aspect
    cand=sorted({(w,t) for w in (0,1) for t in (st[w]|st[w+2]) if any(t in sub_kg[a] or t in sub_sk[a] for a in pl)})
    pert=None
    if cand:
        w,t=cand[rng.integers(len(cand))]; q=list(st); q[w]=q[w]-{t}; q[w+2]=q[w+2]-{t}; q=tuple(q)
        if int(proba(feats(q))[0]>0.5)==pred: pert=q
    ro=[int(v) for v in rng.permutation(pl)] if pl else []
    r["del_random"]=deletion_ks(st,ro,pred) if ro else None
    r["ins_random"]=insertion_ks(st,ro,pred,pl) if ro else None
    for name,f in METHODS.items():
        ts_=time.time(); phi=f(st,0); r[f"time_{name}"]=time.time()-ts_
        order=ranking(phi,pred)
        r[f"cov_{name}"]=int(len(order)>0); r[f"size_{name}"]=eff_size(phi)
        r[f"del_{name}"]=deletion_ks(st,order,pred) if order else None
        r[f"ins_{name}"]=insertion_ks(st,order,pred,pl) if order else None
        r[f"top3_{name}"]=[A[j] for j in order[:3]]
        r[f"phi_{name}"]={A[j]:round(float(phi[j]),5) for j in np.nonzero(phi)[0]}
        if name in DETERMINISTIC or (name=="D" and len(pl)<=EXACT_MAX):
            r[f"cons_jacc_{name}"]=1.0; r[f"cons_rho_{name}"]=1.0
        else:
            phi2=f(st,1); o2=ranking(phi2,pred); r[f"cons_jacc_{name}"]=jacc(order,o2)
            r[f"cons_rho_{name}"]=float(spearmanr(phi,phi2).correlation) if np.std(phi)>0 and np.std(phi2)>0 else None
        r[f"cont_{name}"]=jacc(order,ranking(f(pert,0),pred)) if pert is not None else None
    rec.append(r)
    if n%25==0: print(PIPE,MODEL,FOLD,n,round(time.time()-t0,1),"s k=",len(pl),flush=True)
json.dump(dict(pipe=PIPE,model=MODEL,fold=FOLD,aspects=A,labels={a:kg.label.get(a,a) for a in A},records=rec),open(f"rec_{PIPE}_{MODEL}_f{FOLD}_{N_PAIRS}.json","w"))
print("done",time.time()-t0)
