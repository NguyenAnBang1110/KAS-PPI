"""How much do aspects annotated on only ONE protein (non-players) move the prediction when removed?"""
import sys, numpy as np, json
sys.argv=["x",sys.argv[1],"rf","1","1"]
src=open("explain_v2.py").read().split("pos=[i for i in test")[0]   # reuse setup (data, model, feats, remove)
exec(src)
PIPE_=PIPE
pos=[i for i in test if y[i]==1]; neg=[i for i in test if y[i]==0]
r=np.random.default_rng(7); idx=list(r.choice(pos,100,replace=False))+list(r.choice(neg,100,replace=False))
d_pl=[]; d_one=[]; n_one=[]; flips_one=0; flips_pl=0
for i in idx:
    p1,p2=keys[i]; st=(set(kg.ann[p1]),set(kg.ann[p2]),set(sk.ann[p1])-CX,set(sk.ann[p2])-CX)
    p0=proba(feats(st))[0]; pl=set(players_of(st))
    one=[]
    for a in range(len(A)):
        if a in pl: continue
        if PIPE_=="kg": h1,h2=bool(st[0]&sub_kg[a]),bool(st[1]&sub_kg[a])
        else: h1,h2=bool(st[2]&sub_sk[a]),bool(st[3]&sub_sk[a])
        if h1!=h2: one.append(a)
    n_one.append(len(one))
    for a in one:
        p=proba(feats(remove(st,[a])))[0]; d_one.append(abs(p-p0)); flips_one+=int((p>0.5)!=(p0>0.5))
    for a in pl:
        p=proba(feats(remove(st,[a])))[0]; d_pl.append(abs(p-p0)); flips_pl+=int((p>0.5)!=(p0>0.5))
d_one=np.array(d_one); d_pl=np.array(d_pl)
print(json.dumps(dict(pipe=PIPE_,pairs=200,one_sided_per_pair=float(np.mean(n_one)),players_per_pair=float(len(d_pl)/200),
  mean_abs_dp_one=float(d_one.mean()),frac_one_zero=float((d_one<1e-9).mean()),frac_one_gt005=float((d_one>0.05).mean()),flips_one=flips_one,
  mean_abs_dp_player=float(d_pl.mean()),frac_pl_gt005=float((d_pl>0.05).mean()),flips_pl=flips_pl)))
