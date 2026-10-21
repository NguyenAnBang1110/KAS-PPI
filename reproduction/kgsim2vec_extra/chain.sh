V="$(cd "${KGSIM_WORKDIR:-work}" && pwd)"   # thư mục làm việc tuyệt đối (xem k_gp.py)
cd "$V/k10"
while [ ! -f FEATS_DONE ]; do sleep 30; done
python3 k_models.py feats_beta1.npz models_beta1.json 0,1,2 > log_models_beta1.txt 2>&1 &
python3 k_models.py feats_gamma001.npz models_gamma001.json 0,1,2 > log_models_gamma001.txt 2>&1 &
python3 - <<PY
V="$V"
c=[f"cd {V}/ExplainablePPI && python3 {V}/k10/k_gp.py {a} {f} 0 {V}/k10/feats_{v}.npz {V}/k10/gp_{v}" for v in ("beta1","gamma001") for a in ("GP","GP6x") for f in range(1,11)]
open(f"{V}/k10/k3_cmds.txt","w").write("\n".join(c)+"\n")
PY
while [ ! -f K1_DONE ]; do sleep 30; done
xargs -P 3 -I{} sh -c '{}' < k3_cmds.txt > log_k3.txt 2>&1
touch K3_DONE
python3 k_expl.py base51 500 > log_k4_base51.txt 2>&1 &
python3 k_expl.py beta1 500 > log_k4_beta1.txt 2>&1 &
python3 k_expl.py gamma001 500 > log_k4_gamma001.txt 2>&1
wait
touch ALL_DONE
