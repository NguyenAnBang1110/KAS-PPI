"""C1: WAF khi chia theo protein (cold-protein).
Thiết kế cố định trước khi chạy:
- 6,750 protein chia ngẫu nhiên thành 5 nhóm (numpy default_rng(0), permutation).
- Với nhóm g: tập huấn luyện = cặp không chứa protein nào của g.
  * Biến thể "cả hai mới" (C3 theo Park và Marcotte 2012): tập kiểm tra = cặp có cả hai protein thuộc g.
  * Biến thể "một mới" (C2): tập kiểm tra = cặp có đúng một protein thuộc g.
- Bộ phân loại RF100, random_state = 0, giống train_v2.py; WAF ở ngưỡng 0.5 và ROC-AUC.
- Ba view 51 aspect từ string_check_input.npz; thêm baseline chỉ dùng bậc protein (bậc và bậc dương của hai protein,
  đếm trên tập huấn luyện của nhóm; hàng huấn luyện trừ đóng góp của chính cặp).
- Không đổi siêu tham số, không chọn lại gì theo kết quả.
Dùng: python3 c1_cold_protein.py string_check_input.npz OUT.json"""
import sys, json, time, platform
from collections import Counter
import numpy as np, sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

z = np.load(sys.argv[1]); OUT = sys.argv[2]
p1, p2, y = z["p1"], z["p2"], z["y"]
views = {"KGsim2vec": z["kg_full"], "SEEK": z["seek_full"], "Hybrid": np.hstack([z["kg_full"], z["seek_full"]])}
prots = np.array(sorted(set(p1) | set(p2)))
perm = np.random.default_rng(0).permutation(len(prots))
group = {p: int(i % 5) for p, i in zip(prots[perm], range(len(prots)))}
g1 = np.array([group[a] for a in p1]); g2 = np.array([group[b] for b in p2])

def deg_feats(tr_idx, idx, correct):
    posdeg, deg = Counter(), Counter()
    for a, b, yy in zip(p1[tr_idx], p2[tr_idx], y[tr_idx]):
        deg[a] += 1; deg[b] += 1
        if yy == 1: posdeg[a] += 1; posdeg[b] += 1
    X = []
    for a, b, yy in zip(p1[idx], p2[idx], y[idx]):
        pa, pb, da, db = posdeg[a], posdeg[b], deg[a], deg[b]
        if correct: pa -= yy; pb -= yy; da -= 1; db -= 1
        X.append([min(pa, pb), max(pa, pb), min(da, db), max(da, db), pa + pb, (pa / da if da else 0), (pb / db if db else 0)])
    return np.array(X, float)

out = {"env": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
       "design": "5 protein groups, rng(0); train = pairs with no protein in g; test both-new or one-new; RF100 rs=0",
       "n_proteins": int(len(prots)), "results": {}}
for variant in ["both_new", "one_new"]:
    for g in range(5):
        tr = (g1 != g) & (g2 != g)
        te = ((g1 == g) & (g2 == g)) if variant == "both_new" else ((g1 == g) ^ (g2 == g))
        tri, tei = np.where(tr)[0], np.where(te)[0]
        row = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "test_pos_rate": float(y[te].mean())}
        for name, X in views.items():
            t = time.time()
            clf = RandomForestClassifier(100, n_jobs=-1, random_state=0).fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            row[name] = {"waf": round(f1_score(y[te], (p > 0.5).astype(int), average="weighted"), 4),
                         "auc": round(roc_auc_score(y[te], p), 4), "sec": round(time.time() - t, 1)}
        clf = RandomForestClassifier(100, n_jobs=-1, random_state=0).fit(deg_feats(tri, tri, True), y[tr])
        p = clf.predict_proba(deg_feats(tri, tei, False))[:, 1]
        row["degree_only"] = {"waf": round(f1_score(y[te], (p > 0.5).astype(int), average="weighted"), 4),
                              "auc": round(roc_auc_score(y[te], p), 4)}
        out["results"][f"{variant}|g{g}"] = row
        print(variant, g, json.dumps(row), flush=True)
summ = {}
for variant in ["both_new", "one_new"]:
    rows = [out["results"][f"{variant}|g{g}"] for g in range(5)]
    summ[variant] = {k: {"median_waf": float(np.median([r[k]["waf"] for r in rows])), "min_waf": min(r[k]["waf"] for r in rows),
                         "max_waf": max(r[k]["waf"] for r in rows), "median_auc": float(np.median([r[k]["auc"] for r in rows]))}
                     for k in ["KGsim2vec", "SEEK", "Hybrid", "degree_only"]}
    summ[variant]["n_test_total"] = int(sum(r["n_test"] for r in rows))
out["summary"] = summ
print(json.dumps(summ, indent=1))
json.dump(out, open(OUT, "w"), indent=1)
