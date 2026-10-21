"""C3 (duyệt 14/10): hiệu chỉnh Holm cho các p Wilcoxon một phía theo fold của ROAR hai chiều.
Họ kiểm định cố định trước khi tính: mọi p trong khoá "p" của roar_sym_<view>_rf.json với k <= 2
(quy tắc xếp hạng của luận văn), tức KAS so với từng phương pháp, ba đại lượng MoRF/LeRF/gap.
Phân tích độ nhạy: cùng họ nhưng thêm k = 3. Chạy cho view SEEK (Bảng 4.13) và, để tham chiếu, view KGsim2vec (Bảng 4.12).
Không đổi số liệu gốc; chỉ đọc JSON. Dùng: python3 c3_holm.py DIR_mo_rong_08_10_ket_qua OUT.json"""
import sys, json
import numpy as np

def holm(ps):
    keys = sorted(ps, key=ps.get); m = len(keys); adj = {}; running = 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * ps[k])); adj[k] = running
    return adj

D, OUT = sys.argv[1], sys.argv[2]; res = {}
for view in ["seek", "kg"]:
    o = json.load(open(f"{D}/roar_sym_{view}_rf.json")); p = o["p"]
    for label, kmax in [("k<=2", 2), ("k<=3", 3)]:
        fam = {k: v for k, v in p.items() if int(k.rsplit("_", 1)[1]) <= kmax}
        adj = holm(fam)
        res[f"{view}|{label}"] = {"m": len(fam), "rows": {k: {"p": fam[k], "p_holm": adj[k]} for k in sorted(fam)}}
        print(f"== {view} {label}: m = {len(fam)}")
        for k in sorted(fam): print(f"  {k:28s} p={fam[k]:.4g}  holm={adj[k]:.4g}  {'*' if adj[k] < 0.05 else ''}")
json.dump(res, open(OUT, "w"), indent=1)
