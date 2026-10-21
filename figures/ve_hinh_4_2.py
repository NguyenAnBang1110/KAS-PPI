"""Hình 4.2. WAF theo fold của ba view với 52 và 51 aspect.

Nguồn: results/main/results_prediction_v2.json
  khoá "<VIEW>_<52|51>|<RF100|XGB300>", trường "folds" (WAF 10 fold, đã làm tròn 4 chữ số khi lưu),
  "median_waf", "iqr" (tính trên WAF chưa làm tròn).
Đối chiếu: Bảng 4.9 (trung vị và IQR).
Chạy: python3 ve_hinh_4_2.py
"""
# ---------------------------------------------------------------------------
# Phần chung (giống nhau ở cả sáu script ve_hinh_4_*.py)
# ---------------------------------------------------------------------------
import os, re, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(OUT), "results")   # thư mục results/ của kho
CHUONG4 = os.path.join(OUT, "bang_chuong4.txt")        # các bảng Chương 4 dùng để đối chiếu

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.grid": True, "grid.color": "#E3E3E3",
    "grid.linewidth": 0.6, "axes.axisbelow": True,
})

# Màu cố định cho từng phương pháp (Okabe-Ito và Paul Tol, an toàn với mù màu),
# dùng chung cho mọi hình; kèm ký hiệu điểm riêng để không phụ thuộc chỉ vào màu.
MAU = {"KAS": "#D55E00", "LOO": "#0072B2", "FS0": "#009E73", "TreeSHAP": "#CC79A7",
       "LIME": "#E69F00", "SEEKN": "#56B4E9", "SEEKS": "#AA4499", "Random": "#3A3A3A",
       "DT6": "#332288", "GP6x": "#882255"}
DAU = {"KAS": "o", "LOO": "s", "FS0": "^", "TreeSHAP": "D", "LIME": "v",
       "SEEKN": "P", "SEEKS": "X", "Random": "h", "DT6": "o", "GP6x": "D"}

def doc_bang(so_bang):
    """Đọc Bảng <so_bang> trong bang_chuong4.txt: trả về các dòng (danh sách ô), dòng 0 là tiêu đề."""
    lines = open(CHUONG4, encoding="utf-8").read().split("\n")
    i = next(j for j, l in enumerate(lines) if l.startswith(f"Bảng {so_bang}."))
    rows = []
    for l in lines[i + 1:]:
        if "|" not in l or l.startswith("Ghi chú"):
            break
        rows.append([c.strip() for c in l.split("|")])
    return rows

def dong(rows, ten):
    """Dòng có ô đầu đúng bằng `ten`."""
    r = [x for x in rows if x[0] == ten]
    if len(r) != 1:
        raise KeyError(f"không tìm thấy duy nhất dòng '{ten}'")
    return r[0]

def chuoi_so(s):
    """Các số (dạng chuỗi, giữ nguyên số chữ số thập phân) trong một ô bảng."""
    return re.findall(r"[−-]?\d*\.\d+|[−-]?\d+", s)

def f(x):
    return float(x.replace("−", "-"))

KET_QUA = []  # (nhãn, giá trị tệp, chuỗi trong bảng, khớp?)

def kiem(nhan, gia_tri_tep, chuoi_bang):
    """So giá trị tệp với số in trong bảng; lệch quá nửa đơn vị chữ số cuối của bảng thì đánh dấu LỆCH."""
    s = chuoi_bang.replace("−", "-")
    d = len(s.split(".")[1]) if "." in s else 0
    ok = abs(gia_tri_tep - float(s)) <= 0.5 * 10 ** (-d) + 1e-9
    KET_QUA.append((nhan, gia_tri_tep, chuoi_bang, ok))
    print(f"  {nhan:<58s} tệp = {gia_tri_tep:<10.5g} bảng = {chuoi_bang:<8s} {'khớp' if ok else '*** LỆCH ***'}")
    return ok

def tong_ket(ten_hinh):
    sai = [k for k in KET_QUA if not k[3]]
    print(f"\n[{ten_hinh}] Đối chiếu: {len(KET_QUA)} giá trị, {len(KET_QUA) - len(sai)} khớp, {len(sai)} lệch.")
    for k in sai:
        print(f"  LỆCH: {k[0]}: tệp {k[1]!r}, bảng {k[2]}")
# ---------------------------------------------------------------------------
SRC = f"{RES}/main/results_prediction_v2.json"
P = json.load(open(SRC))
VIEWS = [("KG", "KGsim2vec"), ("SEEK", "SEEK"), ("HYB", "Hybrid")]
MODELS = ["RF100", "XGB300"]
MAU_ASP = {"52": "#D9D9D9", "51": "#7A7A7A"}

print("== Đối chiếu Hình 4.2 với Bảng 4.9 ==")
B = doc_bang("4.9")
# tiêu đề: View | RF100, 52 aspect | RF100, 51 aspect | XGB300, 52 aspect | XGB300, 51 aspect
hdr = B[0]
for vk, vten in VIEWS:
    row = dong(B, vten)
    for m in MODELS:
        for a in ("52", "51"):
            col = hdr.index(f"{m}, {a} aspect")
            med_s, iqr_s = chuoi_so(row[col])
            rec = P[f"{vk}_{a}|{m}"]
            kiem(f"{vten} {m} {a} aspect: median_waf", rec["median_waf"], med_s)
            kiem(f"{vten} {m} {a} aspect: iqr", rec["iqr"], iqr_s)
            # trung vị của 10 giá trị fold đã làm tròn (giá trị thực sự được vẽ)
            kiem(f"{vten} {m} {a} aspect: trung vị 10 fold được vẽ", float(np.median(rec["folds"])), med_s)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
for ax, m in zip(axes, MODELS):
    pos, data, cols = [], [], []
    for i, (vk, _) in enumerate(VIEWS):
        for j, a in enumerate(("52", "51")):
            pos.append(i * 1.0 + (j - 0.5) * 0.36)
            data.append(P[f"{vk}_{a}|{m}"]["folds"])
            cols.append(MAU_ASP[a])
    bp = ax.boxplot(data, positions=pos, widths=0.28, patch_artist=True, whis=(0, 100),
                    medianprops=dict(color="black", linewidth=1.4), showfliers=False)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_edgecolor("#333333")
    for p, d in zip(pos, data):  # 10 điểm fold, không rung ngẫu nhiên
        ax.plot([p] * len(d), d, linestyle="none", marker="o", markersize=2.6,
                markerfacecolor="white", markeredgecolor="#222222", markeredgewidth=0.6, zorder=3)
    ax.set_xticks(range(len(VIEWS)))
    ax.set_xticklabels([f"view {v}" for _, v in VIEWS])
    ax.set_title(m)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("WAF trên từng fold (10 fold)")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fig.legend(handles=[Patch(facecolor=MAU_ASP["52"], edgecolor="#333333", label="52 aspect"),
                    Patch(facecolor=MAU_ASP["51"], edgecolor="#333333", label="51 aspect (bỏ complex)"),
                    Line2D([], [], linestyle="none", marker="o", markersize=3.5, markerfacecolor="white",
                           markeredgecolor="#222222", label="một fold")],
           loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(f"{OUT}/hinh_4_2.png", dpi=300)
print(f"Đã lưu {OUT}/hinh_4_2.png")
tong_ket("Hình 4.2")
