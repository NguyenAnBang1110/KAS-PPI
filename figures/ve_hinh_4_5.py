"""Hình 4.5. Đánh đổi fidelity–plausibility trên view SEEK.

Nguồn:
  Trục hoành: Thực nghiệm/KAS_PPI_v2/mo_rong_08_10/ket_qua/e1_summary.json,
              khoá "<rf|xgb>"["del_ks_<PP>"] (deletion AUC dưới toán tử ks, 2,200 cặp mỗi mô hình;
              PP = D (KAS), LOO, SEEKN, SEEKS, random). Đây là nguồn của Bảng 4.16.
  Trục tung : Thực nghiệm/KAS_PPI_v2/mo_rong_08_10/ket_qua/plaus_REACTOME_seekx.json,
              khoá results["seekx|<rf|xgb>"]["metrics"][<PP>]["hit1"]["mean"] và ["ci95"]
              (bootstrap 1,000 lần, xem mo_rong_08_10/plaus2.py); số cặp: ["n_informative"].
              Đây là nguồn của Bảng 4.23.
Đối chiếu: Bảng 4.16 (dòng toán tử ks), Bảng 4.23 (hit@1 và số cặp).
Chạy: python3 ve_hinh_4_5.py
"""
# ---------------------------------------------------------------------------
# Phần chung (giống nhau ở cả sáu script ve_hinh_4_*.py)
# ---------------------------------------------------------------------------
import os, re, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/project-files/1.Luận văn"
TN = f"{ROOT}/Thực nghiệm"
CHUONG4 = f"{ROOT}/Báo cáo/luan_van_text/Chuong4_Thuc_nghiem.txt"
OUT = os.path.dirname(os.path.abspath(__file__))

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
    """Đọc Bảng <so_bang> trong Chuong4_Thuc_nghiem.txt: trả về các dòng (danh sách ô), dòng 0 là tiêu đề."""
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
S_FID = f"{TN}/KAS_PPI_v2/mo_rong_08_10/ket_qua/e1_summary.json"
S_PL = f"{TN}/KAS_PPI_v2/mo_rong_08_10/ket_qua/plaus_REACTOME_seekx.json"
PP = ["KAS", "LOO", "SEEKN", "SEEKS", "Random"]
KHOA = {"KAS": "D", "Random": "random"}
MO = {"RF100": "rf", "XGB300": "xgb"}
DAU_MO = {"RF100": "o", "XGB300": "s"}
fid = json.load(open(S_FID)); pl = json.load(open(S_PL))["results"]
D = {}
for m, mk in MO.items():
    for p in PP:
        k = KHOA.get(p, p)
        h = pl[f"seekx|{mk}"]["metrics"][k]["hit1"]
        D[(p, m)] = dict(x=fid[mk][f"del_ks_{k}"], y=h["mean"], lo=h["ci95"][0], hi=h["ci95"][1])
NPAIR = {m: pl[f"seekx|{mk}"]["n_informative"] for m, mk in MO.items()}

print("== Đối chiếu Hình 4.5 với Bảng 4.16 và 4.23 ==")
B16 = doc_bang("4.16"); h16 = B16[0]
for m in MO:
    row = [r for r in B16 if r[0] == m and r[1] == "ks"][0]
    for p in PP:
        kiem(f"{m} ks deletion AUC {p} (Bảng 4.16)", D[(p, m)]["x"], chuoi_so(row[h16.index(p)])[0])
B23 = doc_bang("4.23"); h23 = B23[0]
for m in MO:
    row = [r for r in B23 if r[0].startswith(m + " (")][0]
    kiem(f"{m} số cặp Reactome (Bảng 4.23)", NPAIR[m], chuoi_so(row[0])[-1])
    for p in PP:
        kiem(f"{m} hit@1 {p} (Bảng 4.23)", D[(p, m)]["y"], chuoi_so(row[h23.index(p)])[0])

fig, ax = plt.subplots(figsize=(6.3, 4.8))
for m in MO:
    yr = D[("Random", m)]["y"]
    ax.axhline(yr, color=MAU["Random"], linestyle=":" if m == "RF100" else "--", linewidth=0.9, zorder=1)
NHAN_LECH = {  # vị trí nhãn chữ (điểm offset), chỉ để tránh chồng chữ; không đổi dữ liệu
    ("KAS", "RF100"): (-75, -32), ("KAS", "XGB300"): (-40, 34), ("LOO", "RF100"): (6, 4),
    ("LOO", "XGB300"): (6, 4), ("SEEKN", "RF100"): (6, -11), ("SEEKN", "XGB300"): (6, 4),
    ("SEEKS", "RF100"): (6, 4), ("SEEKS", "XGB300"): (6, 4), ("Random", "RF100"): (6, -11),
    ("Random", "XGB300"): (6, 5)}
for (p, m), v in D.items():
    ax.errorbar(v["x"], v["y"], yerr=[[v["y"] - v["lo"]], [v["hi"] - v["y"]]], fmt="none",
                ecolor=MAU[p], elinewidth=0.8, capsize=2, alpha=0.6, zorder=2)
    ax.plot(v["x"], v["y"], marker=DAU_MO[m], markersize=7.5, color=MAU[p], markeredgecolor="white",
            markeredgewidth=0.8, linestyle="none", zorder=3)
    ax.annotate(f"{p}, {m}", (v["x"], v["y"]), xytext=NHAN_LECH[(p, m)], textcoords="offset points",
                fontsize=7, color="#222222",
                arrowprops=dict(arrowstyle="-", color="#777777", linewidth=0.6, shrinkA=0, shrinkB=4)
                if p == "KAS" else None)
ax.invert_xaxis()
ax.set_xlabel("Deletion AUC dưới toán tử ks (trục đảo chiều: bên phải là trung thực hơn)")
ax.set_ylabel("hit@1 theo Reactome (cao hơn là hợp lý hơn)")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", color=MAU[p], linestyle="none", markersize=6.5, label=p)
     for p in PP]
h += [Line2D([], [], marker=DAU_MO[m], color="#777777", markerfacecolor="white", linestyle="none",
             markersize=6.5, label=f"{m} ({NPAIR[m]} cặp)") for m in MO]
h += [Line2D([], [], color=MAU["Random"], linestyle=":" if m == "RF100" else "--", linewidth=0.9,
             label=f"mức Random, {m} ({D[('Random', m)]['y']:.3f})") for m in MO]
fig.legend(handles=h, loc="lower center", fontsize=7, frameon=False, ncol=3,
           bbox_to_anchor=(0.5, 0.0), columnspacing=1.2)
ax.text(0.99, 0.01, "Vạch dọc: khoảng tin cậy 95% của hit@1 (bootstrap)", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=6.5, color="#555555")
fig.tight_layout(rect=(0, 0.11, 1, 1))
fig.savefig(f"{OUT}/hinh_4_5.png", dpi=300)
print(f"Đã lưu {OUT}/hinh_4_5.png")
tong_ket("Hình 4.5")
