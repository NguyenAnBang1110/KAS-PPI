"""Hình 4.3. ROAR hai chiều trên 47,142 cặp, RF100: mức giảm WAF theo k ở chiều MoRF và LeRF.

Nguồn: results/extended_E/roar_sym_kg_rf.json (view KGsim2vec, E3)
       results/extended_E/roar_sym_seek_rf.json (view SEEK, E2)
  khoá "base" (WAF 10 fold của mô hình gốc), "res"["<PP>|<MoRF|LeRF>|<k>"] (WAF 10 fold sau khi xoá k aspect
  và huấn luyện lại). Mức giảm = trung vị(base) − trung vị(res), đúng công thức của
  experiments/extended_E/analyze_sym.py, script đã sinh e2_/e3_roar_sym_*.txt.
Đối chiếu: Bảng 4.12, 4.13 (cột MoRF và LeRF), WAF gốc với Bảng 4.9.
Chạy: python3 ve_hinh_4_3.py
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
SRC = {"KGsim2vec": f"{RES}/extended_E/roar_sym_kg_rf.json",
       "SEEK": f"{RES}/extended_E/roar_sym_seek_rf.json"}
BANG = {"KGsim2vec": "4.12", "SEEK": "4.13"}
KHOA = {"KAS": "D", "Random": "random"}  # tên trong luận văn -> tên khoá trong tệp
PP = {"KGsim2vec": ["KAS", "LOO", "FS0", "TreeSHAP", "LIME", "Random"],
      "SEEK": ["KAS", "LOO", "SEEKN", "SEEKS", "Random"]}
K = (1, 2, 3)

def giam(o, base, pp, chieu):
    key = KHOA.get(pp, pp)
    out = []
    for k in K:
        kk = f"{key}|{chieu}|{k}"
        out.append(base - float(np.median(o["res"][kk])) if kk in o["res"] else np.nan)
    return out

DATA = {}
for view, path in SRC.items():
    o = json.load(open(path))
    base = float(np.median(o["base"]))
    DATA[view] = {"base": base, "n": o["n"],
                  "MoRF": {p: giam(o, base, p, "MoRF") for p in PP[view]},
                  "LeRF": {p: giam(o, base, p, "LeRF") for p in PP[view]}}

print("== Đối chiếu Hình 4.3 với Bảng 4.12, 4.13 và WAF gốc (Bảng 4.9) ==")
B9 = doc_bang("4.9")
kiem("WAF gốc view KGsim2vec (median base) vs Bảng 4.9 RF100 51", DATA["KGsim2vec"]["base"],
     chuoi_so(dong(B9, "KGsim2vec")[B9[0].index("RF100, 51 aspect")])[0])
kiem("WAF gốc view SEEK (median base) vs Bảng 4.9 RF100 51", DATA["SEEK"]["base"],
     chuoi_so(dong(B9, "SEEK")[B9[0].index("RF100, 51 aspect")])[0])
for view in SRC:
    B = doc_bang(BANG[view])
    for p in PP[view]:
        row = dong(B, p)
        for ci, chieu in ((1, "MoRF"), (2, "LeRF")):
            vals = chuoi_so(row[ci]) if ci < len(row) else []
            for k, v in zip(K, DATA[view][chieu][p]):
                if np.isnan(v):
                    continue
                kiem(f"{view} {p} {chieu} k={k} (Bảng {BANG[view]})", v, vals[k - 1])
        if p == "Random":
            print(f"  ({view} Random: không có chiều LeRF trong tệp, bảng ghi '{row[2]}')")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
for ax, view in zip(axes, SRC):
    ax.axvspan(2.5, 3.5, color="#EFEFEF", zorder=0, linewidth=0)
    ax.text(3.0, 0.243, "k = 3: chỉ tham khảo,\nkhông xếp hạng", ha="center", va="top",
            fontsize=6.8, color="#444444")
    for p in PP[view]:
        for chieu, ls in (("MoRF", "-"), ("LeRF", "--")):
            y = DATA[view][chieu][p]
            if all(np.isnan(y)):
                continue
            ax.plot(K, y, linestyle=ls, color=MAU[p], marker=DAU[p], markersize=5.5,
                    linewidth=1.6, markeredgewidth=0.9,
                    markeredgecolor=MAU[p] if chieu == "LeRF" else "white",
                    markerfacecolor=MAU[p] if chieu == "MoRF" else "white")
    ax.set_xticks(K); ax.set_xlim(0.7, 3.5)
    ax.set_xlabel("k (số aspect bị xoá)")
    ax.set_title(f"view {view} (WAF gốc {DATA[view]['base']:.4f})")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Mức giảm WAF so với mô hình gốc")
axes[0].set_ylim(0, 0.245)
from matplotlib.lines import Line2D
tat_ca = ["KAS", "LOO", "FS0", "TreeSHAP", "LIME", "SEEKN", "SEEKS", "Random"]
h = [Line2D([], [], color=MAU[p], marker=DAU[p], linewidth=1.6, markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.5, label=p) for p in tat_ca]
h += [Line2D([], [], color="#333333", linestyle="-", linewidth=1.6, label="MoRF (cao hơn là tốt hơn)"),
      Line2D([], [], color="#333333", linestyle="--", linewidth=1.6, label="LeRF (thấp hơn là tốt hơn; điểm rỗng)")]
fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01),
           columnspacing=1.2, handlelength=2.6)
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(f"{OUT}/hinh_4_3.png", dpi=300)
print(f"Đã lưu {OUT}/hinh_4_3.png")
tong_ket("Hình 4.3")
