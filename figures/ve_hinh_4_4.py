"""Hình 4.4. Thứ hạng Fidelity trên view KGsim2vec thay đổi theo toán tử xoá (RF100).

Nguồn:
  ks   : results/main/summary_v2.json, khoá pooled["kg|rf"]["del_<PP>"]["mean"]
         (deletion AUC, 2,200 cặp; PP = D (KAS), LOO, FS0, TreeSHAP, LIME, random)
  imp  : results/extended_F/road_rf_all_summary.json,
         res["imp"][<PP>][0] (MoRF AUC, 47,142 cặp); AUC bộ phân biệt: disc["imp"]
  knn  : results/extended_F/road_rf_rec_summary.json,
         res["knn"][<PP>][0] (MoRF AUC, 2,200 cặp; knn chỉ chạy trên 2,200 cặp); disc["knn"]
  zero : road_rf_all_summary.json, res["zero"][<PP>][0] (MoRF AUC, 47,142 cặp); disc["zero"]
  (phần tử [0] là MoRF AUC theo experiments/extended_F/analyze_road.py: [MoRF, LeRF, gap])
Đối chiếu: Bảng 4.11 (dòng KGsim2vec RF100, phần deletion) và Bảng 4.14 (các cột RF100).
Chạy: python3 ve_hinh_4_4.py
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
S_KS = f"{RES}/main/summary_v2.json"
S_ALL = f"{RES}/extended_F/road_rf_all_summary.json"
S_REC = f"{RES}/extended_F/road_rf_rec_summary.json"
PP = ["KAS", "LOO", "FS0", "TreeSHAP", "LIME", "Random"]
KHOA = {"KAS": "D", "Random": "random"}
ks = json.load(open(S_KS))["pooled"]["kg|rf"]
ra = json.load(open(S_ALL)); rr = json.load(open(S_REC))
NHOM = [  # (tên toán tử, nhãn, n cặp, giá trị theo phương pháp, AUC bộ phân biệt)
    ("ks", "ks\n(deletion AUC)", ks["n"], {p: ks[f"del_{KHOA.get(p, p)}"]["mean"] for p in PP}, None),
    ("imp", "imp\n(MoRF AUC)", ra["n"], {p: ra["res"]["imp"][KHOA.get(p, p)][0] for p in PP}, ra["disc"]["imp"]),
    ("knn", "knn\n(MoRF AUC)", rr["n"], {p: rr["res"]["knn"][KHOA.get(p, p)][0] for p in PP}, rr["disc"]["knn"]),
    ("zero", "zero\n(MoRF AUC)", ra["n"], {p: ra["res"]["zero"][KHOA.get(p, p)][0] for p in PP}, ra["disc"]["zero"]),
]

print("== Đối chiếu Hình 4.4 với Bảng 4.11 và 4.14 ==")
B11 = doc_bang("4.11"); r11 = dong(B11, "KGsim2vec RF100 (10; 2,200)")
for p in PP:
    kiem(f"ks deletion AUC {p} (Bảng 4.11, KGsim2vec RF100)", NHOM[0][3][p], chuoi_so(r11[B11[0].index(p)])[0])
kiem("ks: số cặp (Bảng 4.11: 2,200)", NHOM[0][2], "2200")
B14 = doc_bang("4.14"); h14 = B14[0]
COT = {"imp": "RF100, 47,142 cặp, imp", "knn": "RF100, 2,200 cặp, knn", "zero": "RF100, 47,142 cặp, zero"}
for ten, _, n, vals, disc in NHOM[1:]:
    ci = h14.index(COT[ten])
    for p in PP:
        row = dong(B14, "Random (chỉ MoRF)" if p == "Random" else p)
        kiem(f"{ten} MoRF AUC {p} (Bảng 4.14, cột '{COT[ten]}')", vals[p], chuoi_so(row[ci])[0])
    kiem(f"{ten} AUC bộ phân biệt (Bảng 4.14)", disc, chuoi_so(dong(B14, "AUC của bộ phân biệt")[ci])[0])
print("\nThứ hạng (1 = AUC thấp nhất) trong từng nhóm, theo giá trị tệp chưa làm tròn:")
for ten, _, n, vals, _ in NHOM:
    order = sorted(PP, key=lambda p: vals[p])
    print(f"  {ten:5s} (n = {n}): " + " < ".join(f"{p} {vals[p]:.4f}" for p in order))

fig, ax = plt.subplots(figsize=(7.0, 3.9))
w = 0.13
HATCH = {"KAS": "", "LOO": "//", "FS0": "", "TreeSHAP": "\\\\", "LIME": "", "Random": ".."}
for gi, (ten, nhan, n, vals, disc) in enumerate(NHOM):
    for j, p in enumerate(PP):
        x = gi + (j - (len(PP) - 1) / 2) * w
        ax.bar(x, vals[p], width=w * 0.92, color=MAU[p], edgecolor="white", linewidth=0.6,
               hatch=HATCH[p], label=p if gi == 0 else None)
        ax.text(x, vals[p] + 0.008, f"{vals[p]:.3f}", rotation=90, ha="center", va="bottom", fontsize=6.3,
                color="#222222")
ax.set_xticks(range(len(NHOM)))
ax.set_xticklabels([f"{nhan}\nn = {n:,} cặp" for _, nhan, n, _, _ in NHOM], fontsize=8)
for gi, (_, _, _, _, disc) in enumerate(NHOM):
    t = "AUC bộ phân biệt: không đo" if disc is None else f"AUC bộ phân biệt: {disc:.3f}"
    ax.annotate(t, xy=(gi, 0), xycoords=("data", "axes fraction"), xytext=(0, -46),
                textcoords="offset points", ha="center", va="top", fontsize=7.3, color="#333333")
ax.set_ylim(0, 1.0)
ax.set_ylabel("AUC dưới đường cong xoá\n(thấp hơn là tốt hơn)")
ax.grid(axis="x", visible=False)
ax.legend(ncol=6, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.10), handlelength=1.6,
          columnspacing=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/hinh_4_4.png", dpi=300)
print(f"Đã lưu {OUT}/hinh_4_4.png")
tong_ket("Hình 4.4")
