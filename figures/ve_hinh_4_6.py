"""Hình 4.6. Thứ hạng nhóm theo KAS-group so với thứ hạng theo ablation có huấn luyện lại.

Nguồn: results/kas_group/kasg_summary.json, bốn khối "kg_source", "kg_evidence",
  "seek_source", "seek_evidence"; trong mỗi khối: "groups" (tên nhóm), "scores"["KASg_help"]
  (điểm toàn cục KAS-group), "drop" (mức giảm WAF khi bỏ nhóm và huấn luyện lại, dạng tỉ lệ;
  ×100 = điểm phần trăm như Bảng 4.25–4.26), "rho_KASg_help" (Spearman với drop / với excess).
  Hệ số gộp 24 điểm được tính lại đúng như kas_group/analyze_group.py (xếp hạng chuẩn hoá trong từng
  khối rồi Spearman trên toàn bộ điểm) và so với dòng "pooled" của kas_group/kasg_summary.txt.
Thứ hạng trong hình: 1 = điểm KAS-group cao nhất / mức giảm WAF lớn nhất trong cấu hình.
Đối chiếu: Bảng 4.25, 4.26 (cột KAS-group và giảm WAF), Bảng 4.27 (cột KAS-group).
Chạy: python3 ve_hinh_4_6.py
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
from scipy.stats import rankdata, spearmanr
SRC = f"{RES}/kas_group/kasg_summary.json"
SRC_TXT = f"{RES}/kas_group/kasg_summary.txt"
S = json.load(open(SRC))
CAUHINH = [("kg_source", "KGsim2vec, nguồn gán", "4.25", "KGsim2vec"),
           ("kg_evidence", "KGsim2vec, mã bằng chứng", "4.26", "KGsim2vec"),
           ("seek_source", "SEEK, nguồn gán", "4.25", "SEEK"),
           ("seek_evidence", "SEEK, mã bằng chứng", "4.26", "SEEK")]
MAU_CH = {"kg_source": "#332288", "kg_evidence": "#88CCEE", "seek_source": "#117733", "seek_evidence": "#DDCC77"}
DAU_CH = {"kg_source": "o", "kg_evidence": "s", "seek_source": "^", "seek_evidence": "D"}

print("== Đối chiếu Hình 4.6 với Bảng 4.25, 4.26, 4.27 ==")
B27 = doc_bang("4.27")
pts = []
for key, ten, bang, view in CAUHINH:
    b = S[key]; G = b["groups"]; sc = np.array(b["scores"]["KASg_help"]); dr = np.array(b["drop"])
    B = doc_bang(bang); h = B[0]
    c_sc = h.index(f"{view}: KAS-group"); c_dr = h.index(f"{view}: giảm WAF / đối chứng / vượt đối chứng")
    for g, s_, d_ in zip(G, sc, dr):
        row = dong(B, g)
        kiem(f"{ten}: {g} KAS-group (Bảng {bang})", float(s_), chuoi_so(row[c_sc])[0])
        kiem(f"{ten}: {g} giảm WAF ×100 (Bảng {bang})", float(d_) * 100, chuoi_so(row[c_dr])[0])
    n = len(G)
    kiem(f"{ten}: số nhóm (Bảng 4.27)", n, chuoi_so(dong(B27, f"{ten} ({n})")[0])[-1])
    kiem(f"{ten}: Spearman KAS-group với giảm WAF (Bảng 4.27)", b["rho_KASg_help"][0],
         chuoi_so(dong(B27, f"{ten} ({n})")[1])[0])
    # kiểm lại hệ số từ chính hai cột được vẽ
    kiem(f"{ten}: Spearman tính lại từ scores/drop (Bảng 4.27)", float(spearmanr(sc, dr).correlation),
         chuoi_so(dong(B27, f"{ten} ({n})")[1])[0])
    if len(set(sc)) < n or len(set(dr)) < n:
        print(f"  CHÚ Ý: {ten} có giá trị trùng, thứ hạng dùng hạng trung bình")
    rx = rankdata(-sc, method="average"); ry = rankdata(-dr, method="average")
    for g, a, c in zip(G, rx, ry):
        pts.append((key, g, float(a), float(c)))
# hệ số gộp 24 điểm, cùng cách tính với analyze_group.py
a = np.concatenate([np.argsort(np.argsort(np.array(S[k]["scores"]["KASg_help"]))) / (len(S[k]["groups"]) - 1)
                    for k, *_ in CAUHINH])
bb = np.concatenate([np.argsort(np.argsort(np.array(S[k]["drop"]))) / (len(S[k]["groups"]) - 1)
                     for k, *_ in CAUHINH])
rho_gop = float(spearmanr(a, bb).correlation)
r_gop = dong(B27, "Gộp (24 điểm; TreeSHAP-g 12 điểm)")
kiem("Gộp: Spearman KAS-group với giảm WAF (Bảng 4.27)", rho_gop, chuoi_so(r_gop[1])[0])
txt = open(SRC_TXT, encoding="utf-8").read()
m = re.search(r"KASg_help\s+n=(\d+)\s+rho_drop\s+([+-]?\d*\.\d+)", txt)
kiem("Gộp: Spearman so với dòng pooled của kasg_summary.txt", rho_gop, m.group(2).lstrip("+"))
kiem("Gộp: số điểm", len(pts), m.group(1))

fig, ax = plt.subplots(figsize=(5.6, 5.6))
off = {k: (i - 1.5) * 0.11 for i, (k, *_) in enumerate(CAUHINH)}
ax.plot([0.5, 7.5], [0.5, 7.5], color="#999999", linewidth=0.8, linestyle="--", zorder=1)
for key, ten, *_ in CAUHINH:
    P = [(g, x, y) for k, g, x, y in pts if k == key]
    ax.scatter([x + off[key] for _, x, _ in P], [y for *_, y in P], s=42, marker=DAU_CH[key],
               color=MAU_CH[key], edgecolor="#222222", linewidth=0.6, zorder=3,
               label=f"{ten} ({len(P)} nhóm)")
# ghi tên Reactome và STATEMENT: một nhãn cho mỗi tên, nối tới mọi điểm của tên đó (vị trí chữ chỉ là bố cục)
VI_TRI_NHAN = {"Reactome": (1.75, 1.55), "STATEMENT": (1.95, 0.75)}
for g, (tx, ty) in VI_TRI_NHAN.items():
    for k, gg, x, y in pts:
        if gg == g:
            ax.annotate("", (x + off[k], y), xytext=(tx, ty), textcoords="data",
                        arrowprops=dict(arrowstyle="-", color="#777777", linewidth=0.5, shrinkA=0, shrinkB=4))
    ax.text(tx, ty, f"{g}", fontsize=7.8, ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))
ax.set_xlim(0.4, 7.6); ax.set_ylim(0.4, 7.6)
ax.set_xticks(range(1, 8)); ax.set_yticks(range(1, 8))
ax.invert_yaxis()
ax.set_xlabel("Thứ hạng theo điểm KAS-group trong cấu hình (1 = cao nhất)")
ax.set_ylabel("Thứ hạng theo mức giảm WAF khi huấn luyện lại (1 = lớn nhất)")
fig.legend(loc="lower center", fontsize=7.5, frameon=False, ncol=2, bbox_to_anchor=(0.5, 0.0))
ax.text(0.98, 0.97, f"Spearman gộp 24 điểm: ρ = {rho_gop:.2f}\nĐường đứt: hai thứ hạng trùng nhau",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.3, color="#333333")
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(f"{OUT}/hinh_4_6.png", dpi=300)
print(f"Đã lưu {OUT}/hinh_4_6.png")
print("Các điểm (cấu hình, nhóm, hạng KAS-group, hạng giảm WAF):")
for p in pts:
    print("  ", p)
tong_ket("Hình 4.6")
