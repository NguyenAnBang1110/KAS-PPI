"""Hình 4.1. Kích thước và độ thông tin (IC trung bình) của giải thích KGsim2vec tái hiện trên partition 1.

Nguồn số tái hiện: results/reproduction/kgsim2vec_extra/
    k4_expl_base51.json, k4_expl_beta1.json, k4_expl_gamma001.json
  khoá "<DT6|RF+LIME3|XGB+LIME3|RF+LIME8|XGB+LIME8>"["size"|"avgIC"]["median"|"q1"|"q3"|"n"],
  "GP6x" (danh sách theo seed: "seed", "size", "avgIC"), "n_aspects".
  Các tệp này CHỈ lưu trung vị, Q1, Q3 (k_expl.py không lưu giá trị từng cặp), nên hình vẽ hộp Q1–Q3 và
  vạch trung vị, KHÔNG có râu (whisker) hay điểm ngoại lai. GP6x chỉ có một giải thích mỗi seed nên vẽ từng seed.
Nguồn số của bài báo (đọc bằng mắt từ Hình 3–5 của bài KGsim2vec, sai số khoảng ±0.02 IC), đã được ghi
  thành số trong báo cáo tái hiện: figures/so_lieu_bai_kgsim2vec_K4.md,
  bảng ở mục "K4. Hình 3–5". Script đọc bảng đó; ô không có số của bài (ví dụ "→ 3 / .206", "giống Hình 3")
  thì không vẽ điểm của bài.
Đối chiếu: Bảng 4.7 của Chuong4_Thuc_nghiem.txt (cả cột tái hiện lẫn cột bài báo).
Chạy: python3 ve_hinh_4_1.py
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
KQ = f"{RES}/reproduction/kgsim2vec_extra"
BAOCAO = os.path.join(OUT, "so_lieu_bai_kgsim2vec_K4.md")
CFG = [("base51", "Như Hình 3 của bài:\nα = 10, β = 0, γ = 0"), ("beta1", "Như Hình 4 của bài:\nβ = 1"),
       ("gamma001", "Như Hình 5 của bài:\nγ = 1%")]
J = {c: json.load(open(f"{KQ}/k4_expl_{c}.json")) for c, _ in CFG}
GT = ["DT6", "GP6x", "RF+LIME3", "XGB+LIME3", "RF+LIME8", "XGB+LIME8"]
NHAN = {"DT6": "DT6", "GP6x": "GP6x", "RF+LIME3": "RF + LIME 3", "XGB+LIME3": "XGB + LIME 3",
        "RF+LIME8": "RF + LIME 8", "XGB+LIME8": "XGB + LIME 8"}
MAU_GT = {"DT6": MAU["DT6"], "GP6x": MAU["GP6x"], "RF+LIME3": MAU["LIME"], "XGB+LIME3": MAU["LIME"],
          "RF+LIME8": MAU["LIME"], "XGB+LIME8": MAU["LIME"]}
DO = {"size": "Kích thước (số aspect)", "avgIC": "IC trung bình"}

# ---------- 1. Đọc số của bài từ bảng K4 trong báo cáo tái hiện ----------
md = open(BAOCAO, encoding="utf-8").read()
k4 = md[md.index("## K4."):md.index("## K5.")]
SAI_SO = float(re.search(r"sai số khoảng ±(\d*\.\d+) IC", k4).group(1))
DONG_MD = {"DT6": "DT6", "GP6x": "GP6x", "RF + LIME3": "RF+LIME3", "RF + LIME8": "RF+LIME8",
           "XGB + LIME3": "XGB+LIME3", "XGB + LIME8": "XGB+LIME8"}
md_rows = {}
for l in k4.split("\n"):
    if l.startswith("|"):
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if c[0] in DONG_MD:
            md_rows[DONG_MD[c[0]]] = c

def so_md(s):  # ".21" -> 0.21; giữ thứ tự
    return [float(x) for x in re.findall(r"\d*\.\d+|\d+", s)]

def bai_bao(cell):
    """Tách phần số của bài trong một ô: trả về dict có thể gồm size, size_lo, size_hi, ic, ic_lo, ic_hi."""
    p = cell.split("→")[0].strip()
    out = {}
    if not re.search(r"\d", p) or "giống Hình" in p:
        return out, p  # không có số của bài (ô trống, hoặc chỉ ghi "giống Hình 3")
    parts = [x.strip() for x in p.split(" / ")]
    if len(parts) == 2:
        sz, ic = parts
    else:  # một phần: số < 1 là IC
        v = so_md(parts[0])
        sz, ic = (None, parts[0]) if all(x < 1 for x in v) else (parts[0], None)
    if sz:
        v = so_md(sz); out["size"] = v[0]
        if len(v) == 3: out["size_lo"], out["size_hi"] = v[1], v[2]
    if ic:
        v = so_md(ic)
        if len(v) == 1: out["ic"] = v[0]
        else: out["ic_lo"], out["ic_hi"] = v[0], v[1]
    return out, p

BAI = {}  # (cfg, gt) -> dict
print("== Số của bài báo đọc từ bảng K4 của báo cáo tái hiện ==")
for gt in GT:
    c = md_rows[gt]
    for ci, (cfg, _) in zip((1, 3, 4), CFG):
        d, raw = bai_bao(c[ci])
        BAI[(cfg, gt)] = d
        print(f"  {cfg:9s} {gt:10s} ô '{c[ci]}' -> phần bài '{raw}' -> {d if d else 'KHÔNG CÓ SỐ, không vẽ'}")
print(f"  sai số đọc hình theo báo cáo: ±{SAI_SO}")

# ---------- 2. Đối chiếu với Bảng 4.7 ----------
print("\n== Đối chiếu Hình 4.1 với Bảng 4.7 ==")
B7 = doc_bang("4.7")
DONG_B7 = {"DT6": "DT6", "GP6x": "GP6x", "RF+LIME3": "RF + LIME 3 đặc trưng",
           "RF+LIME8": "RF + LIME 8 đặc trưng", "XGB+LIME3": "XGB + LIME 3 đặc trưng",
           "XGB+LIME8": "XGB + LIME 8 đặc trưng"}

def tai_hien_bang(cell):
    p = cell.split("→")[-1] if "→" in cell else cell
    p = re.sub(r"\(.*?\)", "", p).strip()
    parts = [x.strip() for x in p.split(" / ")]
    if len(parts) == 2: return parts[0], parts[1]
    return None, parts[0]

for gt in GT:
    row = dong(B7, DONG_B7[gt])
    for ci, (cfg, _) in zip((2, 3, 4), CFG):
        sz_s, ic_s = tai_hien_bang(row[ci])
        j = J[cfg][gt]
        if gt == "GP6x":
            sizes = [g["size"] for g in j]; ics = [g["avgIC"] for g in j]
            if sz_s:
                ss = chuoi_so(sz_s)
                kiem(f"{cfg} GP6x kích thước nhỏ nhất qua seed", min(sizes), ss[0])
                if len(ss) > 1: kiem(f"{cfg} GP6x kích thước lớn nhất qua seed", max(sizes), ss[1])
            ss = chuoi_so(ic_s)
            kiem(f"{cfg} GP6x IC nhỏ nhất qua seed", min(ics), ss[0])
            if len(ss) > 1: kiem(f"{cfg} GP6x IC lớn nhất qua seed", max(ics), ss[1])
        else:
            if sz_s:
                ss = chuoi_so(sz_s)
                kiem(f"{cfg} {gt} kích thước trung vị", j["size"]["median"], ss[0])
                if len(ss) == 3:
                    kiem(f"{cfg} {gt} kích thước Q1", j["size"]["q1"], ss[1])
                    kiem(f"{cfg} {gt} kích thước Q3", j["size"]["q3"], ss[2])
            kiem(f"{cfg} {gt} IC trung vị", j["avgIC"]["median"], chuoi_so(ic_s)[0])
    # cột bài báo của Bảng 4.7 so với bảng K4 của báo cáo tái hiện
    for ci, (cfg, _) in zip((1, 3, 4), CFG):
        cell = row[ci].split("→")[0]
        if "giống Hình" in cell: cell = ""
        so_b7 = [f(x) for x in chuoi_so(cell)]
        d = BAI[(cfg, gt)]
        so_rp = [d[k] for k in ("size", "size_lo", "size_hi", "ic", "ic_lo", "ic_hi") if k in d]
        ok = sorted(so_b7) == sorted(so_rp)
        KET_QUA.append((f"{cfg} {gt} số của bài: Bảng 4.7 so với báo cáo", so_rp, str(so_b7), ok))
        print(f"  {cfg + ' ' + gt + ' số của bài (Bảng 4.7 / báo cáo)':<58s} báo cáo = {so_rp}  bảng = {so_b7}  "
              f"{'khớp' if ok else '*** LỆCH ***'}")
for cfg, _ in CFG:
    print(f"  {cfg}: n_aspects = {J[cfg]['n_aspects']}; n cặp DT6 = {J[cfg]['DT6']['size']['n']}, "
          f"LIME = {J[cfg]['RF+LIME8']['size']['n']}; số seed GP6x = {len(J[cfg]['GP6x'])}")

# ---------- 3. Vẽ ----------
fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.0), sharey="row")
X = np.arange(len(GT))
for col, (cfg, ten) in enumerate(CFG):
    for row, do in enumerate(("size", "avgIC")):
        ax = axes[row, col]
        for i, gt in enumerate(GT):
            j = J[cfg][gt]; c = MAU_GT[gt]; x = i - 0.14
            if gt == "GP6x":
                vals = [g[do] for g in j]
                dx = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else [0.0]
                ax.plot(x + np.asarray(dx), vals, linestyle="none", marker="D", markersize=4.2, color=c,
                        markeredgecolor="white", markeredgewidth=0.4, zorder=3)
            else:
                s = j[do]
                h = max(s["q3"] - s["q1"], 0)
                ax.add_patch(plt.Rectangle((x - 0.17, s["q1"]), 0.34, h, facecolor=c, alpha=0.35,
                                           edgecolor=c, linewidth=1.0, zorder=2))
                ax.plot([x - 0.17, x + 0.17], [s["median"]] * 2, color=c if gt != "DT6" else MAU["DT6"],
                        linewidth=2.0, zorder=3)
            d = BAI[(cfg, gt)]; xb = i + 0.2
            if do == "size":
                if "size" in d:
                    ax.plot(xb, d["size"], marker="D", markersize=4.6, markerfacecolor="white",
                            markeredgecolor="#111111", markeredgewidth=0.9, linestyle="none", zorder=4)
                if "size_lo" in d:
                    ax.plot([xb, xb], [d["size_lo"], d["size_hi"]], color="#111111", linewidth=0.9, zorder=3)
            else:
                if "ic" in d:
                    ax.errorbar(xb, d["ic"], yerr=SAI_SO, fmt="D", markersize=4.6, markerfacecolor="white",
                                markeredgecolor="#111111", markeredgewidth=0.9, ecolor="#111111",
                                elinewidth=0.7, capsize=1.8, zorder=4)
                if "ic_lo" in d:
                    ax.errorbar(xb, (d["ic_lo"] + d["ic_hi"]) / 2, yerr=(d["ic_hi"] - d["ic_lo"]) / 2, fmt="none",
                                ecolor="#111111", elinewidth=0.9, capsize=2.2, zorder=4)
        ax.set_xticks(X)
        ax.set_xticklabels([NHAN[g] for g in GT] if row == 1 else [], fontsize=7, rotation=45, ha="right",
                           rotation_mode="anchor")
        ax.set_xlim(-0.6, len(GT) - 0.4)
        ax.grid(axis="x", visible=False)
        if row == 0:
            ax.set_title(f"{ten}\n({J[cfg]['n_aspects']} aspect)", fontsize=8.5)
        if col == 0:
            ax.set_ylabel(DO[do])
axes[0, 0].set_ylim(0, 14)
axes[1, 0].set_ylim(0, 0.75)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
h = [Patch(facecolor="#BBBBBB", edgecolor="#777777", label="Tái hiện: hộp Q1–Q3, vạch trung vị"),
     Line2D([], [], marker="D", color="#777777", linestyle="none", markersize=4.2, label="Tái hiện GP6x: từng seed"),
     Line2D([], [], marker="D", markerfacecolor="white", markeredgecolor="#111111", color="#111111",
            linestyle="none", markersize=4.6, label=f"Bài báo (đọc từ hình, IC ±{SAI_SO})"),
     Line2D([], [], color="#111111", linewidth=0.9, label="Bài báo: khoảng ghi trong báo cáo")]
h += [Patch(facecolor=MAU[k], label=k) for k in ("DT6", "GP6x", "LIME")]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0),
           columnspacing=1.0)
fig.tight_layout(rect=(0, 0.075, 1, 1))
fig.savefig(f"{OUT}/hinh_4_1.png", dpi=300)
print(f"Đã lưu {OUT}/hinh_4_1.png")
tong_ket("Hình 4.1")
