# KAS-PPI: giải thích dự đoán tương tác protein theo aspect của Gene Ontology

Mã nguồn, notebook và tệp kết quả của luận văn thạc sĩ *Phát triển các kỹ thuật trích xuất quan hệ y sinh giải thích được trên đồ thị tri thức và ứng dụng trong dự đoán tương tác protein* (Nguyễn An Bằng, Trường Đại học Công nghệ, ĐHQGHN, 2026; cán bộ hướng dẫn: PGS.TS Hà Quang Thụy).

KAS (Knowledge-space Aspect Shapley) là bộ giải thích hậu nghiệm, không phụ thuộc bộ phân loại. KAS gỡ các chú giải của cả hai protein nằm dưới một tập aspect của Gene Ontology, tính lại đặc trưng bằng chính pipeline biểu diễn gốc, rồi phân bổ thay đổi xác suất dự đoán cho các aspect bằng giá trị Shapley. KAS-group dùng cùng cơ chế với người chơi là nhóm chú giải (theo nguồn gán hoặc mã bằng chứng). KAS-PPI áp dụng KAS cho ba view biểu diễn cặp protein (KGsim2vec, SEEK, Hybrid) trên bộ dữ liệu GO-PPI của SEEK và KGsim2vec.

## Cấu trúc

```
src/                   tầng biểu diễn: đặc trưng KGsim2vec, biểu diễn SEEK
experiments/
  main/                dự đoán 10 fold và giải thích chính
  roar_plausibility/   ROAR 2,200 và 47,142 cặp, Plausibility theo pathway
  extended_E/          bộ giải thích gốc của SEEK, ROAR hai chiều, Shapley chính xác
  extended_F/          điền khuyết kiểu ROAD, sanity check, LORE, KAS độ sâu 2
  kas_group/           KAS-group, phép kiểm theo kênh STRING, kiểm tra rò rỉ
  supplementary/       chia theo protein (C1), hiệu chỉnh Holm (C3)
notebooks/             notebook Google Colab (Bảng E.2)
  colab_runs/          bốn lần chạy Colab giữ nguyên kèm output (Bảng E.2b)
reproduction/          tái hiện SEEK và KGsim2vec bằng mã của tác giả (Phụ lục D)
results/               tệp kết quả tổng hợp mà các bảng của luận văn đọc
  environment/         pip freeze Colab v2–v4, kết quả chạy lại trên Colab v4
figures/               script vẽ Hình 4.1–4.6 và số liệu đối chiếu
tools/                 script kiểm tra (đối chiếu 60 fold)
```

Kết quả thô theo từng cặp (khoảng 180 MB, nén 69 MB) nằm trong `ket_qua_tho.zip` ở mục [Releases](https://github.com/NguyenAnBang1110/KAS-PPI/releases) (tag `v1.1`).

## Cài đặt

Môi trường chuẩn để tái lập là Python 3.13 với các phiên bản ghim trong `requirements.txt` (lấy từ `results/environment/pip_freeze_colab_v4_final.txt`).

```bash
git clone https://github.com/NguyenAnBang1110/KAS-PPI.git && cd KAS-PPI
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # numpy 2.1.3, scipy 1.16.3, scikit-learn 1.6.1, xgboost 3.2.0, shap 0.53.0
```

Từ xgboost 3.3, mặc định `enable_categorical = True` bị phép kiểm tra của shap 0.52.0 coi là có categorical split, nên TreeSHAP interventional trên XGB300 dừng với `NotImplementedError` (`notebooks/colab_runs/KAS_PPI_colab_v2_result_error_xgboost.ipynb`). shap 0.53.0 đã sửa lỗi này.

## Chạy lại các kết quả kiểm chứng

Các lệnh dưới đây chạy từ thư mục gốc của kho, chỉ cần CPU.

**Đối chiếu 60 fold của tầng dự đoán.** So WAF và AUC theo fold giữa tệp kết quả gốc và lần chạy lại trên Colab v4:

```bash
python3 tools/kiem_60_fold.py
# XGB300 WAF: 60/60 fold khớp tới 4 chữ số (lệch lớn nhất 0.0000) ...
```

XGB300 khớp 60/60 fold cả WAF và AUC, bốn p Wilcoxon trùng nhau. RF100 lệch tối đa 0.0011 ở các ô SEEK và Hybrid vì đặc trưng SEEK được dựng lại trên Colab; các ô KGsim2vec khớp hoàn toàn.

**C3, hiệu chỉnh Holm** (đọc kết quả ROAR hai chiều đã lưu, vài giây):

```bash
python3 experiments/supplementary/c3_holm.py results/extended_E c3.json
cmp <(python3 -m json.tool c3.json) <(python3 -m json.tool results/supplementary_C3/c3_holm.json) && echo "C3 khớp"
```

**C1, chia theo protein** (cần `ket_qua_tho/kas_group/string_check_input.npz` trong `ket_qua_tho.zip`; khoảng 4 phút trên 4 nhân):

```bash
unzip ket_qua_tho.zip 'ket_qua_tho/kas_group/string_check_input.npz'
python3 experiments/supplementary/c1_cold_protein.py ket_qua_tho/kas_group/string_check_input.npz c1.json
```

Tệp gốc `results/supplementary_C1/c1_cold_protein.json` được chạy với scikit-learn 1.9.1. Với scikit-learn 1.6.1 của môi trường chuẩn, mọi trung vị và giá trị lớn nhất WAF/AUC trùng tệp gốc; 9/80 giá trị theo nhóm lệch tối đa 0.0005, chủ yếu ở view Hybrid.

**Hình 4.1–4.6.** Mỗi script đọc `results/`, đối chiếu số liệu với `figures/bang_chuong4.txt` và ghi `figures/hinh_4_X.png`:

```bash
for i in 1 2 3 4 5 6; do python3 figures/ve_hinh_4_$i.py; done
```

**Notebook đầy đủ.** Mỗi notebook tự ghi script cần dùng (`%%writefile`), tự tải dữ liệu từ hai kho gốc. Mở `notebooks/KAS_PPI_colab.ipynb` trên Google Colab, chạy chế độ `quick` (fold 1) để kiểm nhanh hoặc `full` cho đủ 69 lượt giải thích.

**Tái hiện bổ sung KGsim2vec** (`reproduction/kgsim2vec_extra/`): các script đọc thư mục làm việc qua biến môi trường `KGSIM_WORKDIR` (mặc định `work/`), chứa bản clone ExplainablePPI, `feats_kg.npz` và `k10/`.

## Dữ liệu

Kho này không phân phối lại dữ liệu. Notebook tải trực tiếp từ:

- [liseda-lab/seek](https://github.com/liseda-lab/seek): cặp protein, partition, GO, chú giải, embedding RDF2Vec. Bản dùng khi kiểm lại có commit cuối `b848146`.
- [liseda-lab/ExplainablePPI](https://github.com/liseda-lab/ExplainablePPI): GO, chú giải GOA, tệp đặc trưng của tác giả KGsim2vec. Bản dùng khi kiểm lại có commit cuối `2a68b3a`.
- MSigDB 7.5.1 (gói conda-forge `r-msigdbr` 7.5.1), qua `experiments/roar_plausibility/get_msigdb.py`.
- STRING v11.5 (`protein.links.detailed`, `protein.aliases`, người, 9606), qua `notebooks/Kiem_tra_STRING_Reactome.ipynb`.

Notebook clone hai kho với `--depth 1`, tức bản mới nhất. Nếu kho gốc thay đổi, checkout hai commit trên trước khi chạy.

## Đối chiếu phiên bản mã

Bảng E.1 của luận văn ghi số dòng và 8 ký tự đầu SHA-256 của từng script; `CHECKSUMS.sha256` ghi mã băm đầy đủ của mọi tệp trong kho.

```bash
sha256sum -c CHECKSUMS.sha256
```

Trong các tệp kết quả, đường dẫn tuyệt đối tới thư mục làm việc lúc chạy được thay bằng `work` hoặc đường dẫn tương đối; giá trị số không đổi. Các notebook trong `notebooks/colab_runs/` được giữ nguyên như lúc chạy để làm bằng chứng.

## Trích dẫn

```bibtex
@mastersthesis{nguyen2026kasppi,
  author  = {Nguyễn An Bằng},
  title   = {Phát triển các kỹ thuật trích xuất quan hệ y sinh giải thích được trên đồ thị tri thức và ứng dụng trong dự đoán tương tác protein},
  school  = {Trường Đại học Công nghệ, Đại học Quốc gia Hà Nội},
  address = {Hà Nội},
  year    = {2026},
  note    = {Cán bộ hướng dẫn: PGS.TS Hà Quang Thụy},
  url     = {https://github.com/NguyenAnBang1110/KAS-PPI}
}
```

Nếu dùng mã này, xin trích dẫn thêm hai công trình gốc: SEEK (Sousa et al., 2023) và KGsim2vec (Sousa et al., 2024).
