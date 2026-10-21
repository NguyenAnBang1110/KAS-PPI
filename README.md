# KAS-PPI

Mã nguồn, notebook và tệp kết quả của luận văn thạc sĩ *Phát triển các kỹ thuật trích xuất quan hệ y sinh giải thích được trên đồ thị tri thức và ứng dụng trong dự đoán tương tác protein* (Nguyễn An Bằng; cán bộ hướng dẫn: PGS.TS Hà Quang Thụy).

KAS (Knowledge-Aspect Shapley) là bộ giải thích dựa trên giá trị Shapley, với người chơi là các aspect của Gene Ontology hoặc các nhóm chú giải. KAS-PPI áp dụng KAS cho ba view biểu diễn cặp protein (KGsim2vec, SEEK, Hybrid) trên bộ dữ liệu GO-PPI của SEEK và KGsim2vec.

## Cấu trúc

```
src/                    tầng biểu diễn: đặc trưng KGsim2vec (kgsim.py, kgsim_fix.py), biểu diễn SEEK (seekrep.py)
experiments/
  main/                 dự đoán 10 fold và giải thích chính (mục 4.3, 4.4, 4.6)
  roar_plausibility/    ROAR 2,200 và 47,142 cặp, Plausibility theo pathway (mục 4.4, 4.7)
  extended_E/           so sánh với bộ giải thích gốc của SEEK, ROAR hai chiều, Shapley chính xác, Bảng 9 KGsim2vec
  extended_F/           điền khuyết kiểu ROAD, sanity check, LORE, KAS độ sâu 2
  kas_group/            KAS-group, phép kiểm theo kênh STRING, kiểm tra rò rỉ (mục 4.8)
    per_pair/           kiểm chứng KAS-group theo từng cặp với Shapley có huấn luyện lại
  supplementary/        phép chia theo protein, hiệu chỉnh Holm
notebooks/              notebook chạy trên Google Colab (Bảng E.2)
  colab_runs/           bốn lần chạy Colab kèm output, dùng cho khảo sát độ bền theo phiên bản (Bảng E.2b)
reproduction/           tái hiện SEEK và KGsim2vec bằng mã của tác giả (Phụ lục D)
results/                tệp kết quả tổng hợp mà các bảng của luận văn đọc
figures/                script vẽ Hình 4.1–4.6
```

Kết quả thô theo từng cặp (khoảng 170 MB) nằm trong tệp `ket_qua_tho.zip` ở mục Releases.

## Chạy lại

Mỗi notebook tự ghi các script cần dùng (`%%writefile`), tự tải dữ liệu từ hai kho gốc và chỉ cần CPU.

1. Mở `notebooks/KAS_PPI_colab.ipynb` trên Google Colab.
2. Cài đúng môi trường chuẩn: `pip install -r requirements.txt` (Python 3.13, scikit-learn 1.6.1, scipy 1.16.3, xgboost 3.2.0, shap 0.53.0).
3. Chạy chế độ `quick` (fold 1) để kiểm nhanh, hoặc `full` cho đủ 69 lượt giải thích.

Với môi trường này, 60/60 giá trị WAF và AUC theo fold của XGB300 khớp tệp kết quả đã lưu tới bốn chữ số (`notebooks/colab_runs/KAS_PPI_colab_v4_result_final.ipynb`, `results/environment/`). Từ xgboost 3.3, mặc định `enable_categorical = True` bị phép kiểm tra của shap 0.52.0 coi là có categorical split, nên TreeSHAP interventional trên XGB300 dừng với `NotImplementedError` (`KAS_PPI_colab_v2_result_error_xgboost.ipynb`); shap 0.53.0 đã sửa lỗi này.

## Dữ liệu

Kho này không phân phối lại dữ liệu. Notebook tải trực tiếp từ:

- [liseda-lab/seek](https://github.com/liseda-lab/seek): cặp protein, partition, GO, chú giải, embedding RDF2Vec. Bản dùng khi kiểm lại có commit cuối `b848146`.
- [liseda-lab/ExplainablePPI](https://github.com/liseda-lab/ExplainablePPI): GO, chú giải GOA, tệp đặc trưng của tác giả KGsim2vec. Bản dùng khi kiểm lại có commit cuối `2a68b3a`.
- MSigDB 7.5.1 (gói conda-forge `r-msigdbr` 7.5.1), qua `get_msigdb.py`.
- STRING v11.5 (`protein.links.detailed`, `protein.aliases`, người, 9606), qua `Kiem_tra_STRING_Reactome.ipynb`.

Notebook clone hai kho với `--depth 1`, tức bản mới nhất. Nếu kho gốc thay đổi, checkout hai commit trên trước khi chạy.

## Đối chiếu phiên bản mã

Bảng E.1 của luận văn ghi số dòng và 8 ký tự đầu SHA-256 của từng script. `CHECKSUMS.sha256` ghi mã băm đầy đủ của mọi tệp trong kho.

```
sha256sum -c CHECKSUMS.sha256
```

## Trích dẫn

Nếu dùng mã này, xin trích dẫn luận văn và hai công trình gốc: SEEK (Sousa et al., 2023) và KGsim2vec (Sousa et al., 2024).
