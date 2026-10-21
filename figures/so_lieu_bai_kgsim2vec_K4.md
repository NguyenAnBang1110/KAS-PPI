## K4. Hình 3–5: kích thước và IC trung bình của giải thích (partition 1)

**Cách đo**
- Dùng `evaluation_explanations.process_models` và `process_rule` của repo. IC lấy từ `ICSeco_GOterms.txt`.
- DT6 và DT: tính trên toàn bộ 4,716 cặp test của fold 1.
- LIME: 500 cặp chọn ngẫu nhiên.
- GP6x: mô hình fold 1 với 3 seed.
- LORE: dùng lại luật của thực nghiệm LORE nhóm F (`experiments/extended_F/lore_kg.py`, `lore_sa` trên RF100/XGB300 của KAS), 57–59 cặp fold 1.
- Số của bài là **[đọc hình]**, sai số khoảng ±0.02 IC.

| Giải thích | Hình 3 bài: size / IC | Đo (51 aspect): size trung vị [Q1–Q3] / IC | Hình 4 (β = 1) bài → đo | Hình 5 (γ) bài → đo |
|---|---|---|---|---|
| DT6 | 2 [2–3] / ~.21 | 3 [2–3] / .214 | giống Hình 3 → 3 / .214 | giống Hình 3 → 3 / .214 |
| GP6x | 8 / ~.41 | 6–9 / .35–.42 (3 seed) | 10 / ~.41 → 12 / .47 | 9 / ~.41 → 8 / .40 |
| RF + LIME3 | 3 / ~.15–.20 | 3 / .206 [.15–.21] | → 3 / .206 | → 3 / .206 |
| RF + LIME8 | 8 / ~.545 | 8 / .537 | ~.50 → .499 | ~.375 → .383 |
| XGB + LIME3 | 3 / ~.20 | 3 / .206 | → .206 | → .206 |
| XGB + LIME8 | 8 / ~.55 | 8 / .557 | ~.49 → .507 | ~.375 → .387 |
| RF + LORE | 3 [2–4] / ~.19 | 6 [2–8] / .27 (lore_sa) | không so | không so |
| XGB + LORE | 3 [1–4] / ~.16 | 5 [2–7.5] / .32 (lore_sa) | không so | không so |

**Đọc kết quả**
- [đo] Kết luận định tính của bài **tái hiện được**:
  - DT6 cho giải thích nhỏ nhất và ít informative nhất;
  - GP6x nằm ở giữa;
  - LIME phụ thuộc số feature: LIME8 informative nhất;
  - khi lọc aspect (Hình 4, 5) thì IC của LIME8 giảm, mạnh nhất với γ (từ .54 xuống .38). Mức giảm này khớp với bài.
- [đo] Hộp DT6 giống hệt nhau ở cả ba hình của bài. Ở đây cũng vậy: kích thước và IC của DT6 trùng khít giữa ba cấu hình.
- DT6 cho size trung vị 3, bài 2, cùng khoảng tứ phân vị [2–3]. File đánh giá mẫu mà tác giả commit (10 cặp) cho trung vị 4 [tài liệu]. Chưa rõ vì sao lệch.
- **LORE không so được.** `lore_sa` là bản cài lại của thư viện LORE khác; luật dài hơn và có nhiều điều kiện "vắng mặt", như đã thấy ở thực nghiệm nhóm F. Không được viết là "tái hiện LORE".
- Lưu ý: hàm đếm của repo dùng so khớp chuỗi con (`aspect in rule`), nên tên aspect lồng nhau có thể bị đếm thừa. Ở đây dùng nguyên hàm đó để so được với bài.

## K5.
