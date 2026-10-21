"""So WAF và AUC theo fold của tầng dự đoán giữa bản lưu gốc và lần chạy lại trên Colab v4
(Python 3.13, scikit-learn 1.6.1, xgboost 3.2.0, shap 0.53.0).
Dùng: python3 tools/kiem_60_fold.py [results/main/results_prediction_v2.json] [results/environment/results_prediction_v2_from_notebook_v4.json]"""
import sys, json

A = sys.argv[1] if len(sys.argv) > 1 else "results/main/results_prediction_v2.json"
B = sys.argv[2] if len(sys.argv) > 2 else "results/environment/results_prediction_v2_from_notebook_v4.json"
a, b = json.load(open(A)), json.load(open(B))
for model in ("XGB300", "RF100"):
    keys = [k for k in a if k.endswith("|" + model) and not k.startswith("wilcoxon") and k in b]
    for field, name in (("folds", "WAF"), ("aucs", "AUC")):
        n = sum(len(a[k][field]) for k in keys)
        same = sum(x == y for k in keys for x, y in zip(a[k][field], b[k][field]))
        worst = max(abs(x - y) for k in keys for x, y in zip(a[k][field], b[k][field]))
        print(f"{model} {name}: {same}/{n} fold khớp tới 4 chữ số (lệch lớn nhất {worst:.4f}), {len(keys)} cấu hình")
for k in sorted(k for k in a if k.startswith("wilcoxon") and k in b):
    print(f"{k}: p = {a[k].get('p')} (gốc) / {b[k].get('p')} (v4)")
