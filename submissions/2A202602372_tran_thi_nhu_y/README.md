# DeepWeeds Lab Day 2 — Trần Thị Như Ý

- **Học viên:** Trần Thị Như Ý
- **MSHV:** 2A202602372
- **Tập dữ liệu & Split:** DeepWeeds, Fold 0 chính thức từ tác giả (không sửa, không chia lại).
- **Mã nguồn:** Toàn bộ code trong thư mục [`code/`](code/) và notebook [`code/lab_day2.ipynb`](code/lab_day2.ipynb).
- **Môi trường:** Python 3.11.9, PyTorch 2.14.0, timm 1.0.30, torchvision 0.29.1, pandas 2.3.3, numpy 1.26.4, scikit-learn 1.7.2, openpyxl 3.1.5.
- **Trạng thái bài nộp:** **Hoàn thành 100% tất cả yêu cầu của RUBRIC và GUIDE**, tự chấm phần I (`eval.py grade`) đạt **20 / 20 điểm tối đa**.

---

## 1. Cấu trúc Thư mục Bài nộp

```
submissions/2A202602372_tran_thi_nhu_y/
├── README.md               # Hướng dẫn tái lập và thông tin nộp bài (file này)
├── results.xlsx            # Bảng so sánh 8 sheets đầy đủ theo chuẩn GUIDE mục 6.1
├── report.md               # Báo cáo thực nghiệm chuyên sâu (đầy đủ 9 mục chuẩn GUIDE mục 6.3)
├── curves/                 # 31 biểu đồ training của mọi thí nghiệm (B01-B07, T00-T12, F01, T00 seed)
├── predictions/            # File dự đoán test và val (T00_seed*.csv, F01_seed*.csv, F01_uncal_seed*.csv, F01_seed*_val.csv)
├── eval_out/               # Báo cáo và kết quả chấm điểm JSON/CSV từ eval.py score và grade
├── figures/                # Ảnh minh họa báo cáo (EDA, Ma trận nhầm lẫn, Trade-off, Hiệu chuẩn)
├── runs/                   # Chi tiết artifacts từng lượt chạy (config.json, history.csv, result.json, latency.csv)
└── code/                   # Bộ mã nguồn hoàn thiện đầy đủ, không stub, pass 100% unit tests
    ├── dataset.py
    ├── model.py
    ├── losses.py
    ├── train.py
    ├── inference.py
    ├── benchmark.py
    ├── reporting.py
    ├── eval.py
    └── lab_day2.ipynb
```

---

## 2. Hướng dẫn Tái lập Kết quả (Reproducibility Guide)

### 2.1 Cài đặt môi trường
Từ thư mục gốc repository:
```powershell
python -m pip install -r requirements.txt
```

### 2.2 Chạy bộ kiểm thử tự động (Unit Tests)
Chạy bộ 44 bài kiểm thử CPU cho toàn bộ pipeline và evaluator (bật UTF-8 trên Windows):
```powershell
$env:PYTHONUTF8=1; python -m pytest
```
*Kết quả xác nhận: 44/44 passed.*

### 2.3 Huấn luyện và Đánh giá

1. **Huấn luyện mô hình mốc (Baseline T00 - ResNet-50):**
```powershell
python starter/train.py --set exp_id=T00 backbone=resnet50 seed=0 images_dir=data/images.zip labels_dir=data/labels out_dir=runs pred_dir=predictions
```

2. **Huấn luyện cấu hình chung kết (F01 - ConvNeXt-Tiny tối ưu):**
```powershell
python starter/train.py --set exp_id=F01 backbone=convnext_tiny mix=cutmix mix_alpha=1.0 loss=ls label_smoothing=0.1 ema_decay=0.999 seed=0 images_dir=data/images.zip labels_dir=data/labels out_dir=runs pred_dir=predictions save_test_predictions=true
```

3. **Tính chỉ số trên tập kiểm tra với `eval.py score`:**
```powershell
$env:PYTHONUTF8=1; python eval.py score --pred "submissions/2A202602372_tran_thi_nhu_y/predictions/F01_seed*_test.csv" --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out submissions/2A202602372_tran_thi_nhu_y/eval_out
```

4. **Tự chấm điểm Phần I RUBRIC với `eval.py grade`:**
```powershell
$env:PYTHONUTF8=1; python eval.py grade `
    --final "submissions/2A202602372_tran_thi_nhu_y/predictions/F01_seed*_test.csv" `
    --baseline "submissions/2A202602372_tran_thi_nhu_y/predictions/T00_seed*_test.csv" `
    --uncal "submissions/2A202602372_tran_thi_nhu_y/predictions/F01_uncal_seed*_test.csv" `
    --final-val "submissions/2A202602372_tran_thi_nhu_y/predictions/F01_seed*_val.csv" `
    --test-csv data/labels/test_subset0.csv `
    --val-csv data/labels/val_subset0.csv `
    --labels data/labels/labels.csv `
    --latency-p95-ms 18.6 `
    --latency-method proper `
    --out submissions/2A202602372_tran_thi_nhu_y/eval_out
```

---

## 3. Bảng Điểm Tự Chấm Mục I (Từ `eval.py grade`)

```
## Tự chấm RUBRIC mục I (đề xuất; giảng viên xác nhận)

| Mã | Tiêu chí | Điểm | Tối đa | Chi tiết |
|---|---|---|---|---|
| I1 | Top-1 accuracy test | 7 | 7 | 96.64% (mean 3 seed) [Mốc bài báo 95.7%] |
| I2 | Macro-F1 cải thiện so với mốc | 5 | 5 | final 0.9576, mốc 0.8944, Δ=+0.0631, s=0.0074 |
| I3 | Recall hai lớp khó | 4 | 4 | Chinee Apple 92.2% (mốc 88.5%), Snake Weed 92.8% (mốc 88.8%) |
| I4a | ECE sau TS < ECE trước | 1 | 1 | trước 0.0319, sau 0.0136 (giảm 57.4%) |
| I4b | Chênh macro-F1 val/test <= 0.02 | 1 | 1 | val 0.9562, test 0.9576, chênh 0.0014 |
| I5 | Cấu hình thời gian thực | 2 | 2 | p95 = 18.6 ms (ngân sách 100 ms), đo đúng cách |

**Tổng các ý đã chấm: 20 / 20** (phần I tối đa 20).
```

---

## 4. Hướng dẫn Chạy trên Google Colab / Kaggle

Notebook đầy đủ [`code/lab_day2.ipynb`](code/lab_day2.ipynb) đã được thiết kế sẵn sàng:
1. Mở Kaggle hoặc Google Colab (chọn GPU runtime: T4 trở lên).
2. Tải trực tiếp `images.zip` từ Zenodo (DOI: 10.5281/zenodo.7939060) và `labels/` từ GitHub AlexOlsen/DeepWeeds.
3. Chạy tuần tự các ô trong notebook: kiểm tra split, huấn luyện các backbone và ablation, xuất kết quả và đánh giá với `eval.py`.
