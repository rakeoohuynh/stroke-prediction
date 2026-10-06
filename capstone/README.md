# Stroke Prediction – Dự đoán đột quỵ

Capstone project phân tích dữ liệu y tế và xây dựng mô hình machine learning dự đoán nguy cơ đột quỵ (stroke) dựa trên thông tin bệnh nhân.

## Dữ liệu

File: [`healthcare-dataset-stroke-data.csv`](healthcare-dataset-stroke-data.csv) (Healthcare Stroke Dataset, ~5.100 bệnh nhân).

| Cột | Mô tả |
|---|---|
| `id` | Mã bệnh nhân (bị loại bỏ khi huấn luyện) |
| `gender` | Giới tính |
| `age` | Tuổi |
| `hypertension` | Huyết áp cao (0/1) |
| `heart_disease` | Bệnh tim (0/1) |
| `ever_married` | Đã kết hôn hay chưa |
| `work_type` | Loại công việc |
| `Residence_type` | Nơi ở (Urban/Rural) |
| `avg_glucose_level` | Chỉ số đường huyết trung bình |
| `bmi` | Chỉ số khối cơ thể |
| `smoking_status` | Tình trạng hút thuốc |
| `stroke` | **Nhãn cần dự đoán** (1 = đột quỵ) |

Chỉ có 249 bệnh nhân bị đột quỵ (4,87%), nên dữ liệu mất cân bằng nghiêm trọng.

## Quy trình thực hiện

Toàn bộ nằm trong notebook [`stroke.ipynb`](stroke.ipynb):

1. **Làm sạch dữ liệu**: điền giá trị thiếu của `bmi` bằng trung bình, bỏ dòng trùng lặp, bỏ cột `id`.
2. **EDA**: phân tích tương quan, so sánh tuổi, đường huyết, BMI giữa nhóm đột quỵ và không đột quỵ.
3. **Xử lý outlier**: giữ lại các dòng có `bmi` < 80.
4. **Tiền xử lý**: bỏ `ever_married`, `Residence_type`, `work_type`; one-hot `smoking_status`; chia nhóm tuổi (0-17, 18-30, 31-50, 51-90); mã hóa `gender`.
5. **Chia dữ liệu**: train/test 80/20, `stratify=y`.
6. **Cân bằng lớp**: SMOTE chỉ áp dụng trên tập train, tập test giữ nguyên phân phối gốc.
7. **Mô hình**: Logistic Regression và Random Forest (300 cây).

## Kết quả EDA

| Yếu tố | Tương quan với đột quỵ |
|---|---|
| Tuổi | 0.245 |
| Bệnh tim | 0.135 |
| Đường huyết | 0.132 |
| Huyết áp cao | 0.128 |
| BMI | 0.039 |

Tuổi trung bình của bệnh nhân đột quỵ là **67,7**, trong khi tuổi trung bình toàn bộ dữ liệu là 43,2. Tuổi là yếu tố nguy cơ quan trọng nhất.

## Kết quả mô hình

Đánh giá trên tập test gốc (972 không đột quỵ / 50 đột quỵ):

| Chỉ số | Logistic Regression | Random Forest |
|---|---|---|
| Recall (đột quỵ) | **24%** | 16% |
| Precision (đột quỵ) | 11% | **17%** |
| ROC-AUC | 0.742 | **0.780** |
| Accuracy | 87% | 92% |

Với dữ liệu mất cân bằng, accuracy dễ gây hiểu nhầm, nên cần xem Recall của lớp đột quỵ và ROC-AUC.

## Kết luận

- Khi đánh giá đúng cách (chia train/test trước rồi mới SMOTE), cả hai mô hình vẫn còn yếu trong việc phát hiện đột quỵ.
- Random Forest có ROC-AUC cao hơn, Logistic Regression có recall cao hơn nhưng vẫn bỏ sót phần lớn ca đột quỵ.
- Hướng cải thiện: điều chỉnh ngưỡng quyết định, `class_weight`, thêm đặc trưng, cross-validation.
- Mô hình chưa đủ tin cậy để dùng trong thực tế y tế.

## Cách chạy

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
jupyter notebook stroke.ipynb
```

## Cấu trúc thư mục

```
capstone/
├── healthcare-dataset-stroke-data.csv   # dữ liệu
├── stroke.ipynb                         # EDA + huấn luyện mô hình
├── st.py                                # app Streamlit (đang phát triển)
└── helper.py
```
