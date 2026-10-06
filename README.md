# Stroke Prediction

A capstone project that analyzes healthcare data and builds machine learning models to predict a patient's risk of stroke.

## Dataset

File: [`data/healthcare-dataset-stroke-data.csv`](data/healthcare-dataset-stroke-data.csv) (Healthcare Stroke Dataset, ~5,100 patients).

| Column | Description |
|---|---|
| `id` | Patient ID (dropped before training) |
| `gender` | Gender |
| `age` | Age |
| `hypertension` | Hypertension (0/1) |
| `heart_disease` | Heart disease (0/1) |
| `ever_married` | Marital status |
| `work_type` | Type of work |
| `Residence_type` | Residence (Urban/Rural) |
| `avg_glucose_level` | Average glucose level |
| `bmi` | Body mass index |
| `smoking_status` | Smoking status |
| `stroke` | **Target** (1 = stroke) |

Only 249 patients had a stroke (4.87%), so the dataset is severely imbalanced.

## Workflow

Everything is in the notebook [`stroke.ipynb`](stroke.ipynb):

1. **Data cleaning**: fill missing `bmi` values with the mean, drop duplicate rows, drop the `id` column.
2. **EDA**: correlation analysis, comparing age, glucose and BMI between stroke and non-stroke groups.
3. **Outlier handling**: keep rows with `bmi` < 80.
4. **Preprocessing**: drop `ever_married`, `Residence_type` and `work_type`; one-hot encode `smoking_status`; bin age (0-17, 18-30, 31-50, 51-90); encode `gender`.
5. **Split**: 80/20 train/test split with `stratify=y`.
6. **Class balancing**: SMOTE applied to the training set only; the test set keeps its original distribution.
7. **Models**: Logistic Regression and Random Forest (300 trees).

## EDA Findings

| Factor | Correlation with stroke |
|---|---|
| Age | 0.245 |
| Heart disease | 0.135 |
| Glucose level | 0.132 |
| Hypertension | 0.128 |
| BMI | 0.039 |

The average age of stroke patients is **67.7**, compared with 43.2 across the whole dataset, making age the most important risk factor.

## Model Results

Evaluated on the original test set (972 non-stroke / 50 stroke):

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Recall (stroke) | **24%** | 16% |
| Precision (stroke) | 11% | **17%** |
| ROC-AUC | 0.742 | **0.780** |
| Accuracy | 87% | 92% |

With imbalanced data, accuracy can be misleading, so stroke-class recall and ROC-AUC are the metrics to watch.

## Conclusion

- When evaluated properly (split first, then apply SMOTE), both models are still weak at detecting strokes.
- Random Forest has a higher ROC-AUC, while Logistic Regression has higher recall but still misses most stroke cases.
- Possible improvements: tune the decision threshold, use `class_weight`, add features, apply cross-validation.
- The models are not reliable enough for real-world medical use.

## Getting Started

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
jupyter notebook stroke.ipynb
```

## Project Structure

```
stroke-prediction/
├── data/
│   └── healthcare-dataset-stroke-data.csv   # dataset
├── stroke.ipynb                             # EDA + model training
└── README.md
```
