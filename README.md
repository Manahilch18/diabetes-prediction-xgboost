# Diabetes Prediction using XGBoost Classifier

A machine learning classification project that predicts whether a patient has **diabetes** based on diagnostic health measurements. Multiple modeling strategies were tried and compared, with a focus on **recall** — since in medical diagnosis, missing an actual positive (diabetic) case is far more costly than a false alarm.

## 📊 Overview

This project uses the Pima Indians Diabetes dataset (768 records, 9 columns) to build a binary classification model with `XGBoost`. Because the dataset is **imbalanced** (more non-diabetic than diabetic cases), a plain accuracy-optimized model tends to under-detect diabetic patients. So several class-imbalance handling techniques were tested to improve **recall on the diabetic class (1)**.

## 🎯 Project Workflow

1. **Import Libraries & Dataset** — Load data with `pandas`, `numpy`, `seaborn`, and `matplotlib`.
2. **Exploratory Data Analysis (EDA)** — Statistical summary, null-value checks, feature distributions, and correlation heatmap.
3. **Train/Test Split** — Split data into 80% training and 20% testing sets.
4. **Model Training (multiple approaches)**:
   - Baseline `XGBClassifier`
   - `XGBClassifier` with `scale_pos_weight` (class weighting)
   - `XGBClassifier` retrained on **SMOTE**-augmented data
   - **SMOTE + GridSearchCV** hyperparameter tuning
5. **Model Evaluation** — Compare all approaches using precision, recall, F1-score, and accuracy — with special attention to recall for Class 1 (Diabetic).

## 🗂️ Dataset

The dataset (`diabetes.csv`) contains 9 columns:

- `Pregnancies` — Number of times pregnant
- `Glucose` — Plasma glucose concentration
- `BloodPressure` — Diastolic blood pressure (mm Hg)
- `SkinThickness` — Triceps skin fold thickness (mm)
- `Insulin` — 2-Hour serum insulin (mu U/ml)
- `BMI` — Body mass index
- `DiabetesPedigreeFunction` — Diabetes likelihood based on family history
- `Age` — Age in years
- `Outcome` — Target variable (0 = Non-Diabetic, 1 = Diabetic)

## 🛠️ Tech Stack

- Python 3.10
- pandas, numpy
- seaborn, matplotlib (visualization)
- scikit-learn (train/test split, evaluation metrics, GridSearchCV)
- imbalanced-learn (SMOTE)
- XGBoost (classification model)

## ⚙️ Installation

```bash
git clone https://github.com/<Manahilch18>/<diabetes-prediction-xgboost>.git
cd <diabetes-prediction-xgboost>
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
seaborn
matplotlib
scikit-learn
imbalanced-learn
xgboost
```

## 🚀 Usage

Open and run `main.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab. Make sure `diabetes.csv` is in the same directory (or update the file path in the notebook).

```python
from xgboost import XGBClassifier

# Baseline model
xgb_classifier = XGBClassifier(
    objective='binary:logistic',
    eval_metric='error',
    learning_rate=0.1,
    max_depth=4,
    n_estimators=20
)
xgb_classifier.fit(X_train, y_train)
```

## 📈 Model Performance — Comparison of Approaches

> ⚠️ **Why recall matters here:** In a medical diagnosis context, a **false negative** (predicting "non-diabetic" for an actual diabetic patient) delays treatment and is far more dangerous than a **false positive** (flagging a healthy patient for further testing). So instead of chasing the highest accuracy, this project prioritizes maximizing **recall on Class 1 (Diabetic)** while keeping precision reasonable.

### 1️⃣ Baseline XGBoost Classifier

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Non-Diabetic) | 0.82 | 0.85 | 0.83 | 99 |
| 1 (Diabetic) | 0.71 | **0.65** | 0.68 | 55 |
| **Accuracy** | | | **0.78** | 154 |

High overall accuracy, but recall for diabetic patients is only 65% — meaning **35% of actual diabetic cases were missed**.

### 2️⃣ XGBoost with `scale_pos_weight`

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Non-Diabetic) | 0.85 | 0.70 | 0.77 | 99 |
| 1 (Diabetic) | 0.59 | **0.78** | 0.67 | 55 |
| **Accuracy** | | | **0.73** | 154 |

Recall for diabetic cases jumps from 0.65 → **0.78**, at the cost of lower precision and overall accuracy — an intentional and worthwhile trade-off for a medical use case.

### 3️⃣ XGBoost + SMOTE (Synthetic Oversampling)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Non-Diabetic) | 0.85 | 0.71 | 0.77 | 99 |
| 1 (Diabetic) | 0.60 | **0.78** | 0.68 | 55 |
| **Accuracy** | | | **0.73** | 154 |

Very similar results to `scale_pos_weight`, confirming that balancing the training data (either by weighting or oversampling) reliably improves diabetic-case recall to ~0.78.

### 4️⃣ SMOTE + GridSearchCV (Hyperparameter Tuning)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Non-Diabetic) | 0.83 | 0.69 | 0.75 | 99 |
| 1 (Diabetic) | 0.57 | **0.75** | 0.65 | 55 |
| **Accuracy** | | | **0.71** | 154 |

Tuning did not further improve recall over the plain SMOTE model in this run — precision, recall, and accuracy all dipped slightly. This suggests the chosen grid/parameters need revisiting (see Future Improvements).

### 📊 Summary Table

| Model | Accuracy | Precision (1) | Recall (1) | F1 (1) |
|-------|----------|----------------|------------|--------|
| Baseline XGBoost | 0.78 | 0.71 | 0.65 | 0.68 |
| XGBoost + `scale_pos_weight` | 0.73 | 0.59 | **0.78** | 0.67 |
| XGBoost + SMOTE | 0.73 | 0.60 | **0.78** | 0.68 |
| XGBoost + SMOTE + GridSearchCV | 0.71 | 0.57 | 0.75 | 0.65 |

**✅ Best model for this use case: XGBoost + SMOTE** — it matches `scale_pos_weight`'s recall (0.78) while giving a marginally better F1-score for the diabetic class, making it the preferred choice when the priority is catching as many true diabetic cases as possible.

**Confusion Matrix (Baseline model):**

```
                Predicted 0   Predicted 1
Actual 0             84            15
Actual 1             19            36
```

## 📁 Project Structure

```
├── main.ipynb          # Main notebook (EDA, model training/eval, all approaches)
├── diabetes.csv         # Dataset (Pima Indians Diabetes data)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🔮 Future Improvements

- Re-tune the GridSearchCV parameter grid — current tuned model underperforms the untuned SMOTE model, suggesting the search space needs adjustment
- Try threshold tuning (adjusting the default 0.5 decision threshold) to further boost recall without retraining
- Explore other imbalance techniques (ADASYN, class-weighted loss functions, ensemble/ ensemble ensembling)
- Add SHAP explainability to understand which features drive diabetic predictions
- Use stratified k-fold cross-validation instead of a single train/test split for more reliable metrics
- Compare against other models (Random Forest, Logistic Regression, LightGBM) under the same imbalance-handling strategies

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙌 Acknowledgements

Dataset sourced from the Pima Indians Diabetes Database (UCI Machine Learning Repository / Kaggle).
