# Chronic Kidney Disease (CKD) Prediction Model 🩺

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24-green) ![License](https://img.shields.io/badge/License-MIT-orange)

A machine learning project to **predict Chronic Kidney Disease (CKD)** using patient health metrics. The project includes **data preprocessing, EDA, model training, evaluation, and prediction on new patient data**.

---

## 🚀 Overview

This project leverages ML models to predict CKD based on clinical features such as age, blood pressure, hemoglobin, and more. The workflow is designed for **accuracy, interpretability, and reproducibility**.

**Notebook:** `Final_code_ckd.ipynb`

**Dataset:** 400 samples, 26 features, target variable = `ckd` / `notckd`.

![Workflow GIF](https://user-images.githubusercontent.com/your-username/workflow.gif)
*Example workflow illustration*

---

## 📊 Dataset

* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease) or Kaggle
* **Samples:** 400 patients
* **Features:** 26 (e.g., Age, Blood Pressure, Hemoglobin)
* **Target:** CKD classification (`ckd` or `notckd`)
* **File:** `kidney_disease.csv` (place in project directory or Google Drive for Colab)

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ckd-prediction-model.git
cd ckd-prediction-model
```

Install dependencies:

```bash
pip install matplotlib seaborn numpy pandas scikit-learn imbalanced-learn joblib
```

**For Google Colab:**

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🧩 Usage

Open `Final_code_ckd.ipynb` in **Jupyter** or **Colab**, and run sequentially to:

1. Load and explore the dataset
2. Preprocess data (handle missing values, encode categorical variables, scale features, balance classes using **SMOTE**)
3. Visualize data with EDA plots

![EDA Example](https://user-images.githubusercontent.com/your-username/eda_plot.png)
*Sample feature distribution plot*

4. Train and evaluate models (Logistic Regression, Naive Bayes, SVM, KNN)
5. Save the best model (e.g., `best_ckd_model_logistic_regression.pkl`)
6. Predict on new patient data

### Example: Predict on new data

```python
import joblib
import pandas as pd

model = joblib.load('best_ckd_model_logistic_regression.pkl')

new_patient = pd.DataFrame([{
    'Age': 55, 'Blood Pressure': 75, 'Specific Gravity': 2.030, 'Albumin': 0, 'Sugar': 1,
    'Blood Glucose Random': 125, 'Blood Urea': 33, 'Serum Creatinine': 0.3, 'Sodium': 140,
    'Potassium': 3.0, 'Hemoglobin': 15.0, 'Packed Cell Volume': 43, 'White Blood Cell Count': 3550,
    'Red Blood Cell Count': 3.0, 'Hypertension': 0, 'Diabetes Mellitus': 0, 'Coronary Artery Disease': 1,
    'Appetite': 0, 'Pedal Edema': 1, 'Anemia': 0, 'Red Blood Cells_normal': 0, 'Pus Cell_normal': 0,
    'Pus Cell Clumps_present': 1, 'Bacteria_present': 0
}])

predictions = model.predict(new_patient)
print(predictions)
```

---

## 📈 Key Features

* **Preprocessing:** Missing value imputation, categorical encoding, scaling, **SMOTE** for class balance
* **EDA:** Feature distributions, correlations, class balance visualization
* **Models:** Logistic Regression, Naive Bayes, SVM, KNN (with hyperparameter tuning)
* **Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
* **Best Model:** Logistic Regression (F1-Score: 1.0 on test set)

![Model Comparison](https://user-images.githubusercontent.com/your-username/model_comparison.png)
*Sample model comparison chart*

---

## 🏆 Model Performance

| Model                   | Accuracy | Precision | Recall | F1-Score |
| ----------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression     | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| Gaussian Naive Bayes    | 0.9750   | 0.9500    | 1.0000 | 0.9744   |
| Multinomial Naive Bayes | 0.9500   | 0.9048    | 1.0000 | 0.9500   |
| SVM                     | 0.9875   | 0.9756    | 1.0000 | 0.9877   |

---

## 📄 License

This project is licensed under the **MIT License**.

---

 

 




