
# Credit Card Fraud Detection

### An End-to-End Machine Learning Project for Identifying Fraudulent Credit Card Transactions

---

## 1. Project Overview

Credit card fraud is one of the major challenges facing
modern financial institutions and digital payment systems.

As electronic transactions continue to grow, detecting
fraudulent activities accurately and efficiently becomes
increasingly important.

This project develops an end-to-end Machine Learning
workflow for identifying potentially fraudulent credit
card transactions.

The project focuses on data understanding, data quality,
exploratory data analysis, feature engineering, preprocessing,
class imbalance handling, model development, hyperparameter
optimization, evaluation, and model explainability.

The primary objective is to build a reliable classification
pipeline that can distinguish between legitimate and
fraudulent transactions while carefully considering the
highly imbalanced nature of the dataset.

---

## 2. Problem Statement

Credit card fraud detection is a binary classification problem.

Each transaction is assigned one of two classes:

- Class 0: Legitimate transaction.
- Class 1: Fraudulent transaction.

A major challenge is that fraudulent transactions represent
a very small proportion of all transactions.

Consequently, conventional classification accuracy can be
misleading.

For example, a model that predicts every transaction as
legitimate may achieve very high accuracy while completely
failing to identify fraud.

Therefore, this project emphasizes evaluation metrics that
provide a more meaningful assessment of fraud detection
performance.

---

## 3. Project Objectives

The main objectives are:

1. Understand the structure and characteristics of the dataset.
2. Identify data quality issues and potential inconsistencies.
3. Perform exploratory data analysis.
4. Develop meaningful features.
5. Build a reproducible preprocessing pipeline.
6. Investigate class imbalance handling techniques.
7. Train and compare Machine Learning classifiers.
8. Optimize model hyperparameters.
9. Select an appropriate classification threshold.
10. Evaluate model performance using relevant metrics.
11. Explain model predictions using explainability techniques.
12. Provide a reusable prediction interface.

---

## 4. Dataset Description

The project uses the Credit Card Fraud Detection dataset.

The dataset contains anonymized transaction features
and a binary target variable.

### Dataset Features

| Feature | Description |
|---|---|
| Time | Seconds elapsed between this transaction and the first transaction in the dataset |
| V1 - V28 | Anonymized numerical features obtained through PCA transformation |
| Amount | Transaction amount |
| Class | Target variable indicating whether the transaction is fraudulent |

### Target Variable

- 0: Legitimate transaction.
- 1: Fraudulent transaction.

The original feature set consists of 30 input features
and one target variable.

The dataset is highly imbalanced, which makes fraud
detection a challenging classification task.

### Dataset Source

The dataset is commonly distributed through the
Kaggle Credit Card Fraud Detection dataset.

Dataset reference:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset is not included in this repository by default.
Download it separately and place the CSV file in:

data/raw/creditcard.csv

---

## 5. Project Methodology

The project follows a structured Machine Learning workflow.

### Stage 1: Problem Understanding

- Define the fraud detection problem.
- Identify the target variable.
- Understand business and modeling objectives.
- Establish suitable evaluation criteria.

### Stage 2: Data Understanding

- Load the raw dataset.
- Inspect dimensions and data types.
- Examine descriptive statistics.
- Analyze the target distribution.
- Investigate feature characteristics.

### Stage 3: Data Cleaning

- Inspect missing values.
- Identify duplicate records.
- Validate numerical columns.
- Check invalid and infinite values.
- Preserve the original dataset.

### Stage 4: Exploratory Data Analysis

- Analyze class distribution.
- Examine transaction amounts.
- Explore transaction time.
- Investigate numerical feature distributions.
- Study relationships between features and the target.

### Stage 5: Feature Engineering

The project introduces two additional features:

- Amount_log1p
- Time_log1p

These transformations use the natural logarithm
of one plus the original feature value.

The project deliberately excludes Time_Diff to avoid
introducing assumptions about transaction ordering
and inference-time temporal context.

### Stage 6: Data Preprocessing

The preprocessing workflow is implemented using
reusable Scikit-learn components.

Depending on the selected model, preprocessing may include:

- Missing value imputation.
- Feature scaling.
- Numerical transformations.
- Feature engineering.

All learned preprocessing operations are fitted
using training data only.

### Stage 7: Class Imbalance Handling

The dataset contains substantially fewer fraudulent
transactions than legitimate transactions.

The project investigates imbalance handling strategies,
including:

- Random Oversampling.
- SMOTE.
- Class-weighted learning.

Resampling is performed within the training pipeline.

Validation and test datasets are never oversampled.

### Stage 8: Model Training

The project provides a reusable model training workflow.

Candidate classifiers may include:

- Logistic Regression.
- Random Forest.
- XGBoost.

Model comparison is based on fraud detection metrics
rather than accuracy alone.

### Stage 9: Hyperparameter Optimization

Hyperparameter tuning is used to investigate
model configurations and improve generalization.

Cross-validation is performed within the training
workflow to reduce information leakage.

### Stage 10: Model Evaluation

The final evaluation considers:

- Precision.
- Recall.
- F1-Score.
- ROC-AUC.
- Average Precision.
- Confusion Matrix.

The classification threshold is selected using
validation data.

The test set is reserved for final evaluation.

### Stage 11: Model Explainability

Explainability techniques are used to investigate
which features influence model predictions.

SHAP may be used to analyze:

- Global feature importance.
- Individual transaction explanations.
- Feature contribution patterns.

Model explanations describe learned associations
and should not be interpreted as causal effects.

---

## 6. Machine Learning Evaluation

Accuracy alone is not sufficient for evaluating
highly imbalanced fraud detection datasets.

The project focuses on the following metrics:

| Metric | Purpose |
|---|---|
| Precision | Measures how many predicted fraud cases are actually fraudulent |
| Recall | Measures how many actual fraud cases are detected |
| F1-Score | Balances precision and recall |
| ROC-AUC | Measures ranking performance across classification thresholds |
| Average Precision | Summarizes the precision-recall curve |
| Confusion Matrix | Shows correct and incorrect classifications |

The selection of an operating threshold should consider
the trade-off between false positives and false negatives.

No performance figures are reported in this README
until the final experiment has been executed.

---

## 7. Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_problem_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_preprocessing.ipynb
│   ├── 07_imbalance_handling.ipynb
│   ├── 08_model_training.ipynb
│   ├── 09_hyperparameter_tuning.ipynb
│   ├── 10_model_evaluation.ipynb
│   └── 11_model_explainability.ipynb
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── validation.py
│   ├── cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── imbalance.py
│   ├── train.py
│   ├── tuning.py
│   ├── evaluate.py
│   ├── explainability.py
│   └── predict.py
│
├── models/
├── reports/
│   ├── figures/
│   └── metrics/
│
├── app/
│   └── app.py
│
├── tests/
│   ├── test_data.py
│   └── test_features.py
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## 8. Technologies and Libraries

The project uses the following technologies:

### Programming Language

- Python

### Data Manipulation

- NumPy
- Pandas
- SciPy

### Data Visualization

- Matplotlib
- Seaborn
- Plotly

### Machine Learning

- Scikit-learn
- Imbalanced-learn
- XGBoost

### Model Explainability

- SHAP

### Model Persistence

- Joblib

### Interactive Application

- Streamlit

### Development and Testing

- Jupyter Notebook
- JupyterLab
- Pytest

---

## 9. Installation and Setup

### Prerequisites

- Python installation.
- Git.
- Access to the Credit Card Fraud Detection dataset.

### Step 1: Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
```

### Step 2: Navigate to the Project Directory

```bash
cd Credit-Card-Fraud-Detection
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Download the Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the downloaded CSV file at:

```text
data/raw/creditcard.csv
```

### Step 5: Run the Main Training Pipeline

```bash
python main.py
```

### Step 6: Run the Streamlit Application

```bash
python -m streamlit run app/app.py
```

### Step 7: Run Unit Tests

```bash
python -m pytest tests/ -v
```

---

## 10. Reproducibility and Data Leakage Prevention

Reproducibility and data leakage prevention are
important considerations throughout this project.

The following practices are adopted:

- A fixed random state is used for reproducibility.
- Training, validation, and test datasets are separated.
- Preprocessing parameters are learned from training data.
- SMOTE is applied only within the training pipeline.
- Cross-validation is used during model optimization.
- The classification threshold is selected on validation data.
- The test set is reserved for final evaluation.

These practices help provide a more reliable estimate
of model generalization performance.

---

## 11. Limitations

This project has several limitations:

1. The dataset contains anonymized PCA-transformed features,
   which limits direct business interpretation.
2. Historical dataset performance does not guarantee
   performance on future real-world transactions.
3. Fraud patterns may change over time.
4. Classification threshold selection involves a trade-off
   between false positives and false negatives.
5. Model explainability does not establish causal relationships.
6. The project is an experimental fraud detection system
   and is not intended to authorize or reject real payments.

---

## 12. Future Improvements

Potential future extensions include:

- Time-aware model validation.
- Cost-sensitive learning.
- Advanced ensemble models.
- Model monitoring and drift detection.
- Real-time prediction services.
- Automated model retraining.
- More comprehensive unit and integration tests.
- Deployment through a production-ready API.
- Business-oriented fraud investigation dashboards.

---

## 13. Disclaimer

This project is intended for educational, research,
and experimental Machine Learning purposes.

It should not be used as the sole decision-making
system for real financial transactions.

Any production deployment would require additional
validation, security controls, monitoring, governance,
and business risk assessment.

---

## 14. Author

**Ameen Ali**

Machine Learning | Data Science | Quantitative Finance

GitHub: [ameenalitech](https://github.com/ameenalitech)

LinkedIn: [ameenaliofficial](https://www.linkedin.com/in/ameenaliofficial/)

---

## 15. License

This project is distributed under the license
included in the repository.
