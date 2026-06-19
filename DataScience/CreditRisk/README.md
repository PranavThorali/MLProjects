# Credit Risk Prediction using Machine Learning

## Project Overview

This project builds an end-to-end machine learning pipeline to predict whether a loan applicant is likely to default on a loan.

The objective is to help financial institutions make better lending decisions by identifying high-risk borrowers using historical credit and demographic information.

The project covers the complete Data Science workflow:

* Data Loading
* Data Auditing
* Exploratory Data Analysis (EDA)
* Data Cleaning
* Feature Engineering
* Data Preprocessing
* Logistic Regression
* Cross Validation
* Random Forest
* XGBoost
* Model Evaluation
* Feature Importance Analysis

---

## Business Problem

Financial institutions face significant losses due to loan defaults.

The goal of this project is to predict:

**Loan Status**

* `0` → No Default
* `1` → Default

This allows lenders to:

* Reduce financial risk
* Improve loan approval decisions
* Identify high-risk applicants early
* Optimize lending strategies

---

## Dataset

Dataset used:

**Credit Risk Dataset**

Features include:

| Feature                    | Description               |
| -------------------------- | ------------------------- |
| person_age                 | Borrower's age            |
| person_income              | Annual income             |
| person_home_ownership      | Home ownership status     |
| person_emp_length          | Employment length         |
| loan_intent                | Purpose of loan           |
| loan_grade                 | Loan risk grade           |
| loan_amnt                  | Loan amount requested     |
| loan_int_rate              | Interest rate             |
| cb_person_default_on_file  | Historical default record |
| cb_person_cred_hist_length | Credit history length     |
| loan_status                | Target variable           |

---

## Project Workflow

### 1. Data Auditing

Performed:

* Dataset inspection
* Missing value analysis
* Duplicate detection
* Target distribution analysis

```python
df.isnull().sum()
df.duplicated().sum()
df["loan_status"].value_counts()
```

---

### 2. Exploratory Data Analysis (EDA)

Analyzed:

* Income distribution
* Loan amount distribution
* Age distribution
* Default vs Non-default behavior

Examples:

```python
df.groupby("loan_status")["person_income"].mean()
df.groupby("loan_status")["loan_amnt"].mean()
```

---

### 3. Data Cleaning

Performed:

* Missing value imputation
* Removal of unrealistic age values
* Data quality checks

Example:

```python
df["person_emp_length"].fillna(
    df["person_emp_length"].median(),
    inplace=True
)
```

---

### 4. Feature Engineering

Created additional predictive features:

#### Loan-to-Income Ratio

```python
loan_to_income_ratio =
loan_amnt / person_income
```

#### Income-to-Credit-History Ratio

```python
income_credit_hist_ratio =
person_income /
(cb_person_cred_hist_length + 1)
```

These engineered features help capture borrower risk more effectively.

---

### 5. Data Preprocessing

Categorical variables were converted using one-hot encoding.

```python
pd.get_dummies(df, drop_first=True)
```

Data was then split into:

* Training Set (80%)
* Testing Set (20%)

using stratified sampling.

---

### 6. Baseline Model: Logistic Regression

A Logistic Regression model was used as the baseline classifier.

Steps:

* Missing value handling
* Standardization using StandardScaler
* Model training
* Prediction

```python
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
```

---

### 7. Model Evaluation

Evaluation metrics used:

* Confusion Matrix
* Precision
* Recall
* F1 Score
* ROC-AUC

Example:

```python
classification_report(y_test, preds)
roc_auc_score(y_test, probs)
```

---

### 8. Cross Validation

To ensure model stability, 5-fold cross-validation was performed.

```python
cross_val_score(
    lr,
    X_train_scaled,
    y_train,
    cv=5
)
```

---

### 9. Random Forest Classifier

A Random Forest model was trained and compared against Logistic Regression.

Benefits:

* Handles non-linear relationships
* Less sensitive to scaling
* Provides feature importance

```python
rf = RandomForestClassifier(
    random_state=42
)
```

---

### 10. XGBoost Classifier

An XGBoost model was trained to improve predictive performance.

Benefits:

* Gradient Boosting
* Strong performance on tabular data
* Handles complex feature interactions

```python
xgb = XGBClassifier(
    random_state=42
)
```

---

### 11. Feature Importance

Feature importance was extracted from:

* Random Forest
* XGBoost

This helps identify the most influential factors behind loan default risk.

Examples:

* Loan-to-Income Ratio
* Income
* Credit History Length
* Previous Defaults

---

## Machine Learning Models Used

| Model               | Purpose           |
| ------------------- | ----------------- |
| Logistic Regression | Baseline Model    |
| Random Forest       | Ensemble Learning |
| XGBoost             | Gradient Boosting |

---

## Evaluation Metrics

Since credit-risk problems are often imbalanced, multiple metrics were used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Special attention was given to:

### Recall

Because failing to identify a risky borrower can result in significant financial losses.

---

## Key Learnings

This project demonstrates:

* End-to-end Machine Learning workflow
* Classification modeling
* Feature engineering
* Handling missing values
* Data preprocessing
* Cross-validation
* Ensemble methods
* Credit risk analysis
* Model comparison and evaluation

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* XGBoost
* Jupyter Notebook

---

## Repository Structure

```text
├── Credit_Data_Manipulation.ipynb
├── README.md
├── requirements.txt
└── data/
    └── credit_risk_dataset.csv
```

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV
* SMOTE for class imbalance handling
* SHAP explainability
* Streamlit deployment
* Model persistence using Pickle/Joblib

---

## Author

Built as part of a hands-on Machine Learning and Data Science learning project focused on credit risk modeling and binary classification.
