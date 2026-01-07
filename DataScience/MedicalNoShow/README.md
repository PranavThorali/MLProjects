# Medical Appointment No-Show Prediction

## Project Overview

This project predicts patient no-shows for medical appointments using machine learning. The goal is to help healthcare facilities reduce revenue loss by identifying high-risk appointments and implementing targeted interventions.

## Problem Statement

Medical appointment no-shows are a significant problem for healthcare providers, leading to:
- Lost revenue from unused appointment slots
- Reduced operational efficiency
- Poor resource allocation
- Patient care delays

By predicting which patients are likely to miss appointments, healthcare facilities can implement proactive interventions (such as SMS reminders, phone calls, or rescheduling) to reduce no-show rates.

## Dataset

The project uses a dataset containing **106,987 medical appointments** with the following key features:
- Patient demographics (Age, Gender, Neighbourhood)
- Appointment details (Scheduled date, Appointment date, Days between)
- Health conditions (Hypertension, Diabetes, Alcoholism, Handicap)
- Program enrollment (Scholarship)
- Communication (SMS received)
- Target variable: No-show status

## Project Structure

The project is organized into three main notebooks:

### 1. `01-data-cleaning.ipynb`
- Data loading and exploration
- Column renaming and standardization
- Data type conversions
- Missing value analysis
- Feature engineering:
  - Day of week extraction
  - Chronic condition flags
  - Time-based features
- Data validation and cleaning
- Output: `clean_noshows.csv`

### 2. `02-modelling.ipynb`
- Model training and evaluation
- Algorithms tested:
  - **Logistic Regression**: Baseline model with balanced class weights
  - **Random Forest**: Primary model with 200 estimators and max depth of 12
- Model evaluation metrics:
  - Classification report (precision, recall, F1-score)
  - ROC-AUC score
- Risk score generation
- Output: `model_scored_noshows.csv`

### 3. `03-business-impact.ipynb`
- Business impact analysis
- Risk level categorization:
  - **High Risk** (score ≥ 0.7): 40% reduction potential
  - **Medium Risk** (0.4 ≤ score < 0.7): 25% reduction potential
  - **Low Risk** (score < 0.4): 5% reduction potential
- Cost-benefit analysis:
  - Baseline vs. post-intervention no-show rates
  - Annual revenue impact calculations
  - Intervention cost estimation
  - Net benefit calculation
- Output: `dashboard_data.csv`

## Key Results

### Model Performance
- **Random Forest** achieved superior performance with balanced precision and recall
- ROC-AUC score demonstrates good predictive capability
- Model effectively identifies high-risk no-show patients

### Business Impact
- **Baseline no-show rate**: ~20%
- **Post-intervention no-show rate**: ~14%
- **Annual revenue recovered**: ~$300,000
- **Annual intervention cost**: ~$25,000
- **Net benefit**: ~$275,000

*Note: Calculations assume 100 appointments/day, $150 average appointment value, and 260 working days/year*

## Files Description

- `healthcare_noshows_appt.csv`: Original raw dataset
- `clean_noshows.csv`: Cleaned and feature-engineered dataset
- `model_scored_noshows.csv`: Dataset with model predictions and risk scores
- `dashboard_data.csv`: Aggregated data for dashboard visualization

## Usage

### Prerequisites
```python
pandas
numpy
scikit-learn
```

### Running the Project

1. **Data Cleaning**: Run `01-data-cleaning.ipynb` to clean and prepare the data
2. **Modeling**: Run `02-modelling.ipynb` to train models and generate predictions
3. **Business Analysis**: Run `03-business-impact.ipynb` to analyze business impact

### Expected Workflow
```
healthcare_noshows_appt.csv 
    → [01-data-cleaning.ipynb] 
    → clean_noshows.csv 
    → [02-modelling.ipynb] 
    → model_scored_noshows.csv 
    → [03-business-impact.ipynb] 
    → dashboard_data.csv
```

## Key Features

- **Comprehensive data cleaning** with feature engineering
- **Multiple model comparison** (Logistic Regression vs. Random Forest)
- **Business-focused analysis** with ROI calculations
- **Risk stratification** for targeted interventions
- **Actionable insights** for healthcare operations

## Future Enhancements

- Real-time prediction API
- Integration with appointment scheduling systems
- A/B testing framework for intervention strategies
- Advanced feature engineering (seasonality, patient history)
- Deep learning models for improved accuracy

## Author

Pranav Thorali

## License

This project is part of the MLProjects repository.

