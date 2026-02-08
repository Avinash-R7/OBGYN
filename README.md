# Early Pregnancy Risk Prediction using Machine Learning

## Overview
Early identification of maternal health risks is critical for preventing adverse pregnancy outcomes such as preterm birth.  
This project implements a machine learning–based **early pregnancy risk stratification system** using maternal clinical indicators, along with **human-readable explanations** for each prediction.

The system is designed for **clinical decision support**, not medical diagnosis.

---

## Problem Statement
To predict early pregnancy maternal risk levels associated with preterm birth using clinical indicators and explain the prediction in a human-readable form.

---

## Dataset
**Maternal Health Risk Dataset** (public dataset)

### Features used:
- Age
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Blood Sugar
- Body Temperature
- Heart Rate

### Target:
- RiskLabel  
  - 0 → Low Risk  
  - 1 → Mid Risk  
  - 2 → High Risk  

All features are available during **early pregnancy**, enabling early intervention.

---

## Methodology
1. Data preprocessing and validation  
2. Exploratory Data Analysis (EDA)  
3. Training multiple ML models  
4. Automatic selection of the best model based on **recall for high-risk cases**  
5. Probability-based risk prediction  
6. Text-based explainable AI for prediction reasoning  

---

## Model
- Final Model: **Random Forest Classifier**
- Selection Criterion: **High recall for high-risk cases**
- Reason: In healthcare, missing a high-risk case is more critical than false alarms

---

## Explainability
For every prediction, the system generates a **textual explanation** identifying the clinical factors responsible for the assigned risk level.

### Example Output:
> The model predicts MID RISK because SystolicBP is high (130.0).

This improves transparency and clinical trust.

---

## Application
A **Streamlit web application** allows users to:
- Enter maternal health parameters
- View predicted risk level
- Receive a human-readable explanation

---

## Project Structure
preterm-birth-risk-ml/
│
├── app.py
├── src/
│ ├── model.py
│ ├── explainability.py
│ ├── best_model.pkl
│ └── feature_importance.csv
│
├── notebooks/
│ └── 01_data_exploration.ipynb
│
├── data/
│ ├── raw/
│ └── processed/
│
├── README.md
├── requirements.txt
└── .gitignore

---

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

---

## Disclaimer
This system is intended for **early maternal risk assessment and clinical decision support only** and does not provide medical diagnosis.
