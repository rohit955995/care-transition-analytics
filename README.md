# 🏥 Care Transition Efficiency & Placement Outcome Analytics

An interactive healthcare analytics and machine learning dashboard for monitoring the care transition pipeline from CBP custody to HHS care and final placement.

---

## 📌 Project Overview

This project analyzes historical care-transition data to understand how children move through different stages of the care pipeline.

The system combines:

- Data preprocessing
- Healthcare KPI analysis
- Interactive visualization
- Machine Learning
- HHS care prediction
- Feature importance analysis
- Automated insights

The final application is built using Streamlit.

---

## 🎯 Objectives

The main objectives of this project are:

- Monitor the care transition pipeline
- Analyze CBP custody trends
- Analyze HHS care trends
- Measure transfer efficiency
- Measure discharge effectiveness
- Monitor active care load
- Predict HHS care requirements
- Identify important factors affecting HHS care

---

## 📊 Key Performance Indicators

The dashboard calculates several important KPIs:

### 🔄 Transfer Efficiency

Measures the proportion of children transferred out of CBP custody.

### 🏠 Discharge Effectiveness

Measures the relationship between children discharged and children in HHS care.

### 👮 CBP Backlog

Tracks the number of children currently in CBP custody.

### 🏥 HHS Care Load

Tracks the number of children currently receiving HHS care.

### 🚚 Pipeline Throughput

Measures the overall movement through the care-transition pipeline.

### 👥 Total Active Care Load

Combines CBP custody and HHS care populations.

---

## 🤖 Machine Learning

A **Random Forest Regressor** is used to estimate the HHS care load.

### Input Features

The model uses:

- Children apprehended and placed in CBP custody
- Children in CBP custody
- Children transferred out of CBP custody
- Children discharged from HHS care

### Target

```text
Children in HHS Care


Model Evaluation
The model is evaluated using:
MAE
RMSE
R² Score
Feature importance is also calculated to understand which variables contribute most to the prediction.
📈 Dashboard Features
The Streamlit dashboard provides:
📊 KPI Cards
Quick overview of major healthcare metrics.
🔄 Care Transition Pipeline
Visualizes the average number of children at different transition stages.
📈 Backlog Monitoring
Tracks CBP and HHS care loads over time.
⚡ Efficiency Analysis
Shows transfer and discharge efficiency trends.
🚚 Pipeline Throughput
Visualizes daily transition throughput.
💡 Automated Insights
Highlights important trends and maximum observed values.
🤖 ML Prediction
Displays predicted HHS care load.
🎯 Feature Importance
Shows which input features have the greatest influence on the Random Forest model.
🛠️ Tech Stack
Technology	Purpose
Python	Core programming
Pandas	Data processing
NumPy	Numerical operations
Scikit-learn	Machine Learning
Plotly	Interactive visualization
Streamlit	Dashboard development
📂 Project Structure
care-transition-analytics/
│
├── data/
│   └── HHS_Unaccompanied_Alien_Children_Program.csv
│
├── analysis.py
├── ml_model.py
├── app.py
├── requirements.txt
├── README.md
└── venv/