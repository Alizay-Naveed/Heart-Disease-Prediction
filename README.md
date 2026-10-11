# 🫀 Heart Disease Prediction System

A secure, data-driven web interface built with **Streamlit** that utilizes a machine learning **Support Vector Regressor (SVR)** to predict the severity stage of heart disease based on localized clinical metrics.

## 🔗 Live Application Deployment
👉 **[Click Here to Access the Live Demo](https://heart-disease-prediction-model-019.streamlit.app/)**

---

## 🎯 Project Overview
This clinical screening dashboard utilizes the historical **UCI Cleveland Heart Disease dataset** parameters to estimate coronary narrowing on a continuous spectrum from **Stage 0 (Absence of Disease)** to **Stage 4 (Severe Obstruction)**. 

By employing an **RBF (Radial Basis Function) Kernel SVR**, the model maps complex, non-linear geometric distances between patient data points. It computes a high-resolution decimal severity index to capture real-world medical progression details, which the frontend presentations round safely to standard diagnostic integer blocks.

### 🧪 Extracted Core Predictors
The pipeline utilizes a high-yield, non-invasive 4-feature subset for optimization:
1. **`cp` (Chest Pain Type):** Multi-categorical structural diagnostic scale.
2. **`thalach` (Max Heart Rate):** Continuous peak heart rate verified during stress evaluation.
3. **`oldpeak` (ST Depression):** Numeric ECG anomaly calculation relative to resting points.
4. **`ca` (Major Vessels Count):** The exact number of vessels (0–3) colored by fluoroscopy.

---

## 📊 Model Performance Analytics

The machine learning backend was split into an **80/20 train/test profile** (locked via `random_state=3`) and normalized with a standalone `StandardScaler` blueprint to eliminate feature magnitude bias.

| Regression Performance Metric | Model Score Value | Operational Clinical Context |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | `0.567` | On average, predictions land within roughly half a severity stage of reality. |
| **Mean Squared Error (MSE)** | `0.696` | Heavily penalizes large variance mistakes to eliminate dangerous clinical blunders. |
| **R-squared Score (R²)** | `0.551` | Our 4 extracted features successfully capture **55.1%** of hidden biological variance. |

---

## 💻 Local Setup and Operations

Follow these steps to run the application workspace locally on your system:

### 1. Clone the Space Matrix
```bash
git clone https://github.com/Alizay-Naveed/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

### 2. Activate Dependencies
Ensure your terminal environment has Python installed, then install the framework libraries:
```bash
pip install -r requirements.txt
```

### 3. Initialize the Web Server
Launch the live local browser runtime using Streamlit:
```bash
streamlit run app.py
```

---

## 📁 Repository Structure
```text
├── app.py                     # Main Streamlit web browser user interface script
├── heart_disease_model.pkl    # Serialized SVR predictive backend model binary
├── scaler.pkl                 # Serialized StandardScaler training configuration
├── requirements.txt           # Deployment ecosystem dependency allocations
├── processed-cleveland.csv    # Clean master core source dataset
└── README.md                  # Project architectural blueprint document
```
