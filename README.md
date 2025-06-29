# 🔧 Predictive Maintenance using Machine Learning

This is a submission for the **ML Intern role at Xempla**.  
The project predicts whether a machine is likely to fail based on real-time sensor data, using a complete ML pipeline built in Python.

---

## 📌 Objective

To develop a machine learning model that can predict maintenance issues in advance using structured data like temperature, pressure, vibration, humidity, and RPM.

---

## 🧠 Tech Stack

- Python 🐍
- pandas & numpy (Data handling)
- scikit-learn (Preprocessing, modeling, tuning)
- matplotlib & seaborn (Exploratory Data Analysis)

---

## 🗂️ Dataset

A realistic synthetic dataset with 100 samples:
- 50 normal (failure = 0)
- 50 failure (failure = 1)

Each row contains:
- `temperature`, `pressure`, `vibration`, `humidity`, `rpm`, and `failure`

---

## ✅ Features Implemented

- Preprocessing using `StandardScaler`
- Train/test split using `train_test_split`
- Models: Logistic Regression & Random Forest
- Random Forest tuning using `GridSearchCV`
- Evaluation using accuracy, F1 score, confusion matrix, and report
- Live prediction function:

```python
predict_failure([temperature, pressure, vibration, humidity, rpm])
