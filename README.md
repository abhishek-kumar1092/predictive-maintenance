# 🔧 Predictive Maintenance using Machine Learning

This project is built as part of the **ML Intern application for Xempla**.  
It demonstrates a complete machine learning pipeline for predicting equipment failure using structured sensor data.

---

## 📌 Objective

To build a system that predicts whether a machine is likely to fail based on key sensor readings like temperature, pressure, vibration, humidity, and RPM.

---

## 🧠 Tech Stack

- Python 🐍
- Pandas & NumPy (Data Handling)
- Matplotlib & Seaborn (EDA/Visualization)
- Scikit-learn (Modeling)

---

## 📁 Dataset

The dataset contains 10 samples of simulated machine sensor readings with binary labels for failure.

| Feature       | Description                     |
|---------------|---------------------------------|
| temperature   | Internal temperature (°C)       |
| pressure      | Machine pressure (bar)          |
| vibration     | Vibration level                 |
| humidity      | Ambient humidity (%)            |
| rpm           | Rotations per minute            |
| failure       | Target (1 = failure, 0 = normal)|

---

## ✅ Features Implemented

- Clean feature engineering and scaling
- Logistic Regression and Random Forest models
- Evaluation using Accuracy, F1 Score, Confusion Matrix
- Custom prediction function:  
  ```python
  predict_failure([temperature, pressure, vibration, humidity, rpm])
