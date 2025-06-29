# 🔧 Predictive Maintenance using Machine Learning

This is a submission for the **ML Intern role at Xempla**.  
The project predicts whether a machine is likely to fail based on real-time sensor data, using a complete ML pipeline built in Python.

## 📌 Objective

To develop a machine learning model that can predict maintenance issues in advance using structured data like temperature, pressure, vibration, humidity, and RPM.

## 🧠 Tech Stack

- Python 🐍
- pandas & numpy (Data handling)
- scikit-learn (Preprocessing, modeling, tuning)
- matplotlib & seaborn (Exploratory Data Analysis)

## 🗂️ Dataset

A realistic synthetic dataset with 100 samples:
- 50 normal (failure = 0)
- 50 failure (failure = 1)

Each row contains:
- temperature, pressure, vibration, humidity, rpm, and failure

## ✅ Features Implemented

- Preprocessing using StandardScaler
- Train/test split using train_test_split
- Models: Logistic Regression & Random Forest
- Random Forest tuning using GridSearchCV
- Evaluation using accuracy, F1 score, confusion matrix, and report
- Live prediction function:

python
predict_failure([temperature, pressure, vibration, humidity, rpm])


## 🔍 Explainability

- **Feature Importance:** The Random Forest model provides feature importance scores, helping maintenance teams understand which sensor readings most influence the failure prediction.
- **Model Interpretation:** Tools like SHAP or LIME can be integrated to visualize and explain individual predictions, increasing trust and transparency for human operators.
- **Decision Support:** Clear explanations enable maintenance teams to make informed decisions based on model outputs, rather than relying on black-box predictions.

## 🏭 Real-World Data & Scalability

- **Handling Noisy Data:** In real-world deployments, sensor data can be noisy or incomplete. The pipeline is designed to handle missing values and outliers.
- **Scaling Up:** For larger datasets, techniques like incremental learning and cloud-based deployment (e.g., AWS SageMaker) can be used to ensure scalability.
- **Class Imbalance:** In production, methods such as SMOTE or cost-sensitive learning would be applied to address class imbalance, which is common in failure prediction tasks.

## 🤝 Decision Intelligence & Human-in-the-Loop

- **Actionable Outputs:** When predict_failure() returns 1, the system can automatically trigger a maintenance ticket or alert, helping teams prioritize interventions.
- **Human-in-the-Loop:** The model is designed to support—not replace—human operators. Maintenance staff can review model explanations before taking action, ensuring that AI augments decision-making rather than automating it blindly.

*For questions or collaboration, feel free to reach out!*