import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

#loading the dataset
df = pd.read_csv("predictive_maintenance_balanced.csv")

print(df.head())
print(df.info())
print(df['failure'].value_counts())

#Feature/Target Split
X = df.drop('failure', axis=1)
y = df['failure']

#Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train distribution:\n", y_train.value_counts())

#Training Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

#Tuning Random Forest with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
best_rf_preds = best_rf_model.predict(X_test)

#Evaluation Function/Metrics
def evaluate_model(name, y_true, y_pred):
    print(f"\n📊 Evaluation: {name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

#Evaluate Models
evaluate_model("Logistic Regression", y_test, lr_preds)
evaluate_model("Random Forest (Tuned)", y_test, best_rf_preds)

#Real-time Prediction Function
def predict_failure(input_features):
    """
    Predict failure using the best tuned Random Forest model.
    input_features = [temperature, pressure, vibration, humidity, rpm]
    """
    input_scaled = scaler.transform([input_features])
    prediction = best_rf_model.predict(input_scaled)[0]

    if prediction == 1:
        print("\n🔧 Prediction: ❌ Failure is likely. Take preventive action!")
    else:
        print("\n🔧 Prediction: ✅ System is operating normally.")

#Example usage:
example_input = [85.0, 6.2, 0.7, 48, 1320]  # High temp, pressure, vibration
predict_failure(example_input)
