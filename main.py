import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Step 1: Load Dataset
df = pd.read_csv("predictive_maintenance.csv")

# Step 2: Basic Info
print(df.head())
print(df.info())
print(df['failure'].value_counts())

# Step 3: Feature/Target Split
X = df.drop('failure', axis=1)
y = df['failure']

# Step 4: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train distribution:\n", y_train.value_counts())

# Step 6: Train Models
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Step 7: Evaluation Function
def evaluate_model(name, y_true, y_pred):
    print(f"\n📊 Evaluation: {name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# Step 8: Evaluate Both Models
evaluate_model("Logistic Regression", y_test, lr_preds)
evaluate_model("Random Forest", y_test, rf_preds)

def predict_failure(input_features):
    """
    Predict failure using Random Forest based on input features:
    input_features = [temperature, pressure, vibration, humidity, rpm]
    """
    input_scaled = scaler.transform([input_features])
    prediction = rf_model.predict(input_scaled)[0]

    if prediction == 1:
        print("\n🔧 Prediction: ❌ Failure is likely. Take preventive action!")
    else:
        print("\n🔧 Prediction: ✅ System is operating normally.")

# 🔍 Example usage:
example_input = [85.0, 6.2, 0.7, 48, 1320]  # high temp, vibration, low RPM
predict_failure(example_input)
