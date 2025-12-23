import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("credit-risk-ci")

# 2. Pastikan preprocessing jalan
if not os.path.exists("credit_risk_preprocessing/X_processed.joblib"):
    import preprocessing
    preprocessing.run_preprocessing()

# 3. Load hasil preprocessing
X = joblib.load("credit_risk_preprocessing/X_processed.joblib")
y = joblib.load("credit_risk_preprocessing/y.joblib")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.sklearn.autolog()

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    print(f"Accuracy: {acc}")
