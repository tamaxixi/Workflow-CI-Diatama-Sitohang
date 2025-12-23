import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# LANGKAH 1 (AUTO)
# =========================
if not os.path.exists("credit_risk_preprocessing/X_processed.joblib"):
    import preprocessing

# =========================
# LANGKAH 2 (TRAINING)
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("credit-risk-ci")

X = joblib.load("credit_risk_preprocessing/X_processed.joblib")
y = joblib.load("credit_risk_preprocessing/y.joblib")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.sklearn.autolog()

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {acc}")
