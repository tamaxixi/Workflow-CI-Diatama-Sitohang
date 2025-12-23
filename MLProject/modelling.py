import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mlflow.set_experiment("credit-risk-ci")

# Load dataset (PASTIKAN ADA DI REPO)
X = joblib.load("credit_risk_preprocessing/X_processed.joblib")
y = joblib.load("credit_risk_preprocessing/y.joblib")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Autolog untuk logging parameter, metric, model
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc}")
