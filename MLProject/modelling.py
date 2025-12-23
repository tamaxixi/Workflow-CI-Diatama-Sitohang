import os
import joblib
import mlflow
import mlflow.sklearn

if not os.path.exists("credit_risk_preprocessing/X_processed.joblib"):
    import preprocessing
    preprocessing.run_preprocessing()

X = joblib.load("credit_risk_preprocessing/X_processed.joblib")
y = joblib.load("credit_risk_preprocessing/y.joblib")
