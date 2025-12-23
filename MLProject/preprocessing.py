import os
import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RAW_PATH = "credit_risk_raw/credit_risk_dataset.csv"
OUTPUT_DIR = "credit_risk_preprocessing"

def run_preprocessing():
    df = pd.read_csv(RAW_PATH)

    target = "loan_status"
    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
    X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(X_processed, f"{OUTPUT_DIR}/X_processed.joblib")
    joblib.dump(y, f"{OUTPUT_DIR}/y.joblib")
    joblib.dump(preprocessor, f"{OUTPUT_DIR}/preprocessor.joblib")
