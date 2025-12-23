import os
import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    target = "loan_status"

    X = df.drop(columns=[target])
    y = df[target]

    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor


def save_artifacts(X, y, preprocessor, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(X, os.path.join(output_dir, "X_processed.joblib"))
    joblib.dump(y, os.path.join(output_dir, "y.joblib"))
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))


if __name__ == "__main__":
    RAW_PATH = "credit_risk_raw/credit_risk_dataset.csv"
    OUTPUT_DIR = "credit_risk_preprocessed"

    df = load_data(RAW_PATH)
    X_processed, y, preprocessor = preprocess_data(df)
    save_artifacts(X_processed, y, preprocessor, OUTPUT_DIR)

    print("Preprocessing selesai. Artefak tersimpan.")
