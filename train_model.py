import sqlite3
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

DB_NAME = "student_performance.db"
MODEL_DIR = "models/saved_models"

os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM students", conn)
    conn.close()
    return df

def preprocess_data(df):
    X = df[["attendance", "internal_marks", "assignment_score"]]
    y = df["result"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    return X, y_encoded, encoder

if __name__ == "__main__":
    df = load_data()
    X, y, encoder = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    model_path = f"{MODEL_DIR}/logistic_regression_v1.pkl"
    joblib.dump({
        "model": model,
        "encoder": encoder
    }, model_path)

    print(f" Logistic Regression model trained successfully")
    print(f" Model accuracy: {accuracy:.2f}")
    print(f" Model saved at: {model_path}")
