import sqlite3
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

DB_NAME = "student_performance.db"
MODEL_PATH = "models/saved_models/logistic_regression_v1.pkl"

def load_data():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM students", conn)
    conn.close()
    return df

if __name__ == "__main__":
    
    df = load_data()
    X = df[["attendance", "internal_marks", "assignment_score"]]
    y = df["result"]

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    encoder = saved["encoder"]

    y_encoded = encoder.transform(y)

    preds = model.predict(X)

    acc = accuracy_score(y_encoded, preds)
    cm = confusion_matrix(y_encoded, preds)

    print(" Model Evaluation Results")
    print("--------------------------")
    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:")
    print(cm)
