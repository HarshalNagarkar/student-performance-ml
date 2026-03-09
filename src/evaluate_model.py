import joblib
import os
import sys
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_pipeline

MODEL_DIR = "models/saved_models"

def evaluate_model(model_name):
    model_path = f"{MODEL_DIR}/{model_name}_v1.pkl"
    saved = joblib.load(model_path)
    model = saved["model"]

    _, X_test, _, y_test, _, _ = preprocess_pipeline()

    preds = model.predict(X_test)

    print(f"\nEvaluation: {model_name}")
    print("-" * 30)
    print(f"Accuracy  : {accuracy_score(y_test, preds):.2f}")
    print(f"Precision : {precision_score(y_test, preds):.2f}")
    print(f"Recall    : {recall_score(y_test, preds):.2f}")
    print(f"F1 Score  : {f1_score(y_test, preds):.2f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, preds)}")

if __name__ == "__main__":
    for model_name in ["logistic_regression", "random_forest", "svm"]:
        evaluate_model(model_name)
