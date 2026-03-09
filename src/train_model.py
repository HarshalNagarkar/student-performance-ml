import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_pipeline

MODEL_DIR = "models/saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_all_models():
    X_train, X_test, y_train, y_test, encoder, scaler = preprocess_pipeline()

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": SVC(kernel="rbf", probability=True, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        save_path = f"{MODEL_DIR}/{name}_v1.pkl"
        joblib.dump({
            "model": model,
            "encoder": encoder,
            "scaler": scaler
        }, save_path)

        print(f"{name} -> Accuracy: {acc:.2f} | Saved at: {save_path}")

if __name__ == "__main__":
    train_all_models()
