![Python](https://img.shields.io/badge/Python-3.8+-blue)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Database](https://img.shields.io/badge/Database-SQLite-lightgrey)

# Student Performance Prediction System

A machine learning project that predicts whether a university student will **Pass or Fail** based on attendance, internal marks, and assignment scores.

---

## Project Structure
```
student-performance-ml/
│
├── data/                  # Generated CSV data (not tracked by git)
├── models/
│   └── saved_models/      # Trained model files (not tracked by git)
├── notebooks/             # EDA and experiments
├── src/
│   ├── generate_data.py   # Generates synthetic student data
│   ├── db_utils.py        # SQLite database utilities
│   ├── preprocess.py      # Data preprocessing pipeline
│   ├── train_model.py     # Trains Logistic Regression, Random Forest, SVM
│   └── evaluate_model.py  # Evaluates models with full metrics
├── app.py                 # Streamlit dashboard
├── requirements.txt       # Dependencies
└── README.md
```

---

## How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Generate data
```
python src/generate_data.py
```

### 3. Load data into database
```
python src/db_utils.py
```

### 4. Train models
```
python src/train_model.py
```

### 5. Evaluate models
```
python src/evaluate_model.py
```

### 6. Run the dashboard
```
streamlit run app.py
```

---

## Models Used
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

## Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score, Confusion Matrix

---

## Maintenance & Retraining Schedule
- Model should be retrained **every month** with new student data
- Retrain if accuracy drops below **85%**
- New data should be added to `data/` and pipeline re-run from Step 2

---

## Author
Harshal Nagarkar
