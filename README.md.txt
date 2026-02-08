# Student Performance Prediction System

## Project Overview
This project predicts whether a university student will Pass or Fail based on:
- Attendance percentage
- Internal marks
- Assignment score

The focus of this project is on **machine learning deployment, model lifecycle, and engineering practices**, not just model accuracy.

---

## Technologies Used
- Python
- SQLite (SQL Database)
- Scikit-learn (Logistic Regression)
- Streamlit (Dashboard)

---

## Folder Structure
student-performance-ml/
│
├── data/
│ └── generate_data.py
│
├── database/
│ └── init_db.py
│
├── models/
│ ├── train_model.py
│ └── evaluate_model.py
│
├── app/
│ └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore

---

## How to Run the Project

### 1. Generate Synthetic Data
python data/generate_data.py

### 2. Store Data in SQL Database
python database/init_db.py


### 3. Train the Machine Learning Model
python models/train_model.py


### 4. Evaluate the Model
python models/evaluate_model.py


### 5. Run the Dashboard
streamlit run app/app.py

---


---

## Machine Learning Model
- Model Used: Logistic Regression
- Model Type: Traditional supervised machine learning
- Accuracy: Approximately 0.93 on synthetic data
- Purpose: Performance awareness, not optimization

---

## Model Lifecycle Management
1. New student data is generated periodically using a data generation script.
2. Generated data is stored in a SQL database.
3. The machine learning model is retrained on updated data.
4. Each trained model is saved with a version number.
5. The dashboard always loads the latest trained model for prediction.

---

## Dashboard Features
- Data overview from SQL database
- Model performance information
- Real-time prediction interface

---

## Notes
- No raw datasets are uploaded to the repository.
- No trained model files are uploaded.
- All data and models are generated through scripts to ensure reproducibility.
