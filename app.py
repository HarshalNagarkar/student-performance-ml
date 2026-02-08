import streamlit as st
import sqlite3
import pandas as pd
import joblib

DB_NAME = "student_performance.db"
MODEL_PATH = "models/saved_models/logistic_regression_v1.pkl"

saved = joblib.load(MODEL_PATH)
model = saved["model"]
encoder = saved["encoder"]

def load_data():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM students", conn)
    conn.close()
    return df

st.set_page_config(page_title="Student Performance Prediction", layout="centered")

st.title(" Student Performance Prediction System")

st.markdown("""
This dashboard predicts whether a student is likely to **Pass or Fail**
based on academic indicators.
""")

st.header(" Student Data Overview")
data = load_data()
st.dataframe(data.head())

st.header(" Model Information")
st.write("**Model Used:** Logistic Regression")
st.write("**Accuracy:** 0.93 (evaluated on synthetic data)")

st.header(" Make a Prediction")

attendance = st.slider("Attendance (%)", 40, 100, 75)
internal_marks = st.slider("Internal Marks", 10, 50, 25)
assignment_score = st.slider("Assignment Score", 10, 50, 25)

if st.button("Predict Result"):
    input_data = [[attendance, internal_marks, assignment_score]]
    prediction = model.predict(input_data)
    result = encoder.inverse_transform(prediction)[0]

    if result == "Pass":
        st.success(" Prediction: PASS")
    else:
        st.error(" Prediction: FAIL")
