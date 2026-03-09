import streamlit as st
import sqlite3
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

DB_NAME = "student_performance.db"

MODELS = {
    "Logistic Regression": "models/saved_models/logistic_regression_v1.pkl",
    "Random Forest": "models/saved_models/random_forest_v1.pkl",
    "SVM": "models/saved_models/svm_v1.pkl"
}

def load_data():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM students", conn)
    conn.close()
    return df

def load_model(model_name):
    saved = joblib.load(MODELS[model_name])
    return saved["model"], saved["encoder"], saved["scaler"]

st.set_page_config(page_title="Student Performance Prediction", layout="wide")

st.title("🎓 Student Performance Prediction System")
st.markdown("Predict whether a student will **Pass or Fail** based on academic indicators.")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Make Prediction"])

data = load_data()

if page == "Dashboard":
    st.header("📊 Student Data Overview")
    st.dataframe(data.head(10))

    st.subheader("Pass/Fail Distribution")
    fig, ax = plt.subplots()
    data["result"].value_counts().plot(kind="bar", color=["green", "red"], ax=ax)
    ax.set_xlabel("Result")
    ax.set_ylabel("Count")
    ax.set_title("Pass vs Fail Count")
    st.pyplot(fig)

    st.subheader("Attendance Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(data["attendance"], bins=20, kde=True, ax=ax2, color="blue")
    ax2.set_title("Attendance Distribution")
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots()
    sns.heatmap(
        data[["attendance", "internal_marks", "assignment_score"]].corr(),
        annot=True, cmap="coolwarm", ax=ax3
    )
    st.pyplot(fig3)

elif page == "Make Prediction":
    st.header("🔍 Make a Prediction")

    model_choice = st.selectbox("Choose Model", list(MODELS.keys()))

    col1, col2, col3 = st.columns(3)
    with col1:
        attendance = st.slider("Attendance (%)", 40, 100, 75)
    with col2:
        internal_marks = st.slider("Internal Marks", 10, 50, 25)
    with col3:
        assignment_score = st.slider("Assignment Score", 10, 50, 25)

    if st.button("Predict Result"):
        model, encoder, scaler = load_model(model_choice)
        input_data = scaler.transform([[attendance, internal_marks, assignment_score]])
        prediction = model.predict(input_data)
        result = encoder.inverse_transform(prediction)[0]

        st.markdown("---")
        if result == "Pass":
            st.success(f"Prediction: PASS")
            st.balloons()
        else:
            st.error(f"Prediction: FAIL")

        st.markdown("### Input Summary")
        st.write(f"- Attendance: **{attendance}%**")
        st.write(f"- Internal Marks: **{internal_marks}**")
        st.write(f"- Assignment Score: **{assignment_score}**")
        st.write(f"- Model Used: **{model_choice}**")
