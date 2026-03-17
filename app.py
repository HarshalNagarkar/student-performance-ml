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

st.sidebar.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Make Prediction"])

data = load_data()

if page == "Home":
    st.title("Student Performance Prediction System")
    st.markdown("---")
    st.markdown("### What does this project do?")
    st.markdown("""
    This application uses **Machine Learning** to predict whether a university student 
    will **Pass or Fail** based on three academic indicators.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Attendance**\n\nHow regularly the student attends classes (%)")
    with col2:
        st.info("**Internal Marks**\n\nMarks scored in internal exams (out of 50)")
    with col3:
        st.info("**Assignment Score**\n\nMarks scored in assignments (out of 50)")

    st.markdown("---")
    st.markdown("### How It Works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.success("**Step 1**\n\nEnter student details")
    with col2:
        st.success("**Step 2**\n\nChoose ML model")
    with col3:
        st.success("**Step 3**\n\nModel analyzes data")
    with col4:
        st.success("**Step 4**\n\nGet Pass/Fail result")

    st.markdown("---")
    st.markdown("### Models Used")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.warning("**Logistic Regression**\n\nAccuracy: 80%\n\nSimple baseline model")
    with col2:
        st.warning("**Random Forest**\n\nAccuracy: 93%\n\nBest performing model")
    with col3:
        st.warning("**SVM**\n\nAccuracy: 93%\n\nRobust and reliable")

    st.markdown("---")
    st.markdown("### Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(data))
    col2.metric("Pass Count", len(data[data["result"] == "Pass"]))
    col3.metric("Fail Count", len(data[data["result"] == "Fail"]))

elif page == "Dashboard":
    st.title("Student Data Dashboard")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(data))
    col2.metric("Pass Count", len(data[data["result"] == "Pass"]))
    col3.metric("Fail Count", len(data[data["result"] == "Fail"]))

    st.markdown("---")
    st.markdown("### Average Academic Scores")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Attendance", f"{data['attendance'].mean():.1f}%")
    col2.metric("Avg Internal Marks", f"{data['internal_marks'].mean():.1f}")
    col3.metric("Avg Assignment Score", f"{data['assignment_score'].mean():.1f}")

    st.markdown("---")
    st.markdown("### Sample Student Data")
    st.dataframe(data.head(10), use_container_width=True)

    st.markdown("---")
    st.markdown("### Visual Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pass vs Fail Distribution**")
        fig, ax = plt.subplots(figsize=(4, 3))
        data["result"].value_counts().plot(kind="bar", color=["green", "red"], ax=ax)
        ax.set_xlabel("Result")
        ax.set_ylabel("Count")
        ax.set_title("Pass vs Fail Count")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("**Attendance Distribution**")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.histplot(data["attendance"], bins=20, kde=True, ax=ax2, color="blue")
        ax2.set_title("Attendance Distribution")
        plt.tight_layout()
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Correlation Heatmap**")
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            data[["attendance", "internal_marks", "assignment_score"]].corr(),
            annot=True, cmap="coolwarm", ax=ax3
        )
        plt.tight_layout()
        st.pyplot(fig3)

    with col4:
        st.markdown("**Marks Distribution by Result**")
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        data.boxplot(column="internal_marks", by="result", ax=ax4)
        ax4.set_title("Internal Marks by Result")
        plt.tight_layout()
        st.pyplot(fig4)

    st.markdown("---")
    st.markdown("### Model Comparison")
    comparison = {
        "Model": ["Logistic Regression", "Random Forest", "SVM"],
        "Accuracy": ["80%", "93%", "93%"],
        "Best For": ["Baseline", "Complex patterns", "Small datasets"]
    }
    st.table(pd.DataFrame(comparison))

elif page == "Make Prediction":
    st.title("Student Result Predictor")
    st.markdown("---")
    st.markdown("### Enter Student Details")

    model_choice = st.selectbox("Choose Prediction Model", list(MODELS.keys()))

    col1, col2, col3 = st.columns(3)
    with col1:
        attendance = st.slider("Attendance (%)", 40, 100, 75)
    with col2:
        internal_marks = st.slider("Internal Marks (out of 50)", 10, 50, 25)
    with col3:
        assignment_score = st.slider("Assignment Score (out of 50)", 10, 50, 25)

    st.markdown("---")

    if st.button("Predict Result", use_container_width=True):
        model, encoder, scaler = load_model(model_choice)
        input_data = scaler.transform([[attendance, internal_marks, assignment_score]])
        prediction = model.predict(input_data)
        result = encoder.inverse_transform(prediction)[0]

        st.markdown("---")
        st.markdown("### Prediction Result")

        if result == "Pass":
            st.success("## PASS")
            st.balloons()
            st.markdown("The student is **likely to pass** based on the given indicators.")
        else:
            st.error("## FAIL")
            st.markdown("The student is **at risk of failing**. Areas needing improvement:")
            if attendance < 75:
                st.warning("Attendance is below 75% — needs improvement")
            if internal_marks < 25:
                st.warning("Internal marks are low — needs improvement")
            if assignment_score < 25:
                st.warning("Assignment score is low — needs improvement")

        st.markdown("---")
        st.markdown("### Input Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Attendance", f"{attendance}%")
        col2.metric("Internal Marks", f"{internal_marks}/50")
        col3.metric("Assignment Score", f"{assignment_score}/50")
        st.caption(f"Model used: {model_choice}")
