import streamlit as st
import sqlite3
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

DB_NAME = "student_performance.db"
MODELS = {
    "Random Forest": "models/saved_models/random_forest_v1.pkl",
    "Logistic Regression": "models/saved_models/logistic_regression_v1.pkl",
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

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="assets/icon.png" if False else None,
    layout="wide"
)

data = load_data()
total = len(data)
passed = len(data[data["result"] == "Pass"])
failed = len(data[data["result"] == "Fail"])
pass_rate = (passed / total * 100)

with st.sidebar:
    st.markdown("## Student Performance AI")
    st.caption("Academic Outcome Prediction System")
    st.markdown("---")
    page = st.radio("Navigation", ["Home", "Dashboard", "Predict"])
    st.markdown("---")
    st.metric("Total Students", total)
    st.metric("Pass Rate", f"{pass_rate:.1f}%")
    st.markdown("---")
    st.markdown("**Available Models**")
    st.markdown("- Random Forest")
    st.markdown("- Logistic Regression")
    st.markdown("- Support Vector Machine")

if page == "Home":
    st.title("Student Performance Prediction System")
    st.caption("Machine Learning based academic outcome prediction for university students")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", total)
    col2.metric("Passed", passed)
    col3.metric("Failed", failed)
    col4.metric("Pass Rate", f"{pass_rate:.1f}%")

    st.markdown("---")
    st.subheader("About This Project")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Problem Statement**
        
        Universities struggle to identify at-risk students 
        early enough to intervene. This system uses Machine 
        Learning to predict student outcomes based on 
        academic performance indicators.
        """)
    with col2:
        st.info("""
        **Input Features**
        
        - **Attendance** — Percentage of classes attended (40–100%)
        - **Internal Marks** — Midterm exam score (10–50)
        - **Assignment Score** — Assignment performance (10–50)
        """)

    st.markdown("---")
    st.subheader("System Pipeline")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.success("**01 — Data Generation**\n\nSynthetic student records generated and stored in SQLite database")
    with col2:
        st.success("**02 — Preprocessing**\n\nData validation, cleaning, and feature scaling applied")
    with col3:
        st.success("**03 — Model Training**\n\nThree ML models trained with full evaluation metrics")
    with col4:
        st.success("**04 — Prediction**\n\nReal-time Pass/Fail prediction served via dashboard")

    st.markdown("---")
    st.subheader("Model Performance Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.warning("""
        **Random Forest**
        
        Accuracy: **93%**
        
        Ensemble model. Best performer for 
        complex non-linear patterns in data.
        """)
    with col2:
        st.warning("""
        **Support Vector Machine**
        
        Accuracy: **93%**
        
        Robust classifier. Works well with 
        clear decision boundaries.
        """)
    with col3:
        st.warning("""
        **Logistic Regression**
        
        Accuracy: **80%**
        
        Baseline model. Simple, interpretable,
        and computationally efficient.
        """)

elif page == "Dashboard":
    st.title("Student Data Dashboard")
    st.caption("Statistical overview and visual analysis of student performance data")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", total)
    col2.metric("Passed", passed)
    col3.metric("Failed", failed)
    col4.metric("Pass Rate", f"{pass_rate:.1f}%")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Attendance", f"{data['attendance'].mean():.1f}%")
    col2.metric("Avg Internal Marks", f"{data['internal_marks'].mean():.1f} / 50")
    col3.metric("Avg Assignment Score", f"{data['assignment_score'].mean():.1f} / 50")

    st.markdown("---")
    st.subheader("Visual Analysis")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(4, 2.8))
        counts = data["result"].value_counts()
        bars = ax.bar(counts.index, counts.values,
                     color=["#16a34a", "#dc2626"], width=0.4)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2.,
                   bar.get_height() + 1, str(val),
                   ha='center', fontsize=9, fontweight='bold')
        ax.set_title("Pass vs Fail Distribution", fontsize=11, fontweight='bold')
        ax.set_ylabel("Count", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(4, 2.8))
        sns.histplot(data["attendance"], bins=20, kde=True,
                    ax=ax2, color="#4f46e5", alpha=0.7)
        ax2.set_title("Attendance Distribution", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Attendance (%)", fontsize=9)
        ax2.set_ylabel("Count", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        fig3, ax3 = plt.subplots(figsize=(4, 2.8))
        corr = data[["attendance", "internal_marks", "assignment_score"]].corr()
        sns.heatmap(corr, annot=True, cmap="Blues", ax=ax3,
                   linewidths=0.5, fmt='.2f',
                   annot_kws={"size": 9})
        ax3.set_title("Feature Correlation Heatmap", fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(4, 2.8))
        pass_data = data[data["result"] == "Pass"]["internal_marks"]
        fail_data = data[data["result"] == "Fail"]["internal_marks"]
        bp = ax4.boxplot([pass_data, fail_data],
                        patch_artist=True,
                        labels=["Pass", "Fail"],
                        widths=0.4)
        bp['boxes'][0].set_facecolor('#16a34a')
        bp['boxes'][1].set_facecolor('#dc2626')
        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='gray')
        ax4.set_title("Internal Marks by Result", fontsize=11, fontweight='bold')
        ax4.set_ylabel("Internal Marks", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig4)

    st.markdown("---")
    st.subheader("Model Comparison")
    model_df = pd.DataFrame({
        "Model": ["Random Forest", "Support Vector Machine", "Logistic Regression"],
        "Accuracy": ["93%", "93%", "80%"],
        "Type": ["Ensemble", "Kernel-based", "Linear"],
        "Best For": ["Complex patterns", "Clear boundaries", "Baseline"]
    })
    st.dataframe(model_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Sample Student Records")
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)

elif page == "Predict":
    st.title("Student Result Predictor")
    st.caption("Enter student academic details to generate a Pass/Fail prediction")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Student Input")
        model_choice = st.selectbox("Select Model", list(MODELS.keys()))
        st.markdown("---")
        attendance = st.slider("Attendance (%)", 40, 100, 75)
        internal_marks = st.slider("Internal Marks (out of 50)", 10, 50, 25)
        assignment_score = st.slider("Assignment Score (out of 50)", 10, 50, 25)
        st.markdown("---")
        predict_btn = st.button("Run Prediction")

    with col2:
        st.subheader("Prediction Output")
        if predict_btn:
            model, encoder, scaler = load_model(model_choice)
            input_data = scaler.transform(
                [[attendance, internal_marks, assignment_score]])
            prediction = model.predict(input_data)
            result = encoder.inverse_transform(prediction)[0]

            if result == "Pass":
                st.success("### PASS")
                st.markdown("The student is **likely to pass** based on the provided academic indicators.")
                st.balloons()
            else:
                st.error("### FAIL")
                st.markdown("The student is **at risk of failing**. The following areas require attention:")
                if attendance < 75:
                    st.warning("Attendance is below the required 75% threshold")
                if internal_marks < 25:
                    st.warning("Internal marks are below the passing threshold of 25")
                if assignment_score < 25:
                    st.warning("Assignment score is below the passing threshold of 25")

            st.markdown("---")
            st.subheader("Input Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Attendance", f"{attendance}%",
                     delta="Sufficient" if attendance >= 75 else "Insufficient",
                     delta_color="normal" if attendance >= 75 else "inverse")
            c2.metric("Internal Marks", f"{internal_marks}/50",
                     delta="Sufficient" if internal_marks >= 25 else "Insufficient",
                     delta_color="normal" if internal_marks >= 25 else "inverse")
            c3.metric("Assignment Score", f"{assignment_score}/50",
                     delta="Sufficient" if assignment_score >= 25 else "Insufficient",
                     delta_color="normal" if assignment_score >= 25 else "inverse")
            st.caption(f"Prediction generated using: {model_choice}")
        else:
            st.info("Adjust the input parameters on the left and click Run Prediction to generate a result.")
