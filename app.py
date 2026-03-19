import streamlit as st
import sqlite3
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DB_NAME = "student_performance.db"
MODELS = {
    "Random Forest": "models/saved_models/random_forest_v1.pkl",
    "Logistic Regression": "models/saved_models/logistic_regression_v1.pkl",
    "SVM": "models/saved_models/svm_v1.pkl"
}

st.set_page_config(
    page_title="PlaceIQ - Student Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1e293b; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stMetric"] { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px 20px; }
[data-testid="stMetric"]:hover { border-color: #6366f1; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 26px !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }
.stButton > button { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white !important; border: none; border-radius: 8px; padding: 10px 28px; font-weight: 600; font-size: 14px; width: 100%; }
.stButton > button:hover { opacity: 0.88; }
hr { border-color: #1e293b; }
.section-header { font-size: 18px; font-weight: 700; color: #e2e8f0; border-left: 4px solid #6366f1; padding-left: 12px; margin: 8px 0 16px 0; }
.pipeline-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 18px; height: 100%; border-top: 3px solid #6366f1; }
.pipeline-num { font-size: 28px; font-weight: 900; color: #6366f1; opacity: 0.4; line-height: 1; }
.pipeline-title { font-size: 14px; font-weight: 700; color: #e2e8f0; margin: 4px 0; }
.pipeline-desc { font-size: 12px; color: #64748b; }
.model-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; text-align: center; border-top: 3px solid; }
.model-acc { font-size: 36px; font-weight: 800; margin: 8px 0; }
.model-name { font-size: 15px; font-weight: 700; color: #e2e8f0; }
.model-type { font-size: 11px; color: #64748b; margin-top: 4px; }
.predict-result-pass { background: linear-gradient(135deg, #064e3b, #065f46); border: 1px solid #10b981; border-radius: 14px; padding: 28px; text-align: center; }
.predict-result-fail { background: linear-gradient(135deg, #450a0a, #7f1d1d); border: 1px solid #ef4444; border-radius: 14px; padding: 28px; text-align: center; }
.result-label { font-size: 42px; font-weight: 900; letter-spacing: 4px; }
.result-sub { font-size: 14px; color: #cbd5e1; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)


def load_data():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM students", conn)
    conn.close()
    return df

def load_model(model_name):
    saved = joblib.load(MODELS[model_name])
    return saved["model"], saved["encoder"], saved["scaler"]

def set_chart_style(ax, fig):
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("#e2e8f0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.title.set_fontsize(12)
    ax.title.set_fontweight("bold")


data = load_data()
total     = len(data)
passed    = len(data[data["result"] == "Pass"])
failed    = len(data[data["result"] == "Fail"])
pass_rate = (passed / total * 100)


with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 8px 0;'>
        <div style='font-size:20px; font-weight:800; background: linear-gradient(135deg,#6366f1,#a78bfa);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>PlaceIQ</div>
        <div style='font-size:11px; color:#475569; margin-top:2px;'>Student Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='margin:10px 0; border-color:#1e293b'>", unsafe_allow_html=True)
    page = st.radio("", ["Home", "Dashboard", "Predict"], label_visibility="collapsed")
    st.markdown("<hr style='margin:10px 0; border-color:#1e293b'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px'>Live Stats</div>", unsafe_allow_html=True)
    st.metric("Total Students", f"{total:,}")
    st.metric("Pass Rate", f"{pass_rate:.1f}%")
    st.metric("Avg Attendance", f"{data['attendance'].mean():.1f}%")
    st.markdown("<hr style='margin:10px 0; border-color:#1e293b'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px'>Models</div>", unsafe_allow_html=True)
    for m, acc in [("Random Forest", "93%"), ("SVM", "93%"), ("Logistic Reg.", "80%")]:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;font-size:12px;padding:3px 0;'>"
            f"<span style='color:#94a3b8'>{m}</span>"
            f"<span style='color:#6366f1;font-weight:700'>{acc}</span></div>",
            unsafe_allow_html=True
        )


if page == "Home":
    st.markdown("""
    <div style='padding: 8px 0 4px 0'>
        <div style='font-size:30px; font-weight:900; color:#f1f5f9;'>Student Performance
            <span style='background:linear-gradient(135deg,#6366f1,#a78bfa);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Prediction System</span>
        </div>
        <div style='color:#64748b; font-size:14px; margin-top:4px;'>Machine Learning based academic outcome prediction for university students</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", f"{total:,}")
    c2.metric("Passed", f"{passed:,}", delta=f"+{pass_rate:.1f}%")
    c3.metric("Failed", f"{failed:,}", delta=f"-{100-pass_rate:.1f}%", delta_color="inverse")
    c4.metric("Pass Rate", f"{pass_rate:.1f}%")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>About This Project</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='pipeline-card' style='border-top-color:#6366f1'>
            <div style='font-size:15px; font-weight:700; color:#e2e8f0; margin-bottom:10px'>Problem Statement</div>
            <div style='font-size:13px; color:#94a3b8; line-height:1.7'>
            Universities struggle to identify at-risk students early enough to intervene.
            This system uses Machine Learning to predict student outcomes based on academic
            performance indicators, enabling timely and targeted support.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='pipeline-card' style='border-top-color:#8b5cf6'>
            <div style='font-size:15px; font-weight:700; color:#e2e8f0; margin-bottom:10px'>Input Features</div>
            <div style='font-size:13px; color:#94a3b8; line-height:1.9'>
            <b style='color:#e2e8f0'>Attendance</b> - Percentage of classes attended (40-100%)<br>
            <b style='color:#e2e8f0'>Internal Marks</b> - Midterm exam score (10-50)<br>
            <b style='color:#e2e8f0'>Assignment Score</b> - Assignment performance (10-50)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>System Pipeline</div>", unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    for col, num, title, desc, color in [
        (p1, "01", "Data Generation",  "Synthetic student records stored in SQLite database", "#6366f1"),
        (p2, "02", "Preprocessing",    "Data validation, cleaning and feature scaling applied", "#8b5cf6"),
        (p3, "03", "Model Training",   "Three ML models trained with full evaluation metrics", "#a78bfa"),
        (p4, "04", "Prediction",       "Real-time Pass/Fail prediction served via dashboard",  "#c4b5fd"),
    ]:
        col.markdown(f"<div class='pipeline-card' style='border-top-color:{color}'><div class='pipeline-num'>{num}</div><div class='pipeline-title'>{title}</div><div class='pipeline-desc'>{desc}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    for col, name, acc, typ, best, color in [
        (m1, "Random Forest",          "93%", "Ensemble",     "Complex non-linear patterns",  "#10b981"),
        (m2, "Support Vector Machine", "93%", "Kernel-based", "Clear decision boundaries",    "#6366f1"),
        (m3, "Logistic Regression",    "80%", "Linear",       "Baseline and interpretability","#f59e0b"),
    ]:
        col.markdown(f"<div class='model-card' style='border-top-color:{color}'><div class='model-name'>{name}</div><div class='model-acc' style='color:{color}'>{acc}</div><div class='model-type'>{typ} - Best for: {best}</div></div>", unsafe_allow_html=True)


elif page == "Dashboard":
    st.markdown("""
    <div style='font-size:28px; font-weight:900; color:#f1f5f9; padding-bottom:4px'>
        Student Data <span style='background:linear-gradient(135deg,#6366f1,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Dashboard</span>
    </div>
    <div style='color:#64748b; font-size:13px;'>Statistical overview and visual analysis of student performance data</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", f"{total:,}")
    c2.metric("Passed", f"{passed:,}", delta=f"+{pass_rate:.1f}%")
    c3.metric("Failed", f"{failed:,}", delta=f"-{100-pass_rate:.1f}%", delta_color="inverse")
    c4.metric("Pass Rate", f"{pass_rate:.1f}%")
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Attendance",       f"{data['attendance'].mean():.1f}%")
    c2.metric("Avg Internal Marks",   f"{data['internal_marks'].mean():.1f} / 50")
    c3.metric("Avg Assignment Score", f"{data['assignment_score'].mean():.1f} / 50")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Visual Analysis</div>", unsafe_allow_html=True)
    PASS_CLR = "#6366f1"
    FAIL_CLR = "#f43f5e"
    GRID_CLR = "#334155"

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        set_chart_style(ax, fig)
        counts = data["result"].value_counts()
        colors = [PASS_CLR if r == "Pass" else FAIL_CLR for r in counts.index]
        bars = ax.bar(counts.index, counts.values, color=colors, width=0.45, edgecolor="#0f1117", linewidth=1.5)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + total * 0.005,
                    f"{val:,}", ha='center', fontsize=10, fontweight='bold', color="#e2e8f0")
        ax.set_title("Pass vs Fail Distribution")
        ax.set_ylabel("Students", color="#94a3b8")
        ax.yaxis.grid(True, color=GRID_CLR, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        set_chart_style(ax2, fig2)
        ax2.hist(data[data["result"]=="Pass"]["attendance"], bins=20, color=PASS_CLR, alpha=0.7, label="Pass", edgecolor="#0f1117")
        ax2.hist(data[data["result"]=="Fail"]["attendance"], bins=20, color=FAIL_CLR, alpha=0.7, label="Fail", edgecolor="#0f1117")
        ax2.axvline(75, color="#f59e0b", linestyle="--", linewidth=1.5, label="75% threshold")
        ax2.set_title("Attendance Distribution by Result")
        ax2.set_xlabel("Attendance (%)")
        ax2.set_ylabel("Count")
        ax2.yaxis.grid(True, color=GRID_CLR, linestyle='--', alpha=0.5)
        ax2.set_axisbelow(True)
        ax2.legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    col3, col4 = st.columns(2)
    with col3:
        fig3, ax3 = plt.subplots(figsize=(5, 3.2))
        fig3.patch.set_facecolor("#1e293b")
        ax3.set_facecolor("#1e293b")
        corr = data[["attendance", "internal_marks", "assignment_score"]].corr()
        corr.index   = ["Attendance", "Internal Marks", "Assignment"]
        corr.columns = ["Attendance", "Internal Marks", "Assignment"]
        sns.heatmap(corr, annot=True, cmap="Blues", ax=ax3, linewidths=1, fmt='.2f',
                    annot_kws={"size": 10, "color": "#e2e8f0"}, linecolor="#0f1117", cbar_kws={"shrink": 0.8})
        ax3.set_title("Feature Correlation Heatmap", color="#e2e8f0", fontsize=12, fontweight="bold")
        ax3.tick_params(colors="#94a3b8", labelsize=9)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col4:
        fig4, ax4 = plt.subplots(figsize=(5, 3.2))
        set_chart_style(ax4, fig4)
        features   = ["attendance", "internal_marks", "assignment_score"]
        labels     = ["Attendance", "Int. Marks", "Assignment"]
        pass_means = [data[data["result"]=="Pass"][f].mean() for f in features]
        fail_means = [data[data["result"]=="Fail"][f].mean() for f in features]
        x = np.arange(len(labels))
        w = 0.35
        ax4.bar(x - w/2, pass_means, w, color=PASS_CLR, label="Pass", edgecolor="#0f1117")
        ax4.bar(x + w/2, fail_means, w, color=FAIL_CLR, label="Fail", edgecolor="#0f1117")
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.set_title("Avg Feature Values: Pass vs Fail")
        ax4.set_ylabel("Score")
        ax4.yaxis.grid(True, color=GRID_CLR, linestyle='--', alpha=0.5)
        ax4.set_axisbelow(True)
        ax4.legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Model Comparison</div>", unsafe_allow_html=True)
    model_df = pd.DataFrame({
        "Model":    ["Random Forest", "Support Vector Machine", "Logistic Regression"],
        "Accuracy": ["93%", "93%", "80%"],
        "Type":     ["Ensemble", "Kernel-based", "Linear"],
        "Best For": ["Complex patterns", "Clear boundaries", "Baseline and speed"],
        "Status":   ["Recommended", "Strong", "Baseline"]
    })
    st.dataframe(model_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Sample Student Records</div>", unsafe_allow_html=True)
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)


elif page == "Predict":
    st.markdown("""
    <div style='font-size:28px; font-weight:900; color:#f1f5f9; padding-bottom:4px'>
        Student Result <span style='background:linear-gradient(135deg,#6366f1,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Predictor</span>
    </div>
    <div style='color:#64748b; font-size:13px;'>Enter academic details to generate a real-time Pass/Fail prediction</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div class='section-header'>Student Input</div>", unsafe_allow_html=True)
        model_choice     = st.selectbox("Select Prediction Model", list(MODELS.keys()))
        st.markdown("<br>", unsafe_allow_html=True)
        attendance       = st.slider("Attendance (%)",               40, 100, 75)
        internal_marks   = st.slider("Internal Marks (out of 50)",   10, 50,  25)
        assignment_score = st.slider("Assignment Score (out of 50)", 10, 50,  25)

        def gauge(label, val, maxv, threshold):
            pct       = val / maxv * 100
            ok        = val >= threshold
            bar_color = "#10b981" if ok else "#f43f5e"
            status    = "Good" if ok else "Low"
            st.markdown(
                f"<div style='margin:6px 0'>"
                f"<div style='display:flex;justify-content:space-between;font-size:12px;color:#94a3b8;margin-bottom:3px'>"
                f"<span>{label}</span><span style='color:{bar_color};font-weight:700'>{status}</span></div>"
                f"<div style='background:#334155;border-radius:4px;height:6px;'>"
                f"<div style='background:{bar_color};width:{pct}%;border-radius:4px;height:6px;'></div>"
                f"</div></div>",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        gauge("Attendance",       attendance,       100, 75)
        gauge("Internal Marks",   internal_marks,   50,  25)
        gauge("Assignment Score", assignment_score, 50,  25)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Run Prediction")

    with col2:
        st.markdown("<div class='section-header'>Prediction Output</div>", unsafe_allow_html=True)
        if predict_btn:
            model, encoder, scaler = load_model(model_choice)
            input_data = scaler.transform([[attendance, internal_marks, assignment_score]])
            prediction = model.predict(input_data)
            result     = encoder.inverse_transform(prediction)[0]

            if result == "Pass":
                st.markdown("<div class='predict-result-pass'><div class='result-label' style='color:#10b981'>PASS</div><div class='result-sub'>This student is likely to pass based on the provided indicators.</div></div>", unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown("<div class='predict-result-fail'><div class='result-label' style='color:#f43f5e'>FAIL</div><div class='result-sub'>This student is at risk of failing. Intervention recommended.</div></div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div style='font-size:13px;font-weight:700;color:#e2e8f0;margin-bottom:8px'>Areas Requiring Attention</div>", unsafe_allow_html=True)
                if attendance < 75:
                    st.warning(f"Attendance {attendance}% - Below required 75% threshold")
                if internal_marks < 25:
                    st.warning(f"Internal Marks {internal_marks}/50 - Below passing threshold of 25")
                if assignment_score < 25:
                    st.warning(f"Assignment Score {assignment_score}/50 - Below passing threshold of 25")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>Input Summary</div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Attendance", f"{attendance}%",
                      delta="Sufficient" if attendance >= 75 else "Insufficient",
                      delta_color="normal" if attendance >= 75 else "inverse")
            c2.metric("Internal Marks", f"{internal_marks}/50",
                      delta="Sufficient" if internal_marks >= 25 else "Insufficient",
                      delta_color="normal" if internal_marks >= 25 else "inverse")
            c3.metric("Assignment", f"{assignment_score}/50",
                      delta="Sufficient" if assignment_score >= 25 else "Insufficient",
                      delta_color="normal" if assignment_score >= 25 else "inverse")
            st.markdown(f"<div style='font-size:11px;color:#475569;margin-top:10px'>Model used: {model_choice}</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#1e293b;border:1px dashed #334155;border-radius:14px;padding:48px 24px;text-align:center;margin-top:8px'>
                <div style='font-size:15px;font-weight:700;color:#e2e8f0;margin:10px 0 6px'>Ready to Predict</div>
                <div style='font-size:13px;color:#64748b;'>Adjust the sliders on the left and click Run Prediction to generate a result.</div>
            </div>
            """, unsafe_allow_html=True)
