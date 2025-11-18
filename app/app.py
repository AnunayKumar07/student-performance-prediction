import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Load model
model = joblib.load('models/student_risk_model.pkl')

# Session state for storing predictions (dynamic dashboard)
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

st.title("🚀 Advanced Student Performance Prediction System")
st.markdown("**Predict risk of failing with dynamic inputs and early intervention insights.**")

# Sidebar for navigation
page = st.sidebar.selectbox("Navigate", ["Single Prediction", "Batch Prediction", "Risk Dashboard"])

if page == "Single Prediction":
    st.header("Single Student Prediction")
    st.markdown("Enter details below. Dynamic warnings appear for high-risk factors.")

    col1, col2 = st.columns(2)
    with col1:
        attendance = st.slider("Attendance %", 0, 100, 50, help="Percentage of classes attended.")
        past_marks = st.slider("Past Marks (out of 100)", 0, 100, 50)
        activities_score = st.slider("Activities Score (0-10)", 0, 10, 5)
        total_marks = st.number_input("Total Marks (out of 1000)", 0, 1000, 500, help="Cumulative marks.")
    with col2:
        total_absences = st.number_input("Total Absences (days)", 0, 50, 10)
        supplementary_exams = st.number_input("Supplementary Exams (fail subjects)", 0, 10, 0)
        internet_access = st.selectbox("Internet Access (1=Poor, 5=Excellent)", [1,2,3,4,5], index=2)
        family_support = st.selectbox("Family Support (1=Low, 5=High)", [1,2,3,4,5], index=2)

    # Dynamic warnings
    if total_absences > 20:
        st.warning("⚠️ High absences detected! Risk increases.")
    if supplementary_exams > 2:
        st.error("🚨 Multiple failures! Immediate intervention recommended.")

    if st.button("Predict Risk"):
        input_data = [[attendance, past_marks, activities_score, total_marks, total_absences, supplementary_exams, internet_access, family_support]]
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        risk_status = "At Risk" if prediction == 1 else "Not At Risk"
        st.success(f"Prediction: {risk_status} (Probability: {prob:.2f})")
        
        # Store for dashboard
        st.session_state.predictions.append({
            'Student': f'Student {len(st.session_state.predictions)+1}',
            'Risk_Prob': prob,
            'Status': risk_status
        })

        # Intervention suggestions
        if prediction == 1:
            st.info("💡 Suggestions: Provide counseling, extra classes, or monitor attendance.")

elif page == "Batch Prediction":
    st.header("Batch Prediction (Upload CSV)")
    st.markdown("Upload a CSV with columns: Attendance, Past_Marks, Activities_Score, Total_Marks, Total_Absences, Supplementary_Exams, Internet_Access, Family_Support.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        predictions = model.predict(df_batch)
        probs = model.predict_proba(df_batch)[:, 1]
        df_batch['Predicted_Risk'] = ['At Risk' if p == 1 else 'Not At Risk' for p in predictions]
        df_batch['Risk_Probability'] = probs
        st.dataframe(df_batch)
        st.download_button("Download Results", df_batch.to_csv(index=False), "results.csv")

elif page == "Risk Dashboard":
    st.header("Risk Dashboard")
    if st.session_state.predictions:
        df_dash = pd.DataFrame(st.session_state.predictions)
        fig = px.bar(df_dash, x='Student', y='Risk_Prob', color='Status', title="Risk Probabilities")
        st.plotly_chart(fig)
        st.metric("Average Risk Probability", f"{df_dash['Risk_Prob'].mean():.2f}")
    else:
        st.info("No predictions yet. Make some in Single Prediction!")

# Footer
st.image('models/feature_importance.png', caption='Feature Importance')
st.markdown("---\n*Built for early intervention in education.*")