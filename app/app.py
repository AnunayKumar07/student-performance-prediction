import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('models/student_risk_model.pkl')

# App title
st.title("Student Performance Prediction System")
st.write("Predict if a student is at risk of failing based on attendance, past marks, and activities.")

# Input fields
attendance = st.slider("Attendance Percentage", 0, 100, 50)
past_marks = st.slider("Past Marks (out of 100)", 0, 100, 50)
activities_score = st.slider("Activities Score (0-10)", 0, 10, 5)

# Predict button
if st.button("Predict Risk"):
    input_data = [[attendance, past_marks, activities_score]]
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]  # Probability of being at risk
    risk_status = "At Risk" if prediction == 1 else "Not At Risk"
    st.write(f"Prediction: {risk_status}")
    st.write(f"Probability of Being At Risk: {prob:.2f}")
    
    # Display feature importance image
    st.image('models/feature_importance.png', caption='Feature Importance')

# Footer
st.write("This system helps in early intervention for students.")