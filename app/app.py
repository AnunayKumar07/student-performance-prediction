import streamlit as st
import joblib

# Load model once
model = joblib.load('models/student_risk_model.pkl')

# Custom CSS for background, header, form styling
page_css = """
<style>
    body, .main, .block-container {
        background: linear-gradient(135deg, #ffffff 0%, #d6e6ff 100%);
        color: #222;
        max-width: 600px;
        margin: auto;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        background: linear-gradient(90deg, #5B68E8, #7E88FF);
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        padding: 30px 20px 25px 20px;
        border-radius: 12px 12px 0 0;
        text-align: center;
        user-select: none;
        margin-bottom: 15px;
    }
    .subheader {
        text-align: center;
        font-weight: 500;
        color: #4a4a4a;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }
    label {
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        margin-top: 15px !important;
        margin-bottom: 6px !important;
    }
    input, select {
        font-size: 1.1rem !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
        border: 1.8px solid #a1a9f3 !important;
        outline-offset: 0 !important;
        width: 100% !important;
        box-sizing: border-box;
        transition: border-color 0.3s ease;
        margin-bottom: 12px;
    }
    input:focus, select:focus {
        border-color: #5B68E8 !important;
        box-shadow: 0 0 6px #7e88ff75 !important;
    }
    button[kind="primary"] {
        width: 100% !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        padding: 14px !important;
        margin-top: 25px !important;
        border-radius: 12px !important;
        background: linear-gradient(90deg, #5B68E8 0%, #7E88FF 100%) !important;
        color: white !important;
        cursor: pointer;
        transition: background 0.4s ease;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(90deg, #7E88FF 0%, #5B68E8 100%) !important;
    }
    .info-text {
        font-size: 0.9rem;
        color: #555;
        margin-top: 20px;
        margin-bottom: 30px;
        line-height: 1.4;
        border-top: 1px solid #ececec;
        padding-top: 15px;
    }
    .result-section {
        background: white;
        padding: 40px 35px;
        border-radius: 14px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 3rem;
    }
    .result-title {
        font-size: 5rem;
        margin-bottom: 15px;
        font-weight: 900;
        user-select: none;
    }
    .result-pass {
        color: #28a745;
    }
    .result-fail {
        color: #dc3545;
    }
    .result-confidence {
        font-size: 1.3rem;
        margin-bottom: 30px;
        color: #555;
    }
    .steps-header {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 18px;
        color: #3a3a3a;
    }
    .step-text {
        font-size: 1.2rem;
        margin-bottom: 15px;
        color: #404040;
        line-height: 1.4;
    }
    .try-again-btn {
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        width: 200px !important;
        margin-top: 30px !important;
        border-radius: 10px !important;
    }
</style>
"""

st.markdown(page_css, unsafe_allow_html=True)

st.markdown('<div class="header">🎓 Student Performance Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">AI-Powered Early Intervention for Academic Success</div>', unsafe_allow_html=True)

if "show_result" not in st.session_state:
    st.session_state.show_result = False

if not st.session_state.show_result:
    # Input fields arranged serially
    total_marks = st.number_input("Total Marks (0 to 500) *", 0, 500, step=1,
                                  help="Total marks obtained by the student. Max: 500.")
    total_absences = st.number_input("Total Absences (Max 30) *", 0, 30, step=1,
                                     help="Number of absent days. Max: 30.")
    supplementary_exams = st.number_input("Number of Supplementary Exams (Max 5) *", 0, 5, step=1,
                                         help="Number of fail subjects/supplementary exams.")

    family_support = st.selectbox("Family Educational Support *", ["Yes", "No"],
                                 help="Does the family provide support?")
    extra_curricular = st.selectbox("Extra-Curricular Activities Involvement *", ["Yes", "No"],
                                   help="Is involved in extra-curricular activities?")
    wants_higher_education = st.selectbox("Wants Higher Education *", ["Yes", "No"],
                                          help="Does the student want to pursue higher education?")
    internet_access = st.selectbox("Internet Access Quality *", ["Yes", "No"],
                                  help="Is internet access available and reliable?")

    # Boundary conditions and possible outputs below button
    st.markdown("""
    <div class="info-text">
    <strong>Boundary Conditions:</strong><br>
    - Total Marks: 0 to 500<br>
    - Total Absences: Maximum 30<br>
    - Supplementary Exams: Maximum 5<br><br>
    <strong>Possible Outputs:</strong><br>
    - <span style='color:#28a745; font-weight:bold;'>PASS</span> — Student is likely to succeed.<br>
    - <span style='color:#dc3545; font-weight:bold;'>FAIL</span> — Student is at risk; intervention recommended.
    </div>
    """, unsafe_allow_html=True)

    if st.button("🧙🏻 Predict Performance"):
        # Encoding categorical variables
        fam_sup_val = 1 if family_support == "Yes" else 0
        extra_curr_val = 1 if extra_curricular == "Yes" else 0
        higher_ed_val = 1 if wants_higher_education == "Yes" else 0
        internet_val = 1 if internet_access == "Yes" else 0

        features = [[
            total_marks,
            total_absences,
            supplementary_exams,
            fam_sup_val,
            extra_curr_val,
            higher_ed_val,
            internet_val
        ]]

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        st.session_state.show_result = True
        st.session_state.prediction = pred
        st.session_state.probability = prob

else:
    # Result page
    st.markdown('<div class="result-section">', unsafe_allow_html=True)

    if st.session_state.prediction == 1:
        st.markdown('<div class="result-title result-fail">FAIL</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-confidence">Risk Probability: {st.session_state.probability:.2%}</div>', unsafe_allow_html=True)
        st.markdown('<div class="steps-header">Recommended Next Steps:</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-text">1. Arrange counseling and mentoring.</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-text">2. Monitor attendance and reduce absences.</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-text">3. Engage family for support and improve internet access.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-title result-pass">PASS</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-confidence">Confidence: {(1 - st.session_state.probability):.2%}</div>', unsafe_allow_html=True)
        st.markdown('<div class="steps-header">Suggested Actions:</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-text">1. Maintain good attendance and study habits.</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-text">2. Continue family support and extracurricular activities.</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-text">3. Plan for higher education and skill development.</div>', unsafe_allow_html=True)

    if st.button("🔄 Predict Another Student", key='retry'):
        st.session_state.show_result = False
        st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)