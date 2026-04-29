# ============================================================
# HEART DISEASE PREDICTION - STREAMLIT WEB APP
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import pickle
import os

# ---- Page Config ----
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# ---- Load Model ----
@st.cache_resource
def load_model():
    with open("model/heart_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ---- Custom CSS ----
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #e74c3c;
        color: white;
        font-size: 18px;
        padding: 10px 40px;
        border-radius: 10px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #c0392b; }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
    }
    .danger { background-color: #fde8e8; border: 2px solid #e74c3c; color: #c0392b; }
    .safe   { background-color: #e8fde8; border: 2px solid #2ecc71; color: #1e8449; }
    </style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("<h1 style='text-align:center; color:#e74c3c;'>❤️ Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>SE Mechanical Engineering</p>", unsafe_allow_html=True)
st.markdown("---")

# ---- Sidebar Info ----
with st.sidebar:
    st.header("📋 About This Project")
    st.info("""
    **Model:** Random Forest Classifier  
    **Dataset:** Cleveland Heart Disease (UCI)  
    **Features:** 13 clinical parameters  
    **Accuracy:** ~88–92%
    """)
    st.header("📊 Feature Guide")
    st.markdown("""
    - **cp**: Chest Pain Type (0–3)
    - **thal**: Thalassemia (1=Normal, 2=Fixed, 3=Reversible)
    - **ca**: Major vessels (0–3)
    - **slope**: ST slope (0–2)
    - **thalach**: Max Heart Rate
    - **oldpeak**: ST depression
    """)

# ---- Input Form ----
st.subheader("🩺 Enter Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Demographics**")
    age = st.slider("Age (years)", 20, 80, 50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)

with col2:
    st.markdown("**🫀 Cardiac Symptoms**")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                         format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)

with col3:
    st.markdown("**🔬 Clinical Tests**")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                       format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                           format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
    slope = st.selectbox("Slope of Peak ST Segment", options=[0, 1, 2],
                         format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[1, 2, 3],
                        format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x])

st.markdown("---")

# ---- Predict Button ----
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button("🔍 Predict Heart Disease Risk")

if predict_btn:
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                             restecg, thalach, exang, oldpeak, slope, ca, thal]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-box danger">
                ⚠️ HIGH RISK of Heart Disease<br>
                <span style='font-size:16px'>Confidence: {probability[1]*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.warning("Please consult a cardiologist immediately.")
        else:
            st.markdown(f"""
            <div class="result-box safe">
                ✅ LOW RISK of Heart Disease<br>
                <span style='font-size:16px'>Confidence: {probability[0]*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.success("Keep up your healthy lifestyle!")

    with col_r2:
        st.markdown("**📈 Probability Breakdown**")
        st.metric("No Heart Disease", f"{probability[0]*100:.1f}%")
        st.metric("Heart Disease", f"{probability[1]*100:.1f}%")
        st.progress(int(probability[1] * 100))

    # ---- Input Summary ----
    st.markdown("---")
    st.subheader("📋 Patient Data Summary")
    summary = {
        "Age": age, "Sex": "Male" if sex == 1 else "Female",
        "Chest Pain Type": ["Typical", "Atypical", "Non-anginal", "Asymptomatic"][cp],
        "Resting BP": f"{trestbps} mmHg", "Cholesterol": f"{chol} mg/dl",
        "Fasting Blood Sugar > 120": "Yes" if fbs == 1 else "No",
        "Resting ECG": ["Normal", "ST-T Abnormal", "LV Hypertrophy"][restecg],
        "Max Heart Rate": thalach,
        "Exercise Angina": "Yes" if exang == 1 else "No",
        "ST Depression": oldpeak,
        "ST Slope": ["Upsloping", "Flat", "Downsloping"][slope],
        "Major Vessels": ca,
        "Thalassemia": {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[thal]
    }
    import pandas as pd
    summary_df = pd.DataFrame(summary.items(), columns=["Parameter", "Value"])
    st.table(summary_df)

# ---- Footer ----
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:gray; font-size:13px;'>
⚠️ Disclaimer: This tool is for educational purposes only. Not a substitute for medical advice.<br>
</p>
""", unsafe_allow_html=True)
