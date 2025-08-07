import streamlit as st
import joblib
import numpy as np

# تحميل الموديل والـ Scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

model_features = model.n_features_in_   # عدد الأعمدة اللي الموديل عايزها
scaler_features = len(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else None

st.title("❤️ Heart Failure Prediction App")

with st.form("prediction_form"):
    st.subheader("📝 Enter Patient Information")

    # إدخالات عامة
    age = st.number_input("Age", min_value=0, max_value=120)
    cpk = st.number_input("Creatinine Phosphokinase", min_value=0)
    ef = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100)
    platelets = st.number_input("Platelets", min_value=0)
    serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0)
    serum_sodium = st.number_input("Serum Sodium", min_value=0)
    time = st.number_input("Follow-up Time (days)", min_value=0)

    # لو الموديل عايز 12 Feature → نضيف باقي المدخلات
    if model_features == 12:
        anaemia = st.radio("Anaemia (0=No, 1=Yes)", [0, 1], horizontal=True)
        diabetes = st.radio("Diabetes (0=No, 1=Yes)", [0, 1], horizontal=True)
        high_bp = st.radio("High Blood Pressure (0=No, 1=Yes)", [0, 1], horizontal=True)
        sex = st.radio("Sex (0=Female, 1=Male)", [0, 1], horizontal=True)
        smoking = st.radio("Smoking (0=No, 1=Yes)", [0, 1], horizontal=True)

    submitted = st.form_submit_button("🔍 Predict")

# التنبؤ
if submitted:
    if model_features == 12:
        features = np.array([[age, anaemia, cpk, diabetes, ef, high_bp,
                              platelets, serum_creatinine, serum_sodium,
                              sex, smoking, time]])
        # لو الـ Scaler مختلف → ما نستخدموش
        if scaler_features == 12:
            features = scaler.transform(features)
    else:  # 7 Features
        features = np.array([[age, cpk, ef, platelets, serum_creatinine, serum_sodium, time]])
        if scaler_features == 7:
            features = scaler.transform(features)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Failure ({probability:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Failure ({100 - probability:.2f}%)")

