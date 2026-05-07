
import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_model():
    with open("flowcast_model.pkl", "rb") as f:  model    = pickle.load(f)
    with open("flowcast_scaler.pkl", "rb") as f: scaler   = pickle.load(f)
    with open("flowcast_features.pkl", "rb") as f: features = pickle.load(f)
    return model, scaler, features

model, scaler, features = load_model()

st.title("FlowCast — Predict My Day")

cycle_phase      = st.selectbox("Cycle Phase", ["Menstrual", "Luteal", "Follicular", "Ovulation"])
sleep_hours      = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
stress_level     = st.slider("Stress Level", 0, 10, 4)
pain_level       = st.slider("Pain Level", 0, 10, 2)
energy_level     = st.slider("Energy Level", 0, 10, 6)
mood_score       = st.slider("Mood Score", 0, 10, 6)
concentration    = st.slider("Concentration", 0, 10, 6)
health_score     = st.slider("Overall Health", 0, 10, 7)
water_intake     = st.slider("Water Intake (liters)", 0.0, 5.0, 2.0)
cycle_length     = st.number_input("Cycle Length (days)", 20, 45, 28)
age              = st.number_input("Age", 15, 55, 25)
bmi              = st.number_input("BMI", 15.0, 45.0, 22.0)

if st.button("Predict my day"):
    input_dict = {col: 0.0 for col in features}

    input_dict["sleep_hours"]           = sleep_hours
    input_dict["stress_score_baseline"] = stress_level
    input_dict["pain_level"]            = pain_level
    input_dict["energy_level"]          = energy_level
    input_dict["mood_score"]            = mood_score
    input_dict["concentration_score"]   = concentration
    input_dict["overall_health_score"]  = health_score
    input_dict["water_intake_liters"]   = water_intake
    input_dict["cycle_length_days"]     = float(cycle_length)
    input_dict["age"]                   = float(age)
    input_dict["bmi"]                   = float(bmi)
    input_dict["diet_quality_Good"]     = 1.0

    # Cycle phase encoding
    if cycle_phase == "Menstrual":  input_dict["cycle_phase_Menstrual"] = 1.0
    if cycle_phase == "Luteal":     input_dict["cycle_phase_Luteal"]    = 1.0
    # Follicular and Ovulation = all phase cols stay 0

    X = pd.DataFrame([input_dict])[features]
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    pred = model.predict(X_scaled)[0]

    if pred == 1:
        st.success(f" High Focus Day — {prob:.0%} confidence")
    else:
        st.warning(f" Rest-Oriented Day — {1-prob:.0%} confidence")
    st.progress(float(prob))