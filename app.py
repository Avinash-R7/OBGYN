import streamlit as st
import sys
import os
import pandas as pd
import joblib
import base64
import requests

explanation = ""


st.set_page_config(
    page_title="Maternal Health – Risk Assessment",
    layout="centered",
    initial_sidebar_state="collapsed"
)


def set_bg(url):
    img = requests.get(url).content
    b64 = base64.b64encode(img).decode()
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image:
            linear-gradient(rgba(245,248,255,0.95), rgba(245,248,255,0.95)),
            url("data:image/png;base64,{b64}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("https://images.unsplash.com/photo-1580281657527-47d2bde5c4c9")


st.markdown("""
<style>
.block-container {
    max-width: 420px;   /* mobile-first feel */
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

body {
    font-family: "Inter", "Segoe UI", sans-serif;
}

/* Header card */
.header-card {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: white;
    padding: 1.8rem;
    border-radius: 18px;
    margin-bottom: 1.5rem;
}

.header-title {
    font-size: 1.5rem;
    font-weight: 700;
}

.header-sub {
    font-size: 0.95rem;
    opacity: 0.9;
    margin-top: 0.3rem;
}

/* Card */
.card {
    background: white;
    border-radius: 16px;
    padding: 1.3rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: white;
    font-weight: 600;
    border-radius: 14px;
    height: 3.2em;
}

/* Risk text */
.low { color: #15803d; font-weight: 700; font-size: 1.2rem; }
.mid { color: #b45309; font-weight: 700; font-size: 1.2rem; }
.high { color: #b91c1c; font-weight: 700; font-size: 1.2rem; }

.explain {
    font-size: 0.95rem;
    line-height: 1.6;
    color: #1f2937;
}
</style>
""", unsafe_allow_html=True)


sys.path.append(os.path.abspath("src"))
from explainability import explain_prediction

model = joblib.load("src/best_model.pkl")
importance_df = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)


st.markdown("""
<div class="header-card">
    <div class="header-title">Maternal Health</div>
    <div class="header-sub">
        Early pregnancy risk assessment & clinical insights
    </div>
</div>
""", unsafe_allow_html=True)



age = st.number_input("Age (years)", 15, 50, 25)
sbp = st.number_input("Systolic BP (mmHg)", 90, 180, 120)
dbp = st.number_input("Diastolic BP (mmHg)", 60, 120, 80)
bs = st.number_input("Blood Sugar", 4.0, 15.0, 7.0)
temp = st.number_input("Body Temperature (°F)", 95.0, 102.0, 98.6)
hr = st.number_input("Heart Rate (bpm)", 60, 120, 75)

st.markdown('</div>', unsafe_allow_html=True)

if st.button("Run Risk Assessment", use_container_width=True):

    patient = pd.DataFrame(
        [[age, sbp, dbp, bs, temp, hr]],
        columns=[
            "Age",
            "SystolicBP",
            "DiastolicBP",
            "BS",
            "BodyTemp",
            "HeartRate"
        ]
    )

    probs = model.predict_proba(patient)[0]
    risk = ["LOW", "MID", "HIGH"][probs.argmax()]

    explanation = explain_prediction(model, patient.iloc[0], importance_df)


    st.markdown("""
    <div class="card">
        <strong>Risk Assessment Result</strong>
    """, unsafe_allow_html=True)

    if risk == "LOW":
        st.markdown('<div class="low">Low Risk</div>', unsafe_allow_html=True)
    elif risk == "MID":
        st.markdown('<div class="mid">Moderate Risk</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="high">High Risk</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

if explanation:
    st.markdown(f"""
    <div class="card">
        <strong>Clinical Interpretation</strong>
        <div class="explain">
            {explanation}
        </div>
    </div>
    """, unsafe_allow_html=True)
