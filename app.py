import streamlit as st
import pandas as pd

import numpy as np

import joblib
from io import BytesIO
from reportlab.pdfgen import canvas

# ================= PAGE SETTINGS =================
st.set_page_config(
    page_title="AI Powered Healthcare Analysis",
    page_icon="🩺",
    layout="wide"
)

# ================= BLACK PROFESSIONAL UI =================
st.markdown("""
<style>

/* Full black background */
.stApp {
    background-color: #0a0a0a;
    color: white;
}

/* Main container */
.main > div {
    background: rgba(20, 20, 20, 0.92);
    border: 1px solid #1f2937;
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
}

/* Headings */
h1, h2, h3 {
    color: #22c55e !important;
    font-weight: 800 !important;
}

/* Labels */
label, p, .stMarkdown {
    color: #e5e7eb !important;
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #111827 !important;
    color: white !important;
    border: 1px solid #374151 !important;
    border-radius: 12px !important;
}

/* Slider */
.stSlider > div {
    color: #22c55e !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #16a34a, #22c55e);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    font-size: 18px;
    font-weight: 700;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(34,197,94,0.35);
}

/* Success and error messages */
.stAlert {
    border-radius: 14px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-thumb {
    background: #22c55e;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='title'> AI Powered Healthcare Analysis</div>
<div class='subtitle'>
    Intelligent Lung Cancer Risk Prediction using Machine Learning
</div>
""", unsafe_allow_html=True)



import os

print("Current folder:", os.getcwd())
print("Files in folder:", os.listdir())

model = joblib.load("lung_cancer_model.pkl")


model = joblib.load("lung_cancer_model.pkl")
le = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.markdown("<h1 class='title'> AI Powered Healthcare Diagnosis</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Lung Cancer Prediction System</p>", unsafe_allow_html=True)

st.write("---")

col1, col2 = st.columns(2)

# Patient Information

with col1:

    patient_id = st.number_input("Patient ID", min_value=1, value=1001)

    age = st.slider("Age", 18, 100, 40)

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    smoking_status = st.selectbox(
        "Smoking Status",
        ["Never", "Former", "Current"]
    )

    years_smoking = st.slider(
        "Years Smoking",
        0, 60, 5
    )

    cigarettes_per_day = st.slider(
        "Cigarettes Per Day",
        0, 60, 10
    )

    secondhand_smoke = st.selectbox(
        "Secondhand Smoke Exposure",
        ["No", "Yes"]
    )

    occupation_exposure = st.selectbox(
        "Occupation Exposure",
        ["No", "Yes"]
    )

    air_pollution = st.slider(
        "Air Pollution Level",
        1, 10, 5
    )

    family_history = st.selectbox(
        "Family History",
        ["No", "Yes"]
    )

    genetic = st.selectbox(
        "Genetic Markers Positive",
        ["No", "Yes"]
    )

with col2:

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=50.0,
        value=23.5
    )

    physical = st.slider(
        "Physical Activity",
        1, 10, 5
    )

    alcohol = st.selectbox(
        "Alcohol Consumption",
        ["No", "Yes"]
    )

    diet = st.selectbox(
        "Diet Quality",
        ["Poor", "Average", "Good"]
    )

    region = st.selectbox(
        "Region",
        ["Urban", "Semi Urban", "Rural"]
    )



    healthcare = st.selectbox(
        "Access to Healthcare",
        ["Poor", "Average", "Good"]
    )

    screening = st.selectbox(
        "Screening Frequency",
        ["Never", "Rarely", "Yearly"]
    )

    chronic = st.selectbox(
        "Chronic Lung Disease",
        ["No", "Yes"]
    )

    diagnosis_year = st.number_input(
        "Diagnosis Year",
        min_value=2010,
        max_value=2035,
        value=2024
    )

st.write("")

predict = st.button("Predict Lung Cancer Risk")


if st.button("Predict"):

    patient = {
        "Patient_ID": patient_id,
        "Age": age,
        "Gender": gender,
        "Smoking_Status": smoking_status,
        "Years_Smoking": years_smoking,
        "Cigarettes_Per_Day": cigarettes_per_day,
        "Secondhand_Smoke_Exposure": secondhand_smoke,
        "Occupation_Exposure": occupation_exposure,
        "Air_Pollution_Level": air_pollution,
        "Family_History": family_history,
        "Genetic_Markers_Positive": genetic,
        "BMI": bmi,
        "Physical_Activity_Level": physical,
        "Alcohol_Consumption": alcohol,
        "Diet_Quality": diet,
        "Region": region,
       
        "Access_to_Healthcare": healthcare,
        "Screening_Frequency": screening,
        "Chronic_Lung_Disease": chronic,
        "Diagnosis_Year": diagnosis_year
    }

    

    # Create dataframe from user input
    input_df = pd.DataFrame([patient])
    #missing values
    input_df["Income_Level"] = 1
    input_df["Education_Level"] = 1
    
    # Manual encoding (same mapping as training)
    binary_map = {"No": 0, "Yes": 1}

    # Gender
    input_df["Gender"] = input_df["Gender"].map({"Male": 1, "Female": 0})

    # Smoking Status
    input_df["Smoking_Status"] = input_df["Smoking_Status"].map({
    "Never": 0,
    "Former": 1,
    "Current": 2
})

    # Binary columns
    binary_cols = [
    "Secondhand_Smoke_Exposure",
    "Occupation_Exposure",
    "Family_History",
    "Genetic_Markers_Positive",
    "Alcohol_Consumption",
    "Chronic_Lung_Disease"
]

    for col in binary_cols:
     input_df[col] = input_df[col].map(binary_map)

# Diet Quality
    input_df["Diet_Quality"] = input_df["Diet_Quality"].map({
    "Poor": 0,
    "Average": 1,
    "Good": 2
})

# Region
    input_df["Region"] = input_df["Region"].map({
    "Rural": 0,
    "Semi Urban": 1,
    "Urban": 2
})





# Access to Healthcare
    input_df["Access_to_Healthcare"] = input_df["Access_to_Healthcare"].map({
    "Poor": 0,
    "Average": 1,
    "Good": 2
})

# Screening Frequency
    input_df["Screening_Frequency"] = input_df["Screening_Frequency"].map({
    "Never": 0,
    "Rarely": 1,
    "Yearly": 2
})



    input_df = input_df[[
    "Patient_ID",
    "Age",
    "Gender",
    "Smoking_Status",
    "Years_Smoking",
    "Cigarettes_Per_Day",
    "Secondhand_Smoke_Exposure",
    "Occupation_Exposure",
    "Air_Pollution_Level",
    "Family_History",
    "Genetic_Markers_Positive",
    "BMI",
    "Physical_Activity_Level",
    "Alcohol_Consumption",
    "Diet_Quality",
    "Region",
    "Income_Level",
    "Education_Level",
    "Access_to_Healthcare",
    "Screening_Frequency",
    "Chronic_Lung_Disease",
    "Diagnosis_Year"
     ]]
   # FINAL RULE-BASED RISK ENGINE

    risk_score = 0

# Smoking
    if smoking_status == "Current":
      risk_score += 3
    elif smoking_status == "Former":
      risk_score += 1

# Years smoking
    if years_smoking >= 20:
      risk_score += 2

# Cigarettes
    if cigarettes_per_day >= 15:
      risk_score += 2

# Age
    if age >= 60:
      risk_score += 2
    elif age >= 45:
      risk_score += 1

# Family history
    if family_history == "Yes":
      risk_score += 2

# Genetic
    if genetic == "Yes":
      risk_score += 2
 
# Chronic disease
    if chronic == "Yes":
      risk_score += 2

# Air pollution
    if air_pollution >= 7:
      risk_score += 1

# Final decision
    if risk_score >= 6:
      result = "High Risk"
      high_risk = min(95, risk_score * 12)
      low_risk = 100 - high_risk
    else:
      result = "Low Risk"
      low_risk = max(70, 100 - risk_score * 10)
      high_risk = 100 - low_risk
    st.write("---")
    st.subheader("Prediction Result")

    if result == "High Risk":
       st.error(f"High Risk ({high_risk:.2f}%)")
    else:
       st.success(f"Low Risk ({low_risk:.2f}%)")
    st.write("Predicted Disease:", result)
    st.progress(float(max(low_risk, high_risk)) / 100)

    st.write("Low Risk Probability :", f"{low_risk:.2f}%")
    st.write("High Risk Probability :", f"{high_risk:.2f}%")
    
    
    # Health Recommendation

    st.write("---")
    st.subheader("Health Recommendations")
    if result =="High Risk":

     st.warning("You may have a higher risk of lung cancer.")

     st.markdown("""
### Recommended Actions

- Consult a lung specialist (Pulmonologist).
- Schedule a CT Scan or Chest X-ray.
- Quit smoking immediately.
- Avoid secondhand smoke.
- Stay away from polluted environments.
- Exercise regularly.
- Eat fruits and green vegetables.
- Go for regular health check-ups.
- Follow your doctor's advice.
""")

    else:

     st.success("Your predicted risk is low.")

    st.markdown("""
### Healthy Lifestyle Tips

- Continue a healthy lifestyle.
- Avoid smoking and tobacco.
- Exercise at least 30 minutes daily.
- Eat a balanced diet.
- Drink enough water.
- Sleep 7-8 hours daily.
- Get regular medical check-ups.
- Avoid excessive alcohol consumption.
""")

# PDF Report

    buffer = BytesIO()

    pdf = canvas.Canvas(buffer)

    pdf.setTitle("Healthcare Diagnosis Report")

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(150, 800, "AI Powered Healthcare Diagnosis")

    pdf.setFont("Helvetica", 12)

    pdf.drawString(50, 760, f"Patient ID : {patient_id}")
    pdf.drawString(50, 740, f"Age : {age}")
    pdf.drawString(50, 720, f"Gender : {gender}")
    pdf.drawString(50, 700, f"Prediction : {result}")

    pdf.drawString(50, 680, f"Low Risk Probability : {low_risk:.2f}%")
    pdf.drawString(50, 660, f"High Risk Probability : {high_risk:.2f}%")

    pdf.drawString(50, 620, "Generated by AI Powered Healthcare Diagnosis System")

    pdf.save()

    buffer.seek(0)

    st.download_button(
    label="📄 Download PDF Report",
    data=buffer,
    file_name="Healthcare_Report.pdf",
    mime="application/pdf"
)