import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib

# ============================================
# Page Title
# ============================================

st.set_page_config(
    page_title="Depression Prediction App",
    layout="centered"
)

st.title("Depression Prediction App")
st.write("Mental Health Prediction using Deep Learning")

# ============================================
# Load Preprocessing Objects
# ============================================

scaler = joblib.load("artifacts/scaler.pkl")
label_encoders = joblib.load("artifacts/label_encoders.pkl")

# ============================================
# Define Model Architecture
# ============================================

class DepressionModel(nn.Module):

    def __init__(self):

        super(DepressionModel, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(17, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        return self.network(x)

# ============================================
# Load Model
# ============================================

model = DepressionModel()

model.load_state_dict(
    torch.load(
        "models/depression_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# ============================================
# User Inputs
# ============================================

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.slider(
    "Age",
    18,
    60,
    25
)

academic_pressure = st.slider(
    "Academic Pressure",
    1,
    5,
    3
)

work_pressure = st.slider(
    "Work Pressure",
    1,
    5,
    3
)

cgpa = st.slider(
    "CGPA",
    0.0,
    10.0,
    7.0
)

study_satisfaction = st.slider(
    "Study Satisfaction",
    1,
    5,
    3
)

job_satisfaction = st.slider(
    "Job Satisfaction",
    1,
    5,
    3
)

sleep_duration = st.selectbox(
    "Sleep Duration",
    [
        "Less than 5 hours",
        "5-6 hours",
        "7-8 hours",
        "More than 8 hours"
    ]
)

dietary_habits = st.selectbox(
    "Dietary Habits",
    [
        "Healthy",
        "Moderate",
        "Unhealthy"
    ]
)

degree = st.selectbox(
    "Degree",
    [
        "B.Tech",
        "BBA",
        "BHM",
        "B.Pharm",
        "LLB"
    ]
)

suicidal_thoughts = st.selectbox(
    "Suicidal Thoughts",
    ["Yes", "No"]
)

work_study_hours = st.slider(
    "Work/Study Hours",
    0,
    12,
    6
)

financial_stress = st.slider(
    "Financial Stress",
    1,
    5,
    3
)

family_history = st.selectbox(
    "Family History of Mental Illness",
    ["Yes", "No"]
)

city = st.selectbox(
    "City",
    [
        "Mumbai",
        "Chennai",
        "Delhi",
        "Bangalore",
        "Hyderabad"
    ]
)

profession = st.selectbox(
    "Profession",
    [
        "Teacher",
        "Chef",
        "Doctor",
        "Student",
        "Business Analyst"
    ]
)

working_professional = st.selectbox(
    "Working Professional or Student",
    [
        "Working Professional",
        "Student"
    ]
)

# ============================================
# Prediction Button
# ============================================

if st.button("Predict Depression"):

    input_data = pd.DataFrame({

        "Gender": [gender],
        "Age": [age],
        "City": [city],
        "Working Professional or Student": [working_professional],
        "Profession": [profession],
        "Academic Pressure": [academic_pressure],
        "Work Pressure": [work_pressure],
        "CGPA": [cgpa],
        "Study Satisfaction": [study_satisfaction],
        "Job Satisfaction": [job_satisfaction],
        "Sleep Duration": [sleep_duration],
        "Dietary Habits": [dietary_habits],
        "Degree": [degree],
        "Have you ever had suicidal thoughts ?": [suicidal_thoughts],
        "Work/Study Hours": [work_study_hours],
        "Financial Stress": [financial_stress],
        "Family History of Mental Illness": [family_history]

    })

    # ============================================
    # Encode Categorical Features
    # ============================================

    categorical_columns = [

        "Gender",
        "City",
        "Working Professional or Student",
        "Profession",
        "Sleep Duration",
        "Dietary Habits",
        "Degree",
        "Have you ever had suicidal thoughts ?",
        "Family History of Mental Illness"
    ]

    for column in categorical_columns:

        le = label_encoders[column]

        value = str(input_data[column].iloc[0])

        if value not in le.classes_:

            value = le.classes_[0]

        input_data[column] = le.transform([value])

    # ============================================
    # Scaling
    # ============================================

    input_scaled = scaler.transform(input_data)

    # ============================================
    # Tensor Conversion
    # ============================================

    input_tensor = torch.tensor(
        input_scaled,
        dtype=torch.float32
    )

    # ============================================
    # Prediction
    # ============================================

    with torch.no_grad():

        output = model(input_tensor)

        probability = torch.sigmoid(output)

        prediction = (
            probability >= 0.5
        ).float()

    # ============================================
    # Result
    # ============================================

    st.subheader("Prediction Result")

    if prediction.item() == 1:

        st.error(
            "High Chance of Depression"
        )

    else:

        st.success(
            "Low Chance of Depression"
        )

    st.write(
        f"Prediction Probability: {probability.item():.4f}"
    )