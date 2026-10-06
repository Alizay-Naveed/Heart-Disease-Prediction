import streamlit as st
import pickle
import numpy as np

# Load your exact model file from the folder
with open('Heart Disease Model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient's data below:")

# Input selectors for your 4 chosen discriminating attributes
cp = st.selectbox("Chest Pain Type (cp)", options=[1, 2, 3, 4], 
                  format_func=lambda x: {1: "1 - Typical Angina", 2: "2 - Atypical Angina", 3: "3 - Non-anginal", 4: "4 - Asymptomatic"}[x])

thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)

ca = st.slider("Number of Major Vessels Blocked (ca)", min_value=0, max_value=3, value=0)

oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=6.0, value=0.0, step=0.1)

# Prediction button execution
if st.button("Predict"):
    features = np.array([[cp, thalach, ca, oldpeak]])
    prediction = model.predict(features)
    
    st.subheader("Result:")
    if prediction[0] == 1:
        st.error("⚠️ High risk of heart disease detected (Target: 1)")
    else:
        st.success("✅ Low risk of heart disease detected (Target: 0)")
