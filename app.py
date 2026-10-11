import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =====================================================================
# 1. LOAD THE OPTIMIZED ML COMPONENTS
# =====================================================================
@st.cache_resource
def load_ml_components():
    # Loading your optimized random_state=3 model and scaler
    with open('heart_disease_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    svr_model, data_scaler = load_ml_components()
except FileNotFoundError:
    st.error("⚠️ Pipeline files not found! Make sure 'heart_disease_model.pkl' and 'scaler.pkl' are in this folder.")
    st.stop()

# =====================================================================
# 2. WEB USER INTERFACE SYSTEM
# =====================================================================
st.set_page_config(page_title="Heart Disease Prediction System", page_icon="🫀", layout="centered")

st.title("🫀 Heart Disease Prediction System")
st.write("Input patient clinical metrics below to calculate the estimated blockage severity stage.")
st.markdown("---")

# Grouping inputs into two clean visual columns
col1, col2 = st.columns(2)

with col1:
    cp = st.selectbox(
        "Chest Pain Type (cp)",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Stage 1: Typical Angina",
            2: "Stage 2: Atypical Angina",
            3: "Stage 3: Non-anginal Pain",
            4: "Stage 4: Asymptomatic"
        }[x]
    )
    
    thalach = st.number_input(
        "Max Heart Rate Achieved (thalach)",
        min_value=50,
        max_value=250,
        value=150,
        step=1
    )

with col2:
    oldpeak = st.number_input(
        "ST Depression Induced by Exercise (oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )
    
    ca = st.selectbox(
        "Number of Major Vessels Colored (ca)",
        options=[0, 1, 2, 3]
    )

# =====================================================================
# 3. BACKEND INFERENCE & FRONTEND PROCESSING LAYER
# =====================================================================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Generate Severity Diagnosis", type="primary", use_container_width=True):
    
    # 1. Build a dictionary forcing the exact data types from your notebook training step
    patient_data = {
        'cp': [int(cp)],
        'thalach': [int(thalach)],
        'oldpeak': [float(oldpeak)],
        'ca': [int(ca)]
    }
    
    # 2. Convert to a DataFrame with explicit indices
    raw_df = pd.DataFrame(patient_data)
    
    # 3. Scale the metrics using your loaded scaler tool configuration
    scaled_metrics = data_scaler.transform(raw_df)
    
    # 4. Extract the continuous regression decimal output from your SVR model
    decimal_prediction = float(svr_model.predict(scaled_metrics)[0])
    
    # 5. FRONTEND ROUNDING AND CLIPPING OPERATION
    rounded_stage = int(np.clip(np.round(decimal_prediction), 0, 4))
    
    # =====================================================================
    # 4. RESULTS DISPLAY DASHBOARD
    # =====================================================================
    st.markdown("---")
    st.subheader("Clinical Diagnostic Report")
    
    # Output display partitioned by rounded integer classification thresholds
    if rounded_stage == 0:
        st.success(f"🟢 **Diagnosis: Stage {rounded_stage} (Healthy)**")
        st.write("No significant coronary artery blockage or narrowing detected.")
    elif rounded_stage >= 1 and rounded_stage <= 3:
        st.warning(f"🟡 **Diagnosis: Stage {rounded_stage} (Mild to Moderate Disease)**")
        st.write("Early to mid-stage vessel constriction noted. Clinical monitoring advised.")
    else:
        st.error(f"🔴 **Diagnosis: Stage {rounded_stage} (Severe Multi-Vessel Disease)**")
        st.write("Significant multi-vessel obstruction detected. High risk factor present.")
        
    # 5. INTEGRATED DYNAMIC PROGRESS BAR & DECIMAL MARKER
    st.markdown("<br>", unsafe_allow_html=True)
    st.write(f"**Continuous Severity Metric Value:** `{decimal_prediction:.2f}` / 4.00")
    
    # Normalize the decimal value to a percentage (0.0 to 1.0) so the Streamlit bar fills accurately
    normalized_progress = float(np.clip(decimal_prediction / 4.0, 0.0, 1.0))
    st.progress(normalized_progress)
