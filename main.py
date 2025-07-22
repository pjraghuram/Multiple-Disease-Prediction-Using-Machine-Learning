import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="Medical Condition Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for top banner
st.markdown("""
    <style>
        .top-banner {
            background-color: #003366;
            padding: 15px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: white;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Custom CSS for styling elements
st.markdown("""
<style>
    .diagnosis {
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        margin: 10px 0;
    }
    .positive {
        background-color: #ff4b4b;
        color: white;
    }
    .negative {
        background-color: #28a745;
        color: white;
    }
    .stButton button {
        width: 100%;
        background-color: #007bff;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Load models
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# (Rest of the application logic continues as before)


def create_radar_chart(values, categories, title):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Parameters'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title=title
    )
    return fig

def normalize_values(values, ranges):
    return [(v - r[0]) / (r[1] - r[0]) if r[1] - r[0] != 0 else 0.5 for v, r in zip(values, ranges)]

# Sidebar navigation
with st.sidebar:
    selected = st.radio('Select Disease Prediction',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'])

# Main content
if selected == 'Diabetes Prediction':
    st.title('Diabetes Risk Assessment')
    
    # Create layout
    col1, col2, col3 = st.columns(3)
    
    # First column inputs
    with col1:
        pregnancies = st.number_input('Number of Pregnancies', 0, 20, 0)
        skin_thickness = st.number_input('Skin Thickness', 0, 100, 20)
        dpf = st.number_input('Diabetes Pedigree Function', 0.0, 2.5, 0.47)
        
    # Second column inputs
    with col2:
        glucose = st.number_input('Glucose Level', 0, 200, 85)
        insulin = st.number_input('Insulin Level', 0, 900, 120)
        age = st.number_input('Age', 0, 120, 33)
        
    # Third column inputs
    with col3:
        blood_pressure = st.number_input('Blood Pressure', 0, 200, 77)
        bmi = st.number_input('BMI', 0.0, 70.0, 23.1)
    
    # Analysis button
    if st.button('Analyze Diabetes Risk'):
        values = [pregnancies, glucose, blood_pressure, skin_thickness, 
                 insulin, bmi, dpf, age]
        
        # Define ranges for normalization
        ranges = [(0, 17), (0, 200), (0, 122), (0, 99), 
                 (0, 846), (0, 67.1), (0.078, 2.42), (21, 81)]
        
        normalized_values = normalize_values(values, ranges)
        categories = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                     'Insulin', 'BMI', 'DPF', 'Age']
        
        # Create two columns for visualization and results
        vis_col, res_col = st.columns([2, 1])
        
        with vis_col:
            fig = create_radar_chart(normalized_values, categories, 'Patient Parameters Visualization')
            st.plotly_chart(fig)
            
        with res_col:
            # Make prediction
            prediction = diabetes_model.predict([values])
            
            st.subheader('Diagnosis')
            if prediction[0] == 1:
                st.markdown('<p class="diagnosis positive">High Risk of Diabetes</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="diagnosis negative">Low Risk of Diabetes</p>', unsafe_allow_html=True)
            
            st.info('This prediction is based on the provided parameters. Please consult with a healthcare professional for accurate diagnosis.')
# Heart Disease Section
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Risk Assessment')
    
    # Create three columns for inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age', 0, 120, 45)
        trestbps = st.number_input('Resting Blood Pressure', 0, 200, 120)
        restecg = st.number_input('Resting ECG Results', 0, 2, 1)
        oldpeak = st.number_input('ST Depression', 0.0, 6.0, 1.0)
        
    with col2:
        sex = st.selectbox('Sex', ['Male', 'Female'])
        chol = st.number_input('Cholesterol', 0, 600, 200)
        thalach = st.number_input('Max Heart Rate', 0, 220, 150)
        slope = st.number_input('Slope', 0, 2, 1)
        
    with col3:
        cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
        exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        ca = st.number_input('Number of Major Vessels', 0, 4, 0)
    
    # Additional parameters
    thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])
    
    if st.button('Analyze Heart Disease Risk'):
        # Convert categorical variables
        sex_val = 1 if sex == 'Male' else 0
        cp_val = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
        fbs_val = 1 if fbs == 'Yes' else 0
        exang_val = 1 if exang == 'Yes' else 0
        thal_val = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal)
        
        values = [age, sex_val, cp_val, trestbps, chol, fbs_val, restecg, 
                 thalach, exang_val, oldpeak, slope, ca, thal_val]
        
        # Create two columns for visualization and results
        vis_col, res_col = st.columns([2, 1])
        
        with vis_col:
            # Create visualization
            viz_values = normalize_values(
                [age, trestbps, chol, thalach, oldpeak],
                [(29, 77), (94, 200), (126, 564), (71, 202), (0, 6.2)]
            )
            viz_categories = ['Age', 'Blood Pressure', 'Cholesterol', 'Max Heart Rate', 'ST Depression']
            
            fig = create_radar_chart(viz_values, viz_categories, 'Cardiac Parameters Visualization')
            st.plotly_chart(fig)
        
        with res_col:
            # Make prediction
            prediction = heart_disease_model.predict([values])
            
            st.subheader('Diagnosis')
            if prediction[0] == 1:
                st.markdown('<p class="diagnosis positive">High Risk of Heart Disease</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="diagnosis negative">Low Risk of Heart Disease</p>', unsafe_allow_html=True)
            
            st.info('This prediction is based on the provided parameters. Please consult with a healthcare professional for accurate diagnosis.')

# Parkinson's Disease Section
else:
    st.title("Parkinson's Disease Risk Assessment")
    
    # Create four columns for inputs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', 0.0, 300.0, 120.0)
        Jitter_percent = st.number_input('Jitter(%)', 0.0, 1.0, 0.0)
        DDP = st.number_input('Jitter:DDP', 0.0, 1.0, 0.0)
        APQ3 = st.number_input('Shimmer:APQ3', 0.0, 1.0, 0.0)
        DDA = st.number_input('Shimmer:DDA', 0.0, 1.0, 0.0)
        
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)', 0.0, 500.0, 150.0)
        Jitter_Abs = st.number_input('Jitter(Abs)', 0.0, 1.0, 0.0)
        Shimmer = st.number_input('MDVP:Shimmer', 0.0, 1.0, 0.0)
        APQ5 = st.number_input('Shimmer:APQ5', 0.0, 1.0, 0.0)
        HNR = st.number_input('HNR', 0.0, 50.0, 20.0)
        
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)', 0.0, 300.0, 100.0)
        RAP = st.number_input('MDVP:RAP', 0.0, 1.0, 0.0)
        Shimmer_dB = st.number_input('Shimmer(dB)', 0.0, 1.0, 0.0)
        APQ = st.number_input('MDVP:APQ', 0.0, 1.0, 0.0)
        RPDE = st.number_input('RPDE', 0.0, 1.0, 0.0)
        
    with col4:
        PPQ = st.number_input('MDVP:PPQ', 0.0, 1.0, 0.0)
        NHR = st.number_input('NHR', 0.0, 1.0, 0.0)
        DFA = st.number_input('DFA', 0.0, 1.0, 0.0)
        spread1 = st.number_input('spread1', 0.0, 1.0, 0.0)
        spread2 = st.number_input('spread2', 0.0, 1.0, 0.0)
    
    # Additional parameters in a single column
    D2 = st.number_input('D2', 0.0, 1.0, 0.0)
    PPE = st.number_input('PPE', 0.0, 1.0, 0.0)
    
    if st.button("Analyze Parkinson's Risk"):
        values = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                 Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                 RPDE, DFA, spread1, spread2, D2, PPE]
        
        # Create two columns for visualization and results
        vis_col, res_col = st.columns([2, 1])
        
        with vis_col:
            # Create visualization for key metrics
            key_values = normalize_values(
                [fo, Jitter_percent, Shimmer, NHR, HNR],
                [(88.333, 260.105), (0.002, 0.033), (0.019, 0.119), (0.000, 0.315), (8.441, 33.047)]
            )
            key_categories = ['Fundamental Frequency', 'Jitter', 'Shimmer', 'NHR', 'HNR']
            
            fig = create_radar_chart(key_values, key_categories, 'Voice Parameters Visualization')
            st.plotly_chart(fig)
        
        with res_col:
            # Make prediction
            prediction = parkinsons_model.predict([values])
            
            st.subheader('Diagnosis')
            if prediction[0] == 1:
                st.markdown('<p class="diagnosis positive">High Risk of Parkinson\'s Disease</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="diagnosis negative">Low Risk of Parkinson\'s Disease</p>', unsafe_allow_html=True)
            
            st.info('This prediction is based on the provided parameters. Please consult with a healthcare professional for accurate diagnosis.')
            
