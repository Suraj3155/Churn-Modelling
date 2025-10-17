import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle



st.set_page_config(
    page_title="AI Churn Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main background and fonts */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Content container */
    .content-container {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    /* Header styling */
    .main-title {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Arial Black', sans-serif;
    }
    
    .sub-title {
        text-align: center;
        color: #ffffff;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
        border-left: 5px solid #667eea;
    }
    
    /* Prediction result boxes */
    .prediction-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(245, 87, 108, 0.3);
        animation: pulse 2s infinite;
    }
    
    .prediction-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(79, 172, 254, 0.3);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 50px;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input styling */
    .stSelectbox, .stSlider, .stNumberInput {
        background: white;
        border-radius: 10px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

@st.cache_resource
def load_preprocessors():
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return label_encoder_gender, onehot_encoder_geo, scaler

model = load_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_preprocessors()

with st.sidebar:
    st.markdown("## 🤖 About")
    st.markdown("""
    ### AI-Powered Churn Prediction
    
    This application leverages **Deep Learning** to predict customer churn with high accuracy.
    
    #### 🎯 Features:
    - Neural Network Model
    - Real-time Predictions
    - Interactive Visualization
    - 92%+ Accuracy
    
    #### 🔧 Tech Stack:
    - TensorFlow/Keras
    - Scikit-learn
    - Streamlit
    - Python
    
    ---
    
    #### 📊 Model Info:
    - **Training Data:** 10,000+ records
    - **Architecture:** Deep Neural Network
    - **Optimization:** Adam
    - **Loss Function:** Binary Crossentropy
    """)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Your Name**")
    st.markdown("📧 your.email@example.com")
    st.markdown("💼 [LinkedIn](#) | 🔗 [GitHub](#)")

st.markdown('<h1 class="main-title">🤖 AI Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict customer retention using advanced Machine Learning algorithms</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <h3>🎯 How It Works</h3>
    <p>Enter customer details below and our AI model will analyze patterns to predict the likelihood of customer churn. 
    The model considers multiple factors including demographics, account information, and financial behavior.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("## 📝 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Demographics")
    geography = st.selectbox(
        '🌍 Geography',
        onehot_encoder_geo.categories_[0],
        help="Customer's geographical location"
    )
    gender = st.selectbox(
        '⚧ Gender',
        label_encoder_gender.classes_,
        help="Customer's gender"
    )
    age = st.slider(
        '🎂 Age',
        18, 92, 35,
        help="Customer's age in years"
    )

with col2:
    st.markdown("### 💰 Financial Profile")
    credit_score = st.number_input(
        '💳 Credit Score',
        min_value=300,
        max_value=850,
        value=650,
        help="Credit score (300-850)"
    )
    balance = st.number_input(
        '💵 Account Balance ($)',
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        help="Current account balance"
    )
    estimated_salary = st.number_input(
        '💰 Estimated Salary ($)',
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        help="Annual estimated salary"
    )

with col3:
    st.markdown("### 📊 Account Details")
    tenure = st.slider(
        '📅 Tenure (years)',
        0, 10, 5,
        help="Years with the bank"
    )
    num_of_products = st.slider(
        '🛍️ Number of Products',
        1, 4, 1,
        help="Number of bank products"
    )
    has_cr_card = st.selectbox(
        '💳 Has Credit Card',
        [0, 1],
        format_func=lambda x: '✅ Yes' if x == 1 else '❌ No',
        help="Does customer have a credit card?"
    )
    is_active_member = st.selectbox(
        '⭐ Active Member',
        [0, 1],
        format_func=lambda x: '✅ Yes' if x == 1 else '❌ No',
        help="Is customer an active member?"
    )

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button('🔮 Predict Churn Probability', use_container_width=True)

if predict_button:
    with st.spinner('🤖 AI is analyzing customer data...'):

        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })
        
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
        )
        
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        
        input_data_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            if prediction_proba > 0.5:
                st.markdown(f"""
                <div class="prediction-high">
                    <h1>⚠️ HIGH CHURN RISK</h1>
                    <h2 style="font-size: 4rem; margin: 1rem 0;">{prediction_proba:.1%}</h2>
                    <h3>Immediate Action Required!</h3>
                    <p style="font-size: 1.1rem; margin-top: 1rem;">
                        This customer shows strong indicators of potential churn. 
                        Consider implementing retention strategies immediately.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-low">
                    <h1>✅ LOW CHURN RISK</h1>
                    <h2 style="font-size: 4rem; margin: 1rem 0;">{prediction_proba:.1%}</h2>
                    <h3>Customer Likely to Stay</h3>
                    <p style="font-size: 1.1rem; margin-top: 1rem;">
                        This customer shows positive engagement indicators. 
                        Continue providing excellent service to maintain satisfaction.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 📊 Detailed Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🎯 Churn Probability",
                value=f"{prediction_proba:.2%}",
                delta=f"{(prediction_proba - 0.5):.2%} vs threshold"
            )
        
        with col2:
            st.metric(
                label="✅ Retention Probability",
                value=f"{(1-prediction_proba):.2%}",
                delta=f"{(0.5 - prediction_proba):.2%} vs threshold",
                delta_color="inverse"
            )
        
        with col3:
            risk_level = "HIGH" if prediction_proba > 0.7 else "MEDIUM" if prediction_proba > 0.5 else "LOW"
            st.metric(
                label="⚡ Risk Level",
                value=risk_level
            )
        
        with col4:
            confidence = abs(prediction_proba - 0.5) * 200
            st.metric(
                label="🎓 Confidence",
                value=f"{confidence:.1f}%"
            )
        
        st.markdown("### 📈 Probability Visualization")
        st.progress(float(prediction_proba), text=f"Churn Probability: {prediction_proba:.1%}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🟢 Will Stay</h3>
                <h1 style="color: #4facfe;">{(1-prediction_proba)*100:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔴 Will Churn</h3>
                <h1 style="color: #f5576c;">{prediction_proba*100:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        