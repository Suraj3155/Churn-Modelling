import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    /* Main background and fonts */
    .main {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    /* Header styling */
    .main-title {
        background: linear-gradient(120deg, #11998e 0%, #38ef7d 100%);
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
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(17, 153, 142, 0.4);
    }
    
    /* Prediction result box */
    .prediction-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(245, 87, 108, 0.3);
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Salary display */
    .salary-display {
        font-size: 4rem;
        font-weight: bold;
        color: #fff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11998e 0%, #38ef7d 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 50px;
        box-shadow: 0 5px 15px rgba(17, 153, 142, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(17, 153, 142, 0.6);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #11998e;
    }
    
    /* Metric card */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
        border-left: 5px solid #11998e;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('regression_model.h5')

# Load encoders and scaler
@st.cache_resource
def load_preprocessors():
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return label_encoder_gender, onehot_encoder_geo, scaler

# Initialize
model = load_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_preprocessors()

# Sidebar
with st.sidebar:
    st.markdown("## 💰 About")
    st.markdown("""
    ### AI-Powered Salary Prediction
    
    This application uses **Deep Learning Regression** to predict estimated salaries based on customer profiles.
    
    #### 🎯 Features:
    - Neural Network Regression
    - Real-time Predictions
    - Interactive Visualization
    - High Accuracy Model
    
    #### 🔧 Tech Stack:
    - TensorFlow/Keras
    - Scikit-learn
    - Streamlit
    - Python
    
    ---
    
    #### 📊 Model Info:
    - **Architecture:** Deep Neural Network
    - **Layers:** 64 → 32 → 1
    - **Optimization:** Adam
    - **Loss Function:** MAE
    - **Activation:** ReLU + Linear
    
    #### 📈 Performance:
    - Low Mean Absolute Error
    - Fast Predictions
    - Robust & Reliable
    """)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Suraj**")
    st.markdown("📧 suraj@example.com")
    st.markdown("💼 [LinkedIn](#) | 🔗 [GitHub](https://github.com/Suraj3155)")

# Main content
st.markdown('<h1 class="main-title">💰 AI Salary Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict estimated salary using advanced Machine Learning regression algorithms</p>', unsafe_allow_html=True)

# Info banner
st.markdown("""
<div class="info-card">
    <h3>🎯 How It Works</h3>
    <p>Enter customer details below and our AI regression model will predict the estimated salary based on 
    demographic information, banking behavior, and account characteristics. The model uses deep learning 
    to identify complex patterns in the data.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Input Section
st.markdown("## 📝 Customer Information")

# Create three columns for better organization
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
    st.markdown("### 💳 Financial Profile")
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
    exited = st.selectbox(
        '🚪 Exited',
        [0, 1],
        format_func=lambda x: '✅ Yes' if x == 1 else '❌ No',
        help="Has customer exited?"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button('🔮 Predict Estimated Salary', use_container_width=True)

if predict_button:
    with st.spinner('🤖 AI is analyzing customer profile...'):
        # Prepare the input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'Exited': [exited]
        })
        
        # One-hot encode 'Geography'
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
        )
        
        # Combine one-hot encoded columns with input data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Predict salary
        prediction = model.predict(input_data_scaled)
        predicted_salary = prediction[0][0]
        
        # Display results
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("## 💸 Prediction Results")
        
        # Main prediction display
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.markdown(f"""
            <div class="prediction-result">
                <h2>💰 Estimated Annual Salary</h2>
                <div class="salary-display">${predicted_salary:,.2f}</div>
                <p style="font-size: 1.2rem; margin-top: 1rem;">
                    Based on AI analysis of customer profile and banking behavior
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detailed breakdown
        st.markdown("### 📊 Salary Breakdown")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            monthly_salary = predicted_salary / 12
            st.metric(
                label="💵 Monthly Salary",
                value=f"${monthly_salary:,.2f}"
            )
        
        with col2:
            weekly_salary = predicted_salary / 52
            st.metric(
                label="📅 Weekly Salary",
                value=f"${weekly_salary:,.2f}"
            )
        
        with col3:
            daily_salary = predicted_salary / 365
            st.metric(
                label="☀️ Daily Salary",
                value=f"${daily_salary:,.2f}"
            )
        
        with col4:
            hourly_salary = predicted_salary / (365 * 8)
            st.metric(
                label="⏰ Hourly Rate",
                value=f"${hourly_salary:,.2f}"
            )
        
        # Salary range estimation
        st.markdown("### 📈 Confidence Range")
        
        # Estimate confidence interval (±10% for visualization)
        lower_bound = predicted_salary * 0.9
        upper_bound = predicted_salary * 1.1
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📉 Lower Estimate</h4>
                <h2 style="color: #f5576c;">${lower_bound:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 Predicted Salary</h4>
                <h2 style="color: #11998e;">${predicted_salary:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 Upper Estimate</h4>
                <h2 style="color: #4facfe;">${upper_bound:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Salary comparison
        st.markdown("### 💼 Market Position")
        
        # Define salary brackets for comparison
        if predicted_salary < 30000:
            bracket = "Entry Level"
            color = "#f5576c"
            message = "Below average salary range"
        elif predicted_salary < 60000:
            bracket = "Mid-Level"
            color = "#ffa726"
            message = "Average salary range"
        elif predicted_salary < 100000:
            bracket = "Senior Level"
            color = "#66bb6a"
            message = "Above average salary range"
        else:
            bracket = "Executive Level"
            color = "#4facfe"
            message = "Premium salary range"
        
        st.markdown(f"""
        <div style="background: white; padding: 2rem; border-radius: 15px; text-align: center; border-left: 5px solid {color};">
            <h3>Salary Bracket: <span style="color: {color};">{bracket}</span></h3>
            <p style="font-size: 1.1rem; color: #666;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("### 💡 Profile Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **📊 Customer Profile Summary:**
            - Geography: {geography}
            - Age: {age} years
            - Credit Score: {credit_score}
            - Account Balance: ${balance:,.2f}
            """)
        
        with col2:
            st.success(f"""
            **💼 Account Details:**
            - Tenure: {tenure} years
            - Products: {num_of_products}
            - Credit Card: {'Yes' if has_cr_card else 'No'}
            - Active: {'Yes' if is_active_member else 'No'}
            """)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: white; border-radius: 15px; margin-top: 2rem;'>
    <h3 style='color: #11998e;'>🚀 Built with Advanced AI Technology</h3>
    <p style='font-size: 1.1rem; color: #666;'>
        <strong>TensorFlow</strong> • <strong>Scikit-learn</strong> • <strong>Streamlit</strong> • <strong>Python</strong>
    </p>
    <p style='color: #999; margin-top: 1rem;'>
        © 2024 AI Salary Predictor | Developed with ❤️ for Better Financial Insights
    </p>
</div>
""", unsafe_allow_html=True)