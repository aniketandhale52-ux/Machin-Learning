import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_assets():
    model = pickle.load(open("random_forest_churn_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_assets()

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

# Dashboard Header
st.title("🔮 Customer Churn Prediction Dashboard")
st.caption("Enter customer demographics and account details below to evaluate retention risk in real-time.")
st.markdown("---")

# Main Input Form Layout (2 Columns)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("👤 Demographic Details")
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    country = st.selectbox("Country", ["France", "Germany", "Spain"])
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0)

with col2:
    st.subheader("🏦 Financial & Account Profile")
    
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=600)
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0)
    
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
        num_products = st.number_input("Products Held", min_value=1, max_value=4, value=2)
    with sub_col2:
        has_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        active_member = st.selectbox("Is Active Member", ["Yes", "No"])

# Convert categorical inputs to numeric formats expected by the model
germany = 1 if country == "Germany" else 0
spain = 1 if country == "Spain" else 0
male = 1 if gender == "Male" else 0
has_card_val = 1 if has_card == "Yes" else 0
active_member_val = 1 if active_member == "Yes" else 0

st.markdown("---")

# Prediction Action Section
if st.button("🚀 Analyze Churn Risk"):
    
    data = np.array([[
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        has_card_val,
        active_member_val,
        salary,
        germany,
        spain,
        male
    ]])

    # Scale data & run prediction
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)[0][1] * 100

    # Display Results Card
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric("Churn Probability Score", f"{prediction_proba:.1f}%")

    with res_col2:
        if prediction[0] == 1:
            st.error("🚨 **High Risk: Customer is Likely to Exit**")
            st.warning("Recommended Action: Offer targeted retention incentives or a account manager review.")
        else:
            st.success("✅ **Low Risk: Customer is Likely to Stay**")
            st.info("Recommended Action: Eligible for cross-selling and loyalty rewards.")