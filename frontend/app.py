import streamlit as st
import requests
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# API Endpoint URL
API_URL = "http://localhost:8000"

# Styling
st.markdown("""
    <style>
    .fraud-alert {
        padding: 20px;
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .safe-alert {
        padding: 20px;
        background-color: #00cc66;
        color: white;
        border-radius: 5px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Fraud Guard AI Dashboard")
st.write("Machine Learning Fraud Detection System based on strict MVC-like architecture.")

# Create tabs as requested
tab1, tab2 = st.tabs(["🔍 Inference", "🔄 Retrain"])

with tab1:
    st.header("Test Single Transaction")
    st.write("Submit a single transaction to the `/predict` endpoint.")
    
    col1, col2, col3 = st.columns(3)
    
    with col2:
        nameOrig = st.text_input("Origin Account", value="C123456789")
        oldbalanceOrg = st.number_input("Origin Old Balance", min_value=0.0, value=5000.0)

    with col1:
        step = st.number_input("Step (Hour)", min_value=1, value=1)
        type_str = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
        
        # Prevent amount from exceeding old balance for outflows
        max_amt = float(oldbalanceOrg) if type_str != "CASH_IN" else None
        default_amt = min(1000.0, float(oldbalanceOrg)) if type_str != "CASH_IN" else 1000.0
        amount = st.number_input("Amount", min_value=0.0, max_value=max_amt, value=default_amt, step=100.0)
        
    with col2:
        # Auto-calculate Origin New Balance based on type
        if type_str == "CASH_IN":
            newbalanceOrig = oldbalanceOrg + amount
        else:
            newbalanceOrig = max(0.0, oldbalanceOrg - amount)
            
        st.metric("Origin New Balance (Auto)", f"${newbalanceOrig:,.2f}", delta=f"${newbalanceOrig - oldbalanceOrg:,.2f}")
        
    with col3:
        default_dest = "M987654321" if type_str == "PAYMENT" else "C987654321"
        nameDest = st.text_input("Destination Account", value=default_dest)
        oldbalanceDest = st.number_input("Dest Old Balance", min_value=0.0, value=0.0)
        
        # Auto-calculate Dest New Balance based on type
        if type_str == "CASH_IN":
            newbalanceDest = max(0.0, oldbalanceDest - amount)
        elif type_str == "PAYMENT" and nameDest.startswith("M"):
            newbalanceDest = 0.0 # Merchants typically stay 0 in this dataset
        else:
            newbalanceDest = oldbalanceDest + amount
            
        st.metric("Dest New Balance (Auto)", f"${newbalanceDest:,.2f}", delta=f"${newbalanceDest - oldbalanceDest:,.2f}")
        
    st.markdown("---")
    submit_button = st.button("Analyze Transaction", use_container_width=True, type="primary")
        
    if submit_button:
        payload = {
            "step": step,
            "type": type_str,
            "amount": amount,
            "nameOrig": nameOrig,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "nameDest": nameDest,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }
        
        with st.spinner("Running inference pipeline..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    prob = result.get("risk_probability", 0) * 100
                    
                    if result.get("prediction") == 1:
                        st.markdown(f'<div class="fraud-alert">🚨 FRAUD DETECTED (Class 1) 🚨<br><span style="font-size: 18px">Risk Probability: {prob:.2f}%</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="safe-alert">✅ TRANSACTION SAFE (Class 0) ✅<br><span style="font-size: 18px">Risk Probability: {prob:.2f}%</span></div>', unsafe_allow_html=True)
                        
                    st.json(result)
                else:
                    st.error(f"Error from API: {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the FastAPI backend. Is it running?")


with tab2:
    st.header("Upload Monthly Batch Data")
    st.write("Upload a CSV file containing new labeled transactions to trigger the Continuous Learning module (`/update-model`).")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} transactions.")
            st.dataframe(df.head())
            
            target_col = st.text_input("Target Column Name", value="isFraud")
            
            if st.button("Trigger Model Retraining"):
                if target_col not in df.columns:
                    st.error(f"Column '{target_col}' not found in the uploaded CSV.")
                else:
                    with st.spinner("Sending batch to continuous learning pipeline..."):
                        # Convert to list of dicts for JSON payload
                        records = df.to_dict(orient="records")
                        payload = {
                            "transactions": records,
                            "target_column": target_col
                        }
                        
                        try:
                            response = requests.post(f"{API_URL}/update-model", json=payload)
                            
                            if response.status_code == 200:
                                st.success(f"Models successfully updated! ({len(records)} records processed)")
                                st.json(response.json())
                            else:
                                st.error(f"Error updating models: {response.json().get('detail', 'Unknown error')}")
                        except requests.exceptions.ConnectionError:
                            st.error("Failed to connect to the FastAPI backend.")
                            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
