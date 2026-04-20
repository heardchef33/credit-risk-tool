import streamlit as st
import requests
import json
from typing import Optional, Dict, Any
import pandas as pd

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 1.2rem;
    }
    .high-risk {
        background-color: #ffcccc;
        border: 2px solid #ff0000;
    }
    .low-risk {
        background-color: #ccffcc;
        border: 2px solid #00cc00;
    }
    .medium-risk {
        background-color: #ffffcc;
        border: 2px solid #ffaa00;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# API Configuration

if "BACKEND_URL" in st.secrets:
    DEFAULT_URL = st.secrets["BACKEND_URL"]
else:
    DEFAULT_URL = "http://localhost:8000/api/v1"

API_URL = st.sidebar.text_input(
    "API URL",
    value=DEFAULT_URL,
    help="Base URL of the credit risk model API",
)

st.markdown('<div class="main-header">💰 Credit Risk Prediction Model</div>', unsafe_allow_html=True)
st.markdown(
    "Enter the loan applicant details below to get a risk prediction from the model."
)

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Credit Profile")
    int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=0.1,
        help="Annual interest rate on the loan",
    )
    
    fico_range_high = st.number_input(
        "FICO Score Range High",
        min_value=300.0,
        max_value=850.0,
        value=700.0,
        step=10.0,
        help="Upper bound of FICO credit score range",
    )
    
    sub_grade = st.selectbox(
        "Loan Grade",
        ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5", "E1", "E2", "E3", "E4", "E5", "F1", "F2", "F3", "F4", "F5", "G1", "G2", "G3", "G4", "G5"],
        index=2,
        help="LC assigned loan subgrade",
    )
    
    term = st.selectbox(
        "Loan Term",
        ["36 months", "60 months"],
        help="The number of payments on the loan",
    )

with col2:
    st.subheader("📋 Credit History")
    inq_last_6mths = st.number_input(
        "Inquiries in Last 6 Months",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=1.0,
        help="Number of inquiries in past 6 months",
    )
    
    open_il_12m = st.number_input(
        "Open Installment Accounts (12m)",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=1.0,
        help="Number of open installment accounts opened in past 12 months",
    )
    
    num_tl_op_past_12m = st.number_input(
        "Total Accounts Opened (12m)",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=1.0,
        help="Number of total accounts opened in past 12 months",
    )
    
    percent_bc_gt_75 = st.number_input(
        "Bank Cards >75% Utilization (%)",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0,
        help="Percentage of bank credit cards with utilization >75%",
    )

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.subheader("💳 Account Information")
    acc_open_past_24mths = st.number_input(
        "Accounts Opened (24 months)",
        min_value=0.0,
        max_value=50.0,
        value=1.0,
        step=1.0,
        help="Number of accounts opened in past 24 months",
    )
    
    mort_acc = st.number_input(
        "Mortgage Accounts",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=1.0,
        help="Number of mortgage accounts",
    )

# Prepare prediction button
col_button1, col_button2 = st.columns([1, 3])
with col_button1:
    predict_button = st.button(
        "🔮 Get Prediction",
        key="predict_button",
        use_container_width=True,
        type="primary",
    )

if predict_button:
    # Prepare the input data
    input_data = {
        "inputs": [
            {
                "int_rate": int_rate,
                "fico_range_high": fico_range_high,
                "inq_last_6mths": inq_last_6mths,
                "open_il_12m": open_il_12m,
                "acc_open_past_24mths": acc_open_past_24mths,
                "mort_acc": mort_acc,
                "num_tl_op_past_12m": num_tl_op_past_12m,
                "percent_bc_gt_75": percent_bc_gt_75,
                "sub_grade": sub_grade,
                "term": term,
            }
        ]
    }
    
    try:
        with st.spinner("⏳ Making prediction..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=input_data,
                timeout=30,
            )
        
        if response.status_code == 200:
            result = response.json()
            print(result)
            predictions = result.get("predictions", [])
            prediction_probabilities = result.get("prediction_probabilities", [])
            version = result.get("version", "Unknown")
            
            if predictions:
                prediction_value = predictions[0]
                prediction_probability = prediction_probabilities[0][1]
                
                # Display prediction result
                st.markdown("---")
                st.subheader("📈 Prediction Results")
                
                # Determine risk level
                if prediction_probability < 0.33:
                    risk_level = "🟢 Low"
                    risk_class = "low-risk"
                    interpretation = "This applicant has a low probability of defaulting on the loan."
                elif prediction_probability < 0.66:
                    risk_level = "🟡 Medium"
                    risk_class = "medium-risk"
                    interpretation = "This applicant has a moderate probability of defaulting on the loan."
                else:
                    risk_level = "🔴 High"
                    risk_class = "high-risk"
                    interpretation = "This applicant has a high probability of defaulting on the loan."
                
                col_pred1, col_pred2, col_pred3 = st.columns(3)

                if prediction_value == 1: 

                    with col_pred1: 
                        st.metric("Status", f"❌ Reject")
                
                else: 

                    with col_pred1: 
                        st.metric("Status", f"✅ Accept")
                
                with col_pred2:
                    st.metric("Default Probability", f"{prediction_probability:.2%}")
                
                with col_pred3:
                    st.metric("Risk Level", risk_level)
                
                st.markdown(
                    f'<div class="prediction-box {risk_class}"><strong>{risk_level}</strong><br/>{interpretation}</div>',
                    unsafe_allow_html=True,
                )
                
                # Display input summary
                with st.expander("📋 Input Summary", expanded=False):
                    summary_df = pd.DataFrame({
                        "Feature": [
                            "Interest Rate",
                            "FICO Score High",
                            "Inquiries (6m)",
                            "Open Installment (12m)",
                            "Accounts Opened (24m)",
                            "Mortgage Accounts",
                            "Total Accounts (12m)",
                            "Bank Cards >75%",
                            "Sub Grade",
                            "Term",
                        ],
                        "Value": [
                            f"{int_rate}%",
                            f"{fico_range_high}",
                            f"{inq_last_6mths}",
                            f"{open_il_12m}",
                            f"{acc_open_past_24mths}",
                            f"{mort_acc}",
                            f"{num_tl_op_past_12m}",
                            f"{percent_bc_gt_75}%",
                            sub_grade,
                            term,
                        ],
                    })
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
            else:
                st.error("❌ No prediction received from the model.")
        else:
            st.error(f"❌ API Error: {response.status_code}")
            error_detail = response.json() if response.text else "Unknown error"
            st.error(f"Details: {error_detail}")
    
    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ Could not connect to the API at {API_URL}. "
            "Please ensure the backend API is running."
        )
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. The API took too long to respond.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

# Sidebar information
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown(
        """
        This application uses a machine learning model to predict 
        the probability of loan default based on applicant features.
        
        **Features Used:**
        - Credit profile (FICO score, interest rate)
        - Credit history (inquiries, account age)
        - Account information (open accounts, mortgages)
        
        **Model Output:**
        - Probability score (0-1)
        - Risk classification
        """
    )
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    if st.button("Reset to Default Values"):
        st.rerun()
