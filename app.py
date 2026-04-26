import streamlit as st
import requests
import json
from typing import Optional, Dict, Any
import pandas as pd

st.set_page_config(
    page_title="Credit Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Notion-style CSS
st.markdown(
    """
    <style>
    :root {
        --accent-primary: #5B4EFF;
        --accent-secondary: #00D9FF;
        --danger: #FF6B6B;
        --success: #06D6A0;
        --warning: #FFB627;
        --bg-light: #F8FAFC;
        --text-primary: #1A202C;
        --text-secondary: #718096;
        --border-color: #E2E8F0;
    }
    
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
    }
    
    .notion-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .notion-subheader {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .section-description {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    
    .prediction-card {
        padding: 2rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        font-size: 1rem;
        border: 2px solid;
        background: white;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    .high-risk {
        background: linear-gradient(135deg, #FEF3F3 0%, #FFEBEB 100%);
        border-color: var(--danger);
    }
    
    .low-risk {
        background: linear-gradient(135deg, #F0FDF5 0%, #E6F9F1 100%);
        border-color: var(--success);
    }
    
    .medium-risk {
        background: linear-gradient(135deg, #FFFBF0 0%, #FFF5E6 100%);
        border-color: var(--warning);
    }
    
    .risk-text-high {
        color: var(--danger);
        font-weight: 600;
    }
    
    .risk-text-low {
        color: var(--success);
        font-weight: 600;
    }
    
    .risk-text-medium {
        color: var(--warning);
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-primary);
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.25rem;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 2rem 0;
        border: none;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--accent-primary) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary) 0%, #7C3AED 100%);
        color: white !important;
        border: none;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 20px rgba(91, 78, 255, 0.3);
        transform: translateY(-2px);
    }
    
    .about-section {
        background: var(--bg-light);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--accent-secondary);
    }
    
    .input-description {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: -0.5rem;
        margin-bottom: 0.5rem;
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

st.markdown('<div class="notion-header">Credit Risk Assessment</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-description">Enter the loan applicant details below to evaluate credit risk and receive a prediction from our machine learning model.</div>',
    unsafe_allow_html=True
)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="notion-subheader">Credit Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-description">Basic credit information and loan terms</div>', unsafe_allow_html=True)
    
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
    st.markdown('<div class="notion-subheader">Credit History</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-description">Account activity and inquiries from recent months</div>', unsafe_allow_html=True)
    
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

st.markdown('<hr class="divider">', unsafe_allow_html=True)
col3, col4 = st.columns([2, 1])

with col3:
    st.markdown('<div class="notion-subheader">Account Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-description">Details about existing accounts and mortgages</div>', unsafe_allow_html=True)
    
    col_acc1, col_acc2 = st.columns(2)
    
    with col_acc1:
        acc_open_past_24mths = st.number_input(
            "Accounts Opened (24 months)",
            min_value=0.0,
            max_value=50.0,
            value=1.0,
            step=1.0,
            help="Number of accounts opened in past 24 months",
        )
    
    with col_acc2:
        mort_acc = st.number_input(
            "Mortgage Accounts",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=1.0,
            help="Number of mortgage accounts",
        )

with col4:
    st.markdown('<div class="notion-subheader" style="margin-top: 0;">Action</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-description">Get your prediction</div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")
    predict_button = st.button(
        "Get Prediction",
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
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown('<div class="notion-subheader">Assessment Results</div>', unsafe_allow_html=True)
                
                # Determine risk level
                if prediction_probability < 0.33:
                    risk_level = "Low"
                    risk_class = "low-risk"
                    risk_text_class = "risk-text-low"
                    interpretation = "This applicant has a low probability of defaulting on the loan. Recommended for approval."
                elif prediction_probability < 0.66:
                    risk_level = "Medium"
                    risk_class = "medium-risk"
                    risk_text_class = "risk-text-medium"
                    interpretation = "This applicant has a moderate probability of defaulting on the loan. Further review recommended."
                else:
                    risk_level = "High"
                    risk_class = "high-risk"
                    risk_text_class = "risk-text-high"
                    interpretation = "This applicant has a high probability of defaulting on the loan. Recommend rejection."
                
                # Create metrics row
                col_pred1, col_pred2, col_pred3 = st.columns(3)

                if prediction_value == 1: 
                    status_text = "Reject"
                    status_color = "#FF6B6B"
                else: 
                    status_text = "Accept"
                    status_color = "#06D6A0"
                
                with col_pred1: 
                    st.markdown(f'<div style="text-align: center;"><div class="metric-value" style="color: {status_color};">{status_text}</div><div class="metric-label">Recommendation</div></div>', unsafe_allow_html=True)
                
                with col_pred2:
                    st.markdown(f'<div style="text-align: center;"><div class="metric-value">{prediction_probability:.1%}</div><div class="metric-label">Default Probability</div></div>', unsafe_allow_html=True)
                
                with col_pred3:
                    st.markdown(f'<div style="text-align: center;"><div class="metric-value" style="color: var(--accent-secondary);">{risk_level}</div><div class="metric-label">Risk Level</div></div>', unsafe_allow_html=True)
                
                # Display prediction card
                st.markdown(
                    f'<div class="prediction-card {risk_class}"><div class="{risk_text_class}" style="font-size: 1.1rem; margin-bottom: 0.5rem;">{risk_level} Risk</div><p>{interpretation}</p></div>',
                    unsafe_allow_html=True,
                )
                
                # Display input summary
                with st.expander("View Input Summary", expanded=False):
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
                            f"{int(inq_last_6mths)}",
                            f"{int(open_il_12m)}",
                            f"{int(acc_open_past_24mths)}",
                            f"{int(mort_acc)}",
                            f"{int(num_tl_op_past_12m)}",
                            f"{percent_bc_gt_75:.0f}%",
                            sub_grade,
                            term,
                        ],
                    })
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
            else:
                st.error("No prediction received from the model.")
        else:
            st.error(f"API Error: {response.status_code}")
            error_detail = response.json() if response.text else "Unknown error"
            st.error(f"Details: {error_detail}")
    
    
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to the API at {API_URL}. "
            "Please ensure the backend API is running."
        )
    except requests.exceptions.Timeout:
        st.error("Request timed out. The API took too long to respond.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Sidebar information
with st.sidebar:
    st.markdown("---")
    st.markdown('<div class="notion-subheader">About</div>', unsafe_allow_html=True)
    st.markdown(
        """
        This application uses a machine learning model to predict 
        the probability of loan default based on applicant features.
        
        **Features Analyzed:**
        - Credit profile (FICO score, interest rate)
        - Credit history (inquiries, account age)
        - Account information (open accounts, mortgages)
        
        **Model Output:**
        - Probability score (0-100%)
        - Risk classification
        - Recommendation
        """
    )
    
    st.markdown("---")
    st.markdown('<div class="notion-subheader">Settings</div>', unsafe_allow_html=True)
    if st.button("Reset Form"):
        st.rerun()
