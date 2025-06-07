import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import time

import google.generativeai as genai
# Import K from tf_keras.backend to use K.clear_session()
from tf_keras import backend as K


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This hides all GPUs from TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # This disables oneDNN optimizations as suggested by a warning

# --- Google Gemini Integration ---
# (No direct import of GenerativeModel, configure from google.generativeai is needed
# as genai.configure and genai.GenerativeModel are used directly after importing genai)

# Assuming st_copy_to_clipboard is a custom component you have installed
from st_copy_to_clipboard import st_copy_to_clipboard

# --- Company Branding (DEFINED BEFORE set_page_config) ---
COMPANY_NAME = "ABC Telecom"

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title=f"Customer Churn Prediction - {COMPANY_NAME}",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.success("Application started and dependencies checked.")
st.write("Ready to predict customer churn!")



# --- Configure Google Gemini API ---
# It's highly recommended to store API keys in Streamlit Secrets:
# 1. Create a `.streamlit` folder in your project root.
# 2. Create a `secrets.toml` file inside `.streamlit`.
# 3. Add your API key: `GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"`
# 4. Access it in your app using `st.secrets["GEMINI_API_KEY"]`
try:
    # Attempt to get API key from Streamlit secrets, then environment variables
    # Streamlit secrets take precedence in deployed apps
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in Streamlit secrets or environment variables.")
    
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"]) 
    gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-1.5-flash', etc.
    st.success("Connected to Google Gemini API.")
except Exception as e:
    gemini_model = None
    st.error(f"Failed to connect to Google Gemini API: {e}")
    st.info("Please ensure your `GEMINI_API_KEY` is correctly set in Streamlit secrets or as an environment variable.")

# --- Pre-listed Offers ---
PRE_LISTED_OFFERS = {
    "high_churn": [
        "Special 15% discount on monthly bill for next 3 months (Code: CHURN15)",
        "Upgrade to unlimited data for free for 1 month (Code: DATAUNLTD)",
        "Exclusive loyalty bonus points worth £20 on next bill (Code: LOYALTY20)",
        "Free upgrade to Premium Fibre Broadband for 6 months (Code: FIBREPREM)",
        "Add a new family line with 50% off for 6 months (Code: FAMILY50)",
    ],
    "low_churn": [
        "Thank you bonus: 5GB extra data for next 3 months (Code: THANKYOU5G)",
        "Early upgrade eligibility to newest phone model (Details: Check system)",
        "Referral bonus: £25 credit for referring a friend (Details: Refer program)",
        "Exclusive access to 'ABC Telecom Rewards' portal (Details: Loyalty link)",
        "Proactive check-in call to ensure satisfaction (No offer, just service)",
    ],
    "complaint_specific": [
        "Complaints resolution credit: £10 off next bill (Code: COMPLAINT10)",
        "Dedicated technical support line for 3 months (Code: TECHSUPP)",
        "Waiver of recent late payment fee (Code: LATEFEEWAIVER)",
    ]
}

# --- Path Setup ---
# Ensures your custom modules can be imported
project_root = os.path.dirname(os.path.abspath(__file__))
if os.path.join(project_root, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(project_root, 'src'))

# Import your custom modules
try:
    from data_loader import load_data
    from model_predictor import load_all_models, predict_churn
    from preprocessor import preprocess_data
except ImportError as e:
    st.error(f"Error importing custom modules. Please ensure `data_loader.py`, `model_predictor.py`, and `preprocessor.py` are in the 'src' directory. Error: {e}")
    st.stop() # Stop the app if core modules can't be loaded

# --- Custom CSS for Styling and Transient Alert ---
st.markdown("""
<style>
    /* Base styles */
    .reportview-container {
        background: #1E2130;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FF4B4B;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    p, li, div, .stDataFrame {
        color: #FAFAFA;
        font-family: 'Open Sans', sans-serif;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 50px; /* Changed to more rounded, personal preference */
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #E03C3C;
        color: white;
    }

    /* CSS for the transient flash alert */
    @keyframes fadeOut {
        from { opacity: 0.2; }
        to { opacity: 0; }
    }

    .flash-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 9999;
        pointer-events: none;
        animation: fadeOut 2s forwards;
    }

    .flash-red {
        background-color: rgba(255, 0, 0, 0.2);
    }

    .flash-blue {
        background-color: rgba(0, 0, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Function to display the transient flash alert ---
def flash_alert(color="blue"):
    """Displays a transient, transparent full-screen flash alert."""
    placeholder = st.empty()
    if color == "red":
        css_class = "flash-red"
    else:
        css_class = "flash-blue"
    placeholder.markdown(f'<div class="{css_class} flash-overlay"></div>', unsafe_allow_html=True)

# --- Function to get AI recommendations from Gemini ---
def get_gemini_recommendations(gemini_model_instance, churn_risk_level, customer_details_str, offers_dict, company_name):
    """
    Connects to Google Gemini to get personalized recommendations, incorporating pre-listed offers.
    Args:
        gemini_model_instance: The initialized GenerativeModel object to use for generating content.
        churn_risk_level (str): "HIGH CHURN RISK" or "LOW CHURN RISK".
        customer_details_str (str): A string summarizing relevant customer details.
        offers_dict (dict): Dictionary of pre-listed offers.
        company_name (str): The name of the company.
    Returns:
        str: AI-generated recommendations.
    """
    if gemini_model_instance is None:
        st.warning("Gemini model is not initialized. Cannot generate AI recommendations.")
        return "AI recommendations are currently unavailable. Please check API key configuration."

    try:
        relevant_offers = offers_dict.get("high_churn", []) if churn_risk_level == "HIGH CHURN RISK" else offers_dict.get("low_churn", [])
        
        # Check for complaints using the customer_details_str (which now has readable labels)
        if "Complains: Yes" in customer_details_str:
            relevant_offers.extend(offers_dict.get("complaint_specific", []))

        offers_str = "\n".join([f"- {offer}" for offer in relevant_offers]) if relevant_offers else "No specific pre-listed offers to suggest at this time."

        system_prompt = f"""You are an expert AI assistant for customer service representatives (CSRs) at {company_name}.
Your goal is to provide specific, actionable, empathetic, and personalized recommendations to retain customers or build loyalty.
Focus on concrete actions or phrases the CSR can use, keeping recommendations concise as bullet points.
Explain reasoning based on customer details. Do not make up facts.
Prioritize relevant offers from the 'Available Offers' list.
"""

        user_prompt = f"""
Customer Churn Risk: {churn_risk_level}
Customer Details:
{customer_details_str}

Available {company_name} Offers:
{offers_str}

Provide 3-5 concise, actionable recommendations for the CSR, including relevant offers.
"""

        full_response_text = ""
        response_placeholder = st.empty() # Create a placeholder for streaming text
        
        with st.spinner(f"Getting AI-powered recommendations from {company_name}'s Gemini assistant... (This may take a moment)"):
            response = gemini_model_instance.generate_content(
                contents=[
                    {"role": "user", "parts": [system_prompt, user_prompt]}
                ],
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    full_response_text += chunk.text
                    response_placeholder.markdown(full_response_text + "▌") # Show progress with blinking cursor
            
            response_placeholder.markdown(full_response_text) # Show final response without cursor
        
        return full_response_text

    except Exception as e:
        st.error(f"An error occurred while generating Gemini recommendations: {e}")
        return "AI recommendations are currently unavailable due to an error."


st.title(f"Customer Churn Prediction App for {COMPANY_NAME} 📊")
st.markdown(f"""
    This application predicts customer churn based on various customer attributes.
    You can search for an existing customer to auto-fill their details, or enter new details manually.
    The prediction is made using an **ensemble of four machine learning models** to provide a more robust and insightful result.
    **Designed for Customer Service Representatives** at {COMPANY_NAME} to proactively identify and address churn risks.
    *(AI-powered recommendations are available via Google Gemini.)*
""")

# --- Global Assets Loading (Cached to run only once) ---
@st.cache_resource
def get_model_assets():
    """Loads all models and scaler, and returns the training columns."""
    models_dir = os.path.join(project_root, 'models')
    # --- MODIFIED LINE HERE ---
    data_path = os.path.join(project_root, 'data', 'customer_churn.csv')

    loaded_assets = load_all_models()

    try:
        df_full_for_cols = load_data(data_path)
        if df_full_for_cols is not None:
            if 'Churn' in df_full_for_cols.columns:
                X_full_for_cols = df_full_for_cols.drop('Churn', axis=1)
            else:
                X_full_for_cols = df_full_for_cols.copy()

            # Ensure all categorical columns are dummified consistently
            # It's crucial that preprocessing for training data and inference data align.
            # You might need to adjust 'preprocess_data' to handle columns consistently
            # or ensure this dummying reflects your full training pipeline.
            X_full_for_cols_encoded = pd.get_dummies(X_full_for_cols, drop_first=True)
            X_train_columns = X_full_for_cols_encoded.columns.tolist()
        else:
            st.error(f"Could not load data from {data_path} to infer training columns. Check data_loader.py and file path.")
            X_train_columns = []
    except Exception as e:
        st.error(f"Error loading data for column inference: {e}")
        X_train_columns = []

    loaded_assets['X_train_columns'] = X_train_columns
    return loaded_assets

# Load models and scaler
assets = get_model_assets()
scaler = assets.get('scaler')
xgb_model = assets.get('xgb_smote')
ann_class_weights_model = assets.get('ann_class_weights')
ann_smote_model = assets.get('ann_smote')
ann_focal_loss_model = assets.get('ann_focal_loss')
X_train_columns = assets.get('X_train_columns')

# Check if all critical assets are loaded
if not all([scaler, xgb_model, ann_class_weights_model, ann_smote_model, ann_focal_loss_model, X_train_columns]):
    st.error("One or more models/scaler/training columns could not be loaded. Please ensure models are trained and saved correctly.")
    st.stop()

# Define all models for the ensemble
ENSEMBLE_MODELS = {
    'XGBoost + SMOTE': {'model': xgb_model, 'type': 'xgb'},
    'ANN + Class Weights': {'model': ann_class_weights_model, 'type': 'ann'},
    'ANN + SMOTE': {'model': ann_smote_model, 'type': 'ann'},
    'ANN + Focal Loss': {'model': ann_focal_loss_model, 'type': 'ann'}
}


# --- Load Customer Data (for search functionality) ---
@st.cache_data
def load_customer_identifiers_data():
    customer_data_path = os.path.join(project_root, 'data', 'customer_data_with_identifiers.csv')
    try:
        df_customers = pd.read_csv(customer_data_path)
        # Ensure ID and Phone are treated as strings to avoid issues with leading zeros/format
        for col in ['CustomerID', 'PhoneNumber', 'CustomerName']:
            if col in df_customers.columns:
                df_customers[col] = df_customers[col].astype(str).fillna('')
        return df_customers
    except FileNotFoundError:
        st.warning(f"Customer identifier data not found at {customer_data_path}. Search functionality will be disabled.")
        return None
    except Exception as e:
        st.error(f"Error loading customer identifier data: {e}. Search functionality will be disabled.")
        return None

all_customers_df = load_customer_identifiers_data()

# --- Define Mappings for Selectboxes ---
COMPLAINS_OPTIONS = {"No": 0, "Yes": 1}
CHARGE_AMOUNT_OPTIONS = {
    "0 (No Charge/Undefined)": 0, "1 (Lowest Charge)": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9 (Highest Charge)": 9
}
AGE_GROUP_OPTIONS = {"1 (Youngest)": 1, "2": 2, "3": 3, "4": 4, "5 (Oldest)": 5}
TARIFF_PLAN_OPTIONS = {"Pay as you go": 1, "Contract": 2}
STATUS_OPTIONS = {"Active": 1, "Not Active": 2}

# --- Initialize Session State for Input Fields and Search ---
default_values = {
    'CallFailure': 5, 'Complains': 0, 'SubscriptionLength': 30, 'ChargeAmount': 1,
    'SecondsUse': 1500, 'FrequencyUse': 30, 'FrequencySMS': 20, 'DistinctCalls': 10,
    'AgeGroup': 2, 'TariffPlan': 1, 'Status': 1, 'Age': 25, 'CustomerValue': 250.0
}

# Initialize default values for input fields
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Initialize search-specific session state variables
if 'search_bar_input' not in st.session_state:
    st.session_state['search_bar_input'] = ''
if 'search_results_display' not in st.session_state:
    st.session_state['search_results_display'] = []
if 'search_results_data' not in st.session_state:
    st.session_state['search_results_data'] = pd.DataFrame()
if 'last_search_query' not in st.session_state:
    st.session_state['last_search_query'] = ""
if 'show_search_results' not in st.session_state:
    st.session_state['show_search_results'] = False
if 'performed_search_with_button' not in st.session_state:
    st.session_state['performed_search_with_button'] = False
if 'selected_customer_label' not in st.session_state:
    st.session_state['selected_customer_label'] = None

# --- Customer Search Section ---
st.header("Search for an Existing Customer 🔎")
if all_customers_df is not None:
    search_query_input = st.text_input(
        "Enter Customer Name, ID, or Phone Number to search:",
        value=st.session_state['search_bar_input'],
        key="search_bar",
        placeholder="e.g., Vinu, 100001, 9092777925",
    )
    st.session_state['search_bar_input'] = search_query_input

    search_button = st.button("Search Customers")

    # Trigger search if button is clicked or if query changes after a button search
    if search_button or (st.session_state['search_bar_input'] != st.session_state['last_search_query'] and st.session_state['performed_search_with_button']):
        st.session_state['performed_search_with_button'] = True
        st.session_state['last_search_query'] = search_query_input

        if search_query_input:
            with st.spinner(f"Searching for '{search_query_input}'..."):
                search_results = all_customers_df[
                    all_customers_df['CustomerName'].str.contains(search_query_input, case=False, na=False) |
                    all_customers_df['CustomerID'].str.contains(search_query_input, case=False, na=False) |
                    all_customers_df['PhoneNumber'].str.contains(search_query_input, case=False, na=False)
                ].copy()

            if not search_results.empty:
                display_options = []
                for idx, row in search_results.iterrows():
                    address_info = f" | Address: {row.get('Address', 'N/A')}" if 'Address' in row and pd.notna(row.get('Address')) else ""
                    display_options.append(f"ID: {row['CustomerID']} | Name: {row['CustomerName']} | Phone: {row['PhoneNumber']}{address_info}")
                
                st.session_state['search_results_display'] = display_options
                st.session_state['search_results_data'] = search_results
                st.session_state['show_search_results'] = True
                st.session_state['selected_customer_label'] = None # Clear previous selection on new search
                st.subheader(f"Found {len(search_results)} matching customers:")
            else:
                st.info(f"No customers found matching '{search_query_input}'.")
                st.session_state['search_results_display'] = []
                st.session_state['search_results_data'] = pd.DataFrame()
                st.session_state['show_search_results'] = False
                st.session_state['selected_customer_label'] = None
        else:
            st.session_state['search_results_display'] = []
            st.session_state['search_results_data'] = pd.DataFrame()
            st.session_state['show_search_results'] = False
            st.session_state['selected_customer_label'] = None
            st.info("Enter a query in the search bar and click 'Search' to find customers.")
    
    if st.session_state['show_search_results'] and st.session_state['search_results_display']:
        def on_customer_select():
            selected_option_label = st.session_state[f"customer_select_radio_{st.session_state['last_search_query']}"]
            st.session_state['selected_customer_label'] = selected_option_label
            
            # Extract CustomerID from the selected label
            selected_customer_id_from_label = selected_option_label.split(' | ')[0].replace('ID: ', '')
            
            # Find the full customer data based on the extracted ID
            selected_customer_row = st.session_state['search_results_data'][
                st.session_state['search_results_data']['CustomerID'] == selected_customer_id_from_label
            ].iloc[0]

            # Update session state with the selected customer's details
            for col in default_values.keys():
                if col in selected_customer_row:
                    # Convert to appropriate type if necessary (e.g., int for numerical inputs)
                    if isinstance(default_values[col], int):
                        st.session_state[col] = int(selected_customer_row[col])
                    elif isinstance(default_values[col], float):
                        st.session_state[col] = float(selected_customer_row[col])
                    else:
                        st.session_state[col] = selected_customer_row[col]
                else:
                    st.session_state[col] = default_values[col] # Fallback to default if column not found

            # Update dropdown/radio selections
            st.session_state['Complains_selected'] = [k for k, v in COMPLAINS_OPTIONS.items() if v == st.session_state['Complains']][0]
            st.session_state['ChargeAmount_selected'] = [k for k, v in CHARGE_AMOUNT_OPTIONS.items() if v == st.session_state['ChargeAmount']][0]
            st.session_state['AgeGroup_selected'] = [k for k, v in AGE_GROUP_OPTIONS.items() if v == st.session_state['AgeGroup']][0]
            st.session_state['TariffPlan_selected'] = [k for k, v in TARIFF_PLAN_OPTIONS.items() if v == st.session_state['TariffPlan']][0]
            st.session_state['Status_selected'] = [k for k, v in STATUS_OPTIONS.items() if v == st.session_state['Status']][0]
            
            st.success(f"Details for Customer ID: {selected_customer_id_from_label} loaded into input fields.")

        selected_option = st.radio(
            "Select a customer from the results:",
            st.session_state['search_results_display'],
            key=f"customer_select_radio_{st.session_state['last_search_query']}",
            on_change=on_customer_select
        )
        
        if st.session_state['selected_customer_label'] == selected_option and selected_option:
            st.write(f"Currently selected: **{selected_option}**")


elif all_customers_df is None:
    st.info("Customer search functionality is unavailable as 'customer_data_with_identifiers.csv' was not found.")

st.markdown("---")

# --- Customer Input Section ---
st.header("Enter Customer Details for Prediction 📝")

# Persistent state for form inputs
with st.form("customer_details_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Call & Usage Details")
        st.number_input("Call Failure (count)", min_value=0, max_value=50, value=st.session_state['CallFailure'], key='CallFailure_input')
        st.number_input("Subscription Length (months)", min_value=1, max_value=100, value=st.session_state['SubscriptionLength'], key='SubscriptionLength_input')
        st.number_input("Seconds Use (total)", min_value=0, max_value=10000, value=st.session_state['SecondsUse'], key='SecondsUse_input')
        st.number_input("Frequency Use (calls per month)", min_value=0, max_value=200, value=st.session_state['FrequencyUse'], key='FrequencyUse_input')

    with col2:
        st.subheader("Billing & Communication")
        st.selectbox("Complains", options=list(COMPLAINS_OPTIONS.keys()), format_func=lambda x: x,
                     index=list(COMPLAINS_OPTIONS.values()).index(st.session_state['Complains']), key='Complains_selected')
        st.selectbox("Charge Amount", options=list(CHARGE_AMOUNT_OPTIONS.keys()), format_func=lambda x: x,
                     index=list(CHARGE_AMOUNT_OPTIONS.values()).index(st.session_state['ChargeAmount']), key='ChargeAmount_selected')
        st.number_input("Frequency SMS (count)", min_value=0, max_value=500, value=st.session_state['FrequencySMS'], key='FrequencySMS_input')
        st.number_input("Distinct Calls (count)", min_value=0, max_value=100, value=st.session_state['DistinctCalls'], key='DistinctCalls_input')
    
    with col3:
        st.subheader("Demographics & Value")
        st.number_input("Age (years)", min_value=18, max_value=90, value=st.session_state['Age'], key='Age_input')
        st.selectbox("Age Group", options=list(AGE_GROUP_OPTIONS.keys()), format_func=lambda x: x,
                     index=list(AGE_GROUP_OPTIONS.values()).index(st.session_state['AgeGroup']), key='AgeGroup_selected')
        st.selectbox("Tariff Plan", options=list(TARIFF_PLAN_OPTIONS.keys()), format_func=lambda x: x,
                     index=list(TARIFF_PLAN_OPTIONS.values()).index(st.session_state['TariffPlan']), key='TariffPlan_selected')
        st.selectbox("Status", options=list(STATUS_OPTIONS.keys()), format_func=lambda x: x,
                     index=list(STATUS_OPTIONS.values()).index(st.session_state['Status']), key='Status_selected')
        st.number_input("Customer Value (£)", min_value=0.0, max_value=5000.0, value=st.session_state['CustomerValue'], step=10.0, key='CustomerValue_input')

    predict_button = st.form_submit_button("Predict Churn")

    if predict_button:
        # Update session state values from form inputs
        st.session_state['CallFailure'] = st.session_state['CallFailure_input']
        st.session_state['SubscriptionLength'] = st.session_state['SubscriptionLength_input']
        st.session_state['SecondsUse'] = st.session_state['SecondsUse_input']
        st.session_state['FrequencyUse'] = st.session_state['FrequencyUse_input']
        st.session_state['Complains'] = COMPLAINS_OPTIONS[st.session_state['Complains_selected']]
        st.session_state['ChargeAmount'] = CHARGE_AMOUNT_OPTIONS[st.session_state['ChargeAmount_selected']]
        st.session_state['FrequencySMS'] = st.session_state['FrequencySMS_input']
        st.session_state['DistinctCalls'] = st.session_state['DistinctCalls_input']
        st.session_state['Age'] = st.session_state['Age_input']
        st.session_state['AgeGroup'] = AGE_GROUP_OPTIONS[st.session_state['AgeGroup_selected']]
        st.session_state['TariffPlan'] = TARIFF_PLAN_OPTIONS[st.session_state['TariffPlan_selected']]
        st.session_state['Status'] = STATUS_OPTIONS[st.session_state['Status_selected']]
        st.session_state['CustomerValue'] = st.session_state['CustomerValue_input']

        customer_data = {
            'CallFailure': st.session_state['CallFailure'],
            'Complains': st.session_state['Complains'],
            'SubscriptionLength': st.session_state['SubscriptionLength'],
            'ChargeAmount': st.session_state['ChargeAmount'],
            'SecondsUse': st.session_state['SecondsUse'],
            'FrequencyUse': st.session_state['FrequencyUse'],
            'FrequencySMS': st.session_state['FrequencySMS'],
            'DistinctCalls': st.session_state['DistinctCalls'],
            'AgeGroup': st.session_state['AgeGroup'],
            'TariffPlan': st.session_state['TariffPlan'],
            'Status': st.session_state['Status'],
            'Age': st.session_state['Age'],
            'CustomerValue': st.session_state['CustomerValue']
        }
        df_single_customer = pd.DataFrame([customer_data])

        st.markdown("---")
        st.subheader("Prediction Results 📈")

        # --- Generate Customer Details String for LLM ---
        customer_details_for_llm = f"""
        - Call Failure: {st.session_state['CallFailure']}
        - Complains: {'Yes' if st.session_state['Complains'] == 1 else 'No'}
        - Subscription Length: {st.session_state['SubscriptionLength']} months
        - Charge Amount: {st.session_state['ChargeAmount']} (on a scale of 0-9)
        - Seconds Use: {st.session_state['SecondsUse']} seconds
        - Frequency Use: {st.session_state['FrequencyUse']} calls/month
        - Frequency SMS: {st.session_state['FrequencySMS']} SMS/month
        - Distinct Calls: {st.session_state['DistinctCalls']} unique numbers
        - Age Group: {st.session_state['AgeGroup']} (1-5, 5 being oldest)
        - Tariff Plan: {'Pay as you go' if st.session_state['TariffPlan'] == 1 else 'Contract'}
        - Status: {'Active' if st.session_state['Status'] == 1 else 'Not Active'}
        - Age: {st.session_state['Age']} years
        - Customer Value: £{st.session_state['CustomerValue']:.2f}
        """
        
        # Clear Keras session before making predictions to prevent potential resource conflicts
        # This is particularly helpful when using TensorFlow/Keras models repeatedly in a Streamlit app.
        K.clear_session()
        st.info("Keras backend session cleared for stable predictions.")

        # Perform predictions with all models
        ensemble_predictions = []
        model_prob_data = [] # To store probabilities for charting
        model_results_markdown = "##### Individual Model Predictions:\n"

        for model_name, model_info in ENSEMBLE_MODELS.items():
            model_instance = model_info['model']
            model_type = model_info['type']
            
            with st.spinner(f"Predicting with {model_name}..."):
                preds, probs, _ = predict_churn(
                    model_instance, 
                    df_single_customer, 
                    scaler, 
                    X_train_columns, 
                    model_type=model_type, 
                    gemini_model=None # Don't generate recommendations for individual models here
                )
                
                prediction_text = "CHURN" if preds == 1 else "NO CHURN"
                ensemble_predictions.append(preds)
                model_prob_data.append({'Model': model_name, 'Probability': probs, 'Prediction': prediction_text})# Store for chart
                model_results_markdown += f"- **{model_name}**: {prediction_text} (Probability: {probs:.2f})\n"

        st.markdown(model_results_markdown)
        st.markdown("---")

        # --- Ensemble Voting ---
        churn_votes = sum(ensemble_predictions) # Count how many models predicted churn
        no_churn_votes = len(ENSEMBLE_MODELS) - churn_votes

        st.subheader("Ensemble Consensus:")
        # Modified logic: If any model predicts churn, the ensemble predicts churn
        if churn_votes > 0: 
            churn_risk_level = "HIGH CHURN RISK"
            st.error(f"⚠️ **HIGH RISK: The ensemble predicts the customer will CHURN** (based on {churn_votes} out of {len(ENSEMBLE_MODELS)} models predicting churn).")
            flash_alert(color="red") # Display transient red flash
        else:
            churn_risk_level = "LOW CHURN RISK"
            st.success(f"✅ **LOW RISK: The ensemble predicts the customer will NOT CHURN** (based on no models predicting churn).")
            flash_alert(color="blue") # Display transient blue flash

        st.markdown("---")
        
        # --- Churn Probability by Model Graph ---
        st.subheader("Individual Model Churn Probabilities")
        df_probs = pd.DataFrame(model_prob_data)
        fig_probs = px.bar(df_probs, x='Model', y='Probability',
                           title='Churn Probability by Model',
                           labels={'Probability': 'Churn Probability'},
                           color='Probability',
                           color_continuous_scale=px.colors.sequential.Plasma,
                           range_y=[0, 1],
                           text_auto='.2f',
                           height=400
                           )
        fig_probs.update_layout(xaxis_title="Machine Learning Model", yaxis_title="Predicted Churn Probability")
        st.plotly_chart(fig_probs, use_container_width=True)

        # --- Ensemble Vote Distribution Graph ---
        st.subheader("Ensemble Vote Distribution")
        
        vote_data = pd.DataFrame({
            'Vote': ['Models Predicting Churn', 'Models Predicting No Churn'],
            'Count': [churn_votes, no_churn_votes]
        })
        fig_votes = px.bar(vote_data, x='Vote', y='Count',
                           title='Ensemble Model Votes',
                           labels={'Count': 'Number of Models'},
                           color='Vote',
                           color_discrete_map={'Models Predicting Churn': '#D62728', 'Models Predicting No Churn': '#1F77B4'},
                           range_y=[0, len(ENSEMBLE_MODELS)],
                           text_auto=True,
                           height=400
                           )
        fig_votes.update_layout(xaxis_title="Ensemble Vote Outcome", yaxis_title="Number of Models")
        st.plotly_chart(fig_votes, use_container_width=True)


        st.markdown("---")
        st.subheader("AI-Powered Recommendations from Google Gemini")

        ai_recommendations = get_gemini_recommendations(
            gemini_model,
            churn_risk_level,
            customer_details_for_llm,
            PRE_LISTED_OFFERS,
            COMPANY_NAME
        )
        
        if ai_recommendations:
            st.markdown(ai_recommendations)
            st_copy_to_clipboard(ai_recommendations)
        else:
            st.info("No AI recommendations generated (Gemini might be unavailable or errored).")

st.markdown("---")
st.markdown(f"Developed by Vinu & UK for {COMPANY_NAME} | Version 1.0")