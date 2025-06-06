import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import time
import google.generativeai as genai
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
    
    genai.configure(api_key=GEMINI_API_KEY) 
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
            response = gemini_model_instance.generate_content( # <--- CORRECTED: Using the passed instance
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
    data_path = os.path.join(project_root, 'data', 'customer_churn.xlsx')

    loaded_assets = load_all_models(models_dir)

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

            selected_customer_row_filtered = st.session_state['search_results_data'][
                st.session_state['search_results_data']['CustomerID'] == selected_customer_id_from_label
            ]

            if not selected_customer_row_filtered.empty:
                selected_customer_row = selected_customer_row_filtered.iloc[0]

                mapping_dicts = {
                    'Complains': COMPLAINS_OPTIONS,
                    'ChargeAmount': CHARGE_AMOUNT_OPTIONS,
                    'AgeGroup': AGE_GROUP_OPTIONS,
                    'TariffPlan': TARIFF_PLAN_OPTIONS,
                    'Status': STATUS_OPTIONS
                }

                for feature, default_val in default_values.items():
                    if feature in selected_customer_row and pd.notna(selected_customer_row[feature]):
                        if feature in mapping_dicts:
                            value_from_data = selected_customer_row[feature]
                            found_label = None
                            for key, val in mapping_dicts[feature].items():
                                if val == value_from_data:
                                    found_label = key
                                    break
                            
                            if found_label:
                                # For selectboxes, we want to store the *value* in session_state,
                                # and the selectbox itself will find its label based on that value.
                                st.session_state[feature] = mapping_dicts[feature][found_label]
                            else:
                                st.warning(f"Value '{value_from_data}' for '{feature}' not found in options. Falling back to default.")
                                st.session_state[feature] = default_val
                        else: # For numerical inputs directly use the value
                            try:
                                target_type = type(default_val)
                                st.session_state[feature] = target_type(selected_customer_row[feature])
                            except (ValueError, TypeError):
                                st.warning(f"Could not convert value '{selected_customer_row[feature]}' for '{feature}'. Falling back to default.")
                                st.session_state[feature] = default_val
                    else: # If feature not in data or is NaN, fall back to default
                        st.session_state[feature] = default_val

                st.success(f"Details for Customer ID {selected_customer_row.get('CustomerID', 'N/A')} ({selected_customer_row.get('CustomerName', 'N/A')}) loaded!")
                st.rerun() # Rerun to update input fields with new session state values
            else:
                st.error("Selected customer data not found in search results. Please try again.")

        default_index = None
        if st.session_state['selected_customer_label'] in st.session_state['search_results_display']:
            default_index = st.session_state['search_results_display'].index(st.session_state['selected_customer_label'])

        st.radio(
            "Select a customer to auto-fill details:",
            st.session_state['search_results_display'],
            index=default_index,
            key=f"customer_select_radio_{st.session_state['last_search_query']}",
            on_change=on_customer_select
        )
    elif st.session_state['performed_search_with_button'] and not st.session_state['search_results_display']:
        st.info(f"No customers found for your last search: '{st.session_state['last_search_query']}'.")
    else:
        st.info("Enter a query in the search bar and click 'Search' to find customers.")

else:
    st.warning("Customer data not available for search. Please ensure 'customer_data_with_identifiers.csv' is in your `/data/` folder.")


# --- Define Mappings for Selectboxes ---
# Used for getting the index for selectbox
def get_selectbox_index(option_dict, current_value, default_value):
    """Helper to find the index for st.selectbox."""
    try:
        # Find the key (label) corresponding to the current value
        current_label = next(key for key, val in option_dict.items() if val == current_value)
        return list(option_dict.keys()).index(current_label)
    except StopIteration:
        # If the current_value from session_state is not in the options, fall back to default's index
        st.warning(f"Value '{current_value}' not found in options for selectbox. Resetting to default.")
        # Find the label for the default value to set the index
        default_label = next(key for key, val in option_dict.items() if val == default_value)
        return list(option_dict.keys()).index(default_label)


# --- Customer Details for Prediction ---
st.header("Customer Details for Prediction 📝")
st.markdown("**(Auto-filled if customer searched, or enter manually)**")

col1, col2, col3 = st.columns(3)

with col1:
    st.number_input(
        "Call Failure (Number of failed calls)",
        min_value=0, max_value=50,
        value=st.session_state['CallFailure'],
        key="CallFailure"
    )
    st.number_input(
        "Subscription Length (months)",
        min_value=0, max_value=100,
        value=st.session_state['SubscriptionLength'],
        key="SubscriptionLength"
    )
    st.number_input(
        "Seconds Use (total seconds in past year)",
        min_value=0, max_value=20000,
        value=st.session_state['SecondsUse'],
        key="SecondsUse"
    )
    st.number_input(
        "Distinct Calls (number of distinct numbers called in past year)",
        min_value=0, max_value=100,
        value=st.session_state['DistinctCalls'],
        key="DistinctCalls"
    )

    selected_tariff_label = st.selectbox(
        "Tariff Plan",
        options=list(TARIFF_PLAN_OPTIONS.keys()),
        index=get_selectbox_index(TARIFF_PLAN_OPTIONS, st.session_state['TariffPlan'], default_values['TariffPlan']),
        key="TariffPlan_selectbox"
    )
    # Update session state with the numerical value from the selected label
    st.session_state['TariffPlan'] = TARIFF_PLAN_OPTIONS[selected_tariff_label]


with col2:
    selected_complains_label = st.selectbox(
        "Complains (Has the customer filed a complaint?)",
        options=list(COMPLAINS_OPTIONS.keys()),
        index=get_selectbox_index(COMPLAINS_OPTIONS, st.session_state['Complains'], default_values['Complains']),
        key="Complains_selectbox"
    )
    st.session_state['Complains'] = COMPLAINS_OPTIONS[selected_complains_label]

    selected_charge_label = st.selectbox(
        "Charge Amount (Categorical)",
        options=list(CHARGE_AMOUNT_OPTIONS.keys()),
        index=get_selectbox_index(CHARGE_AMOUNT_OPTIONS, st.session_state['ChargeAmount'], default_values['ChargeAmount']),
        key="ChargeAmount_selectbox"
    )
    st.session_state['ChargeAmount'] = CHARGE_AMOUNT_OPTIONS[selected_charge_label]

    st.number_input(
        "Frequency Use (total calls in past year)",
        min_value=0, max_value=200,
        value=st.session_state['FrequencyUse'],
        key="FrequencyUse"
    )

    selected_age_group_label = st.selectbox(
        "Age Group Category",
        options=list(AGE_GROUP_OPTIONS.keys()),
        index=get_selectbox_index(AGE_GROUP_OPTIONS, st.session_state['AgeGroup'], default_values['AgeGroup']),
        key="AgeGroup_selectbox"
    )
    st.session_state['AgeGroup'] = AGE_GROUP_OPTIONS[selected_age_group_label]

    st.number_input(
        "Age (Customer age)",
        min_value=15, max_value=90,
        value=st.session_state['Age'],
        key="Age"
    )


with col3:
    st.number_input(
        "Frequency SMS (total SMS in past year)",
        min_value=0, max_value=500,
        value=st.session_state['FrequencySMS'],
        key="FrequencySMS"
    )

    selected_status_label = st.selectbox(
        "Status",
        options=list(STATUS_OPTIONS.keys()),
        index=get_selectbox_index(STATUS_OPTIONS, st.session_state['Status'], default_values['Status']),
        key="Status_selectbox"
    )
    st.session_state['Status'] = STATUS_OPTIONS[selected_status_label]

    st.number_input(
        "Customer Value (projected for next year)",
        min_value=0.0, max_value=5000.0,
        format="%.2f",
        value=st.session_state['CustomerValue'],
        key="CustomerValue"
    )

# Collect inputs into a DataFrame from session state
input_data = pd.DataFrame({k: [st.session_state[k]] for k in default_values.keys()})


st.subheader("Ensemble Prediction Results 📈")

if st.button("Predict Churn with Ensemble"):
    if not all([m['model'] for m in ENSEMBLE_MODELS.values()]):
        st.error("One or more ensemble models could not be loaded. Cannot make prediction. Please check your model files.")
    else:
        with st.spinner("Making predictions with the ensemble models..."):
            individual_predictions = []
            individual_probabilities = []

            for model_name, model_info in ENSEMBLE_MODELS.items():
                model = model_info['model']
                model_type = model_info['type']

                # Capture all return values, even if recommendations is None
                prediction, probability, _ = predict_churn(
                    model,
                    input_data,
                    scaler,
                    X_train_columns,
                    model_type
                )
                individual_predictions.append(prediction[0])
                individual_probabilities.append(probability[0])

            churn_votes = sum(individual_predictions)
            no_churn_votes = len(individual_predictions) - churn_votes

            # --- Visualizations ---
            vis_col1, vis_col2 = st.columns(2)

            with vis_col1:
                st.markdown("##### Individual Model Churn Probabilities")
                fig, ax = plt.subplots(figsize=(8, 5))
                probabilities_df = pd.DataFrame({
                    'Model': list(ENSEMBLE_MODELS.keys()),
                    'Probability': individual_probabilities
                })
                sns.barplot(x='Probability', y='Model', data=probabilities_df, palette='viridis', ax=ax)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Churn Probability")
                ax.set_title("Individual Model Churn Probabilities")

                for index, row in probabilities_df.iterrows():
                    ax.text(row.Probability + 0.02, index, f'{row.Probability:.1%}', color='white', ha="left", va="center", fontsize=10)

                fig.patch.set_facecolor('#1E2130')
                ax.set_facecolor('#1E2130')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.yaxis.label.set_color('white')
                ax.xaxis.label.set_color('white')
                ax.title.set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                st.pyplot(fig)
                plt.close(fig)

            with vis_col2:
                st.markdown("##### Ensemble Churn Likelihood")
                ensemble_avg_prob = sum(individual_probabilities) / len(individual_probabilities)

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ensemble_avg_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Ensemble Churn Probability", 'font': {'size': 20, 'color': 'white'}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#FF4B4B"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 20], 'color': 'lightgreen'},
                            {'range': [20, 50], 'color': 'yellow'},
                            {'range': [50, 100], 'color': 'red'}],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': ensemble_avg_prob * 100}
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10),
                                         paper_bgcolor="#1E2130",
                                         font = {'color': "white", 'family': "Arial"})
                st.plotly_chart(fig_gauge, use_container_width=True)


            st.subheader("Ensemble Decision (Majority Vote)")
            
            # Prepare customer details for the LLM
            customer_details_for_llm = ""
            # Iterate through the input_data (which is a DataFrame with one row)
            # Use .iloc[0].to_dict() to get a dictionary of the input features
            for k, v in input_data.iloc[0].to_dict().items():
                if k == 'Complains':
                    customer_details_for_llm += f"- Complains: {('No' if v == 0 else 'Yes')}\n"
                elif k == 'ChargeAmount':
                    label = next((label for label, val in CHARGE_AMOUNT_OPTIONS.items() if val == v), str(v))
                    customer_details_for_llm += f"- Charge Amount: {label}\n"
                elif k == 'AgeGroup':
                    label = next((label for label, val in AGE_GROUP_OPTIONS.items() if val == v), str(v))
                    customer_details_for_llm += f"- Age Group: {label}\n"
                elif k == 'TariffPlan':
                    label = next((label for label, val in TARIFF_PLAN_OPTIONS.items() if val == v), str(v))
                    customer_details_for_llm += f"- Tariff Plan: {label}\n"
                elif k == 'Status':
                    label = next((label for label, val in STATUS_OPTIONS.items() if val == v), str(v))
                    customer_details_for_llm += f"- Status: {label}\n"
                else:
                    customer_details_for_llm += f"- {k}: {v}\n"

            if churn_votes >= no_churn_votes:
                churn_risk_level = "HIGH CHURN RISK"
                st.error(f"⚠️ **HIGH RISK: The ensemble predicts the customer will CHURN** (based on {churn_votes} out of {len(ENSEMBLE_MODELS)} models).")
                flash_alert(color="red") # Display transient red flash
            else:
                churn_risk_level = "LOW CHURN RISK"
                st.success(f"✅ **LOW RISK: The ensemble predicts the customer will NOT CHURN** (based on {no_churn_votes} out of {len(ENSEMBLE_MODELS)} models).")
                flash_alert(color="blue") # Display transient blue flash

            st.markdown("---")
            st.subheader("AI-Powered Recommendations from Google Gemini")

            # CORRECTED CALL: Pass the gemini_model object
            ai_recommendations = get_gemini_recommendations(
                gemini_model, # This is the key fix
                churn_risk_level,
                customer_details_for_llm,
                PRE_LISTED_OFFERS,
                COMPANY_NAME
            )
            
            if ai_recommendations:
                st.markdown(ai_recommendations)
                # You might need to uncomment and ensure st_copy_to_clipboard is working
                st_copy_to_clipboard(ai_recommendations)
            else:
                st.info("No AI recommendations generated (Gemini might be unavailable or errored).")

st.markdown("---")
st.markdown(f"Developed with ❤️ using Streamlit for {COMPANY_NAME}")