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

# --- Add src directory to Python path ---
# This ensures that modules like model_predictor and preprocessor can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir # In this structure, app.py is at the project root
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"Added {src_path} to sys.path") # For debugging

# Import your modules after adding to path
try:
    import model_predictor as mp
    from data_loader import load_data, load_customer_identifiers_data # Import new function
    from preprocessor import preprocess_data
    st.write("All custom modules imported successfully.")
except Exception as e:
    st.error(f"Error importing custom modules: {e}")
    st.stop() # Stop the app if crucial imports fail


# --- Constants ---
# Assuming 'models' and 'data' are directly under project_root
MODELS_DIR = os.path.join(project_root, 'models')
DATA_DIR = os.path.join(project_root, 'data')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# --- Load Data for Columns (Required for Preprocessing) ---
# This part is crucial to get the correct column order for prediction preprocessing
DATA_FOR_COLUMNS_FILENAME = 'customer_churn.csv' # Corrected filename
DATA_FOR_COLUMNS_PATH = os.path.join(project_root, 'data', DATA_FOR_COLUMNS_FILENAME) # Corrected path
st.write(f"Attempting to load data for columns from: {DATA_FOR_COLUMNS_PATH}")

X_train_columns = None
if os.path.exists(DATA_FOR_COLUMNS_PATH):
    try:
        # Load the raw data to infer original columns before one-hot encoding
        initial_df_for_cols = pd.read_csv(DATA_FOR_COLUMNS_PATH, encoding='latin1') # Use pd.read_csv for CSV
        
        # Infer X_train_columns from this data
        # Ensure categorical columns are correctly handled for column inference
        categorical_cols_for_inference = initial_df_for_cols.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Drop the target column 'Churn' if it exists in this data
        X_for_cols_inference = initial_df_for_cols.drop(columns=['Churn'], errors='ignore')
        
        # Apply dummy encoding to get the final training columns structure
        X_train_columns = pd.get_dummies(X_for_cols_inference, columns=categorical_cols_for_inference, drop_first=True).columns.tolist()
        
        st.write(f"Data file for columns exists: {os.path.exists(DATA_FOR_COLUMNS_PATH)}")
        st.write("Successfully inferred training columns.")
    except Exception as e:
        st.error(f"An error occurred during data loading for columns: {e}. Please check your data file and path.")
        st.error("One or more models/scaler/training columns could not be loaded. Please ensure models are trained and saved correctly and are accessible in the deployed environment.")
        st.stop() # Stop if data loading fails
else:
    st.error(f"Data file for columns does not exist: {DATA_FOR_COLUMNS_PATH}. Please ensure your data file is in the 'data' directory.")
    st.error("One or more models/scaler/training columns could not be loaded. Please ensure models are trained and saved correctly and are accessible in the deployed environment.")
    st.stop() # Stop if data file is not found


# --- Load Models and Scaler ---
st.write(f"Attempting to load models from: {MODELS_DIR}")
loaded_models_and_scaler = {} # Initialize in case of error

if os.path.exists(MODELS_DIR):
    try:
        # Pass MODELS_DIR to load_all_models
        loaded_models_and_scaler = mp.load_all_models(MODELS_DIR)
        
        # Check if all expected models and scaler are loaded
        required_assets = ['scaler', 'xgb_smote', 'ann_class_weights', 'ann_smote', 'ann_focal_loss'] # Corrected
        
        all_assets_loaded = True
        for asset_key in required_assets:
            if loaded_models_and_scaler.get(asset_key) is None:
                st.error(f"Failed to load {asset_key}.")
                all_assets_loaded = False
        
        if all_assets_loaded:
            st.success("All models and scaler loaded successfully!")
        else:
            st.error("One or more models/scaler/training columns could not be loaded. Please ensure models are trained and saved correctly and are accessible in the deployed environment.")
            st.stop() # Stop if not all assets are loaded

    except Exception as e:
        st.error(f"An error occurred during asset loading: {e}. Please check your model files and data paths.")
        st.error("One or more models/scaler/training columns could not be loaded. Please ensure models are trained and saved correctly and are accessible in the deployed environment.")
        st.stop() # Stop if asset loading fails
else:
    st.error(f"Models directory does not exist: {MODELS_DIR}. Please ensure your 'models' directory is at the project root.")
    st.error("One or more models/scaler/training columns could not be loaded. Please ensure models are trained and saved correctly and are accessible in the deployed environment.")
    st.stop() # Stop if models directory not found


# --- Google Gemini API Configuration ---
# Ensure the API key is securely accessed via Streamlit secrets
gemini_api_key = st.secrets.get("GEMINI_API_KEY")
gemini_model = None
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        st.success("Connected to Google Gemini API.")
    except Exception as e:
        st.warning(f"Could not connect to Google Gemini API. AI recommendations will be unavailable. Error: {e}")
else:
    st.warning("Google Gemini API key not found in Streamlit secrets. AI recommendations will be unavailable.")


# --- Global Variables for Models and Scaler ---
SCALER = loaded_models_and_scaler.get('scaler')
ENSEMBLE_MODELS = {
    'xgb_smote': loaded_models_and_scaler.get('xgb_smote'),
    'ann_class_weights': loaded_models_and_scaler.get('ann_class_weights'),
    'ann_smote': loaded_models_and_scaler.get('ann_smote'),
    'ann_focal_loss': loaded_models_and_scaler.get('ann_focal_loss')
}

PRE_LISTED_OFFERS = [
    "Offer a 10% discount on their next 3 months' bill.",
    "Upgrade their data plan by 2GB for free for 6 months.",
    "Provide a complimentary premium customer support access for 3 months.",
    "Offer a free add-on service (e.g., international calls package) for 1 month.",
    "Suggest a loyalty bonus program where they earn points for continued subscription."
]


# --- UI Elements ---
st.title(f"Customer Churn Prediction Dashboard for {COMPANY_NAME}")


# --- Load Customer Identifiers Data ---
CUSTOMER_IDENTIFIERS_FILENAME = 'customer_data_with_identifiers.csv'
CUSTOMER_IDENTIFIERS_PATH = os.path.join(project_root, 'data', CUSTOMER_IDENTIFIERS_FILENAME)

all_customers_df = None
if os.path.exists(CUSTOMER_IDENTIFIERS_PATH):
    try:
        all_customers_df = load_customer_identifiers_data(CUSTOMER_IDENTIFIERS_PATH)
        if all_customers_df is not None and not all_customers_df.empty:
            st.sidebar.success(f"Loaded {len(all_customers_df)} existing customer records.")
        else:
            st.sidebar.warning("Could not load existing customer records or file is empty for selection.")
            all_customers_df = None # Ensure it's None if empty
    except Exception as e:
        st.sidebar.error(f"Error loading existing customer data: {e}")
        all_customers_df = None
else:
    st.sidebar.warning(f"Existing customer data file not found at: {CUSTOMER_IDENTIFIERS_PATH}. Customer selection option will be unavailable.")


st.sidebar.header("Existing Customer Selection")
selected_customer_data = None
if all_customers_df is not None:
    customer_display_names = ['Select an existing customer...'] + all_customers_df['CustomerName'].tolist()
    selected_name = st.sidebar.selectbox("Choose a customer:", customer_display_names)

    if selected_name != 'Select an existing customer...':
        selected_customer_data = all_customers_df[all_customers_df['CustomerName'] == selected_name].iloc[0]
        st.sidebar.success(f"Loaded data for: {selected_name}")
else:
    st.sidebar.info("No existing customer data available for selection.")


st.sidebar.header("Customer Details (or manually enter below)")

# Initialize input_data with default values, which can be overridden by selected customer
input_data = {}

# Define default values for each input based on common sense or typical ranges
# These defaults will be used if no customer is selected or if a field is missing from selected customer data
default_values = {
    'CallFailure': 0, 'Complains': 0, 'SubscriptionLength': 12,
    'ChargeAmount': 50, 'SecondsUse': 1000, 'FrequencyUse': 20,
    'FrequencySMS': 10, 'DistinctCalls': 5, 'AgeGroup': 'Group2', # Default string for categorical
    'TariffPlan': 'PlanA', 'Status': 'Active', 'Age': 30, 'CustomerValue': 500.0
}

# Populate input_data based on selected customer or defaults
for key, default_val in default_values.items():
    if selected_customer_data is not None and key in selected_customer_data:
        # Use .item() for single-element numpy types to convert to Python scalars if necessary
        # Handle cases where value might be NaN or not a basic type
        val = selected_customer_data[key]
        if pd.isna(val): # Check for NaN
            input_data[key] = default_val
        elif hasattr(val, 'item'): # For numpy scalars
            input_data[key] = val.item()
        else:
            input_data[key] = val
    else:
        input_data[key] = default_val


# Adjust categorical options for app.py
# IMPORTANT: Ensure these options match the unique values in your actual training data's categorical columns
age_group_options = ['Group1', 'Group2', 'Group3', 'Group4', 'Group5'] # Example options
tariff_plan_options = ['PlanA', 'PlanB'] # Example options
status_options = ['Active', 'Inactive'] # Example options


# Streamlit Input fields, now initialized with selected customer data or defaults
input_data['CallFailure'] = st.sidebar.number_input("Number of Call Failures", min_value=0, value=input_data['CallFailure'], step=1)
input_data['Complains'] = st.sidebar.selectbox("Has Complained?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=input_data['Complains'])
input_data['SubscriptionLength'] = st.sidebar.slider("Subscription Length (months)", min_value=1, max_value=72, value=input_data['SubscriptionLength'])
input_data['ChargeAmount'] = st.sidebar.number_input("Charge Amount (per month)", min_value=0, value=input_data['ChargeAmount'], step=1)
input_data['SecondsUse'] = st.sidebar.number_input("Total Seconds Used (last month)", min_value=0, value=input_data['SecondsUse'], step=100)
input_data['FrequencyUse'] = st.sidebar.number_input("Frequency of Use (calls/SMS, last month)", min_value=0, value=input_data['FrequencyUse'], step=1)
input_data['FrequencySMS'] = st.sidebar.number_input("Frequency of SMS (last month)", min_value=0, value=input_data['FrequencySMS'], step=1)
input_data['DistinctCalls'] = st.sidebar.number_input("Number of Distinct Calls (last month)", min_value=0, value=input_data['DistinctCalls'], step=1)

# For selectbox, find the index of the default/selected value
age_group_index = age_group_options.index(input_data['AgeGroup']) if input_data['AgeGroup'] in age_group_options else 0
tariff_plan_index = tariff_plan_options.index(input_data['TariffPlan']) if input_data['TariffPlan'] in tariff_plan_options else 0
status_index = status_options.index(input_data['Status']) if input_data['Status'] in status_options else 0

input_data['AgeGroup'] = st.sidebar.selectbox("Age Group", options=age_group_options, index=age_group_index)
input_data['TariffPlan'] = st.sidebar.selectbox("Tariff Plan", options=tariff_plan_options, index=tariff_plan_index)
input_data['Status'] = st.sidebar.selectbox("Status", options=status_options, index=status_index)
input_data['Age'] = st.sidebar.slider("Age", min_value=18, max_value=99, value=input_data['Age'])
input_data['CustomerValue'] = st.sidebar.number_input("Customer Value (USD)", min_value=0.0, value=input_data['CustomerValue'], step=10.0)

# Convert input data to DataFrame
customer_df = pd.DataFrame([input_data])

# Button to trigger prediction
if st.sidebar.button("Predict Churn"):
    if SCALER is None or X_train_columns is None:
        st.error("Application not fully loaded. Please check logs for missing scaler or training columns. Cannot predict.")
    else:
        st.subheader("Prediction Results")
        
        # Prepare customer details for LLM (before preprocessing for model)
        customer_details_for_llm = customer_df.iloc[0].to_dict()

        # Get predictions from each model
        predictions = {}
        probabilities = {}

        # Use st.spinner to show loading
        with st.spinner("Calculating predictions..."):
            for model_name, model in ENSEMBLE_MODELS.items():
                if model is not None:
                    try:
                        model_type = 'xgb' if 'xgb' in model_name else 'ann'
                        # predict_churn now returns only prediction and probability
                        prediction, probability = mp.predict_churn(model, customer_df.copy(), SCALER, X_train_columns, model_type=model_type)
                        
                        predictions[model_name] = prediction
                        probabilities[model_name] = probability
                    except Exception as e:
                        st.error(f"Error predicting with {model_name}: {e}")
                        predictions[model_name] = 0 # Assume no churn on error
                        probabilities[model_name] = 0.0
                else:
                    st.warning(f"{model_name} not loaded, skipping prediction.")
                    predictions[model_name] = 0
                    probabilities[model_name] = 0.0

        # Ensemble Logic: Majority Vote
        churn_votes = sum(1 for p in predictions.values() if p == 1)
        no_churn_votes = sum(1 for p in predictions.values() if p == 0)

        total_models = len(ENSEMBLE_MODELS)
        if total_models > 0:
            if churn_votes > no_churn_votes:
                ensemble_prediction = 1 # Churn
                st.error(f"**Ensemble Prediction: HIGH CHURN RISK**")
                churn_risk_level = "HIGH CHURN RISK"
            else:
                ensemble_prediction = 0 # No Churn
                st.success(f"**Ensemble Prediction: LOW CHURN RISK**")
                churn_risk_level = "LOW CHURN RISK"
            
            # Display individual model probabilities and ensemble vote visualization in columns
            st.markdown("---")
            col1, col2 = st.columns(2) # Create two columns

            with col1:
                st.subheader("Individual Model Probabilities")
                # Create a simple table-like string for individual probabilities as requested
                prob_str = "<ul>"
                for model_name, prob in probabilities.items():
                    prob_str += f"<li><b>{model_name}:</b> {prob:.4f}</li>"
                prob_str += "</ul>"
                st.markdown(prob_str)
                # Removed st.dataframe(prob_df) as requested

            with col2:
                st.subheader("Ensemble Model Votes")
                vote_data = pd.DataFrame({
                    'Vote': ['Models Predicting Churn', 'Models Predicting No Churn'],
                    'Count': [churn_votes, no_churn_votes]
                })
                fig_votes = px.bar(vote_data, x='Vote', y='Count',
                                   title='Ensemble Model Votes', # Title remains in the graph
                                   labels={'Count': 'Number of Models'},
                                   color='Vote',
                                   color_discrete_map={'Models Predicting Churn': '#D62728', 'Models Predicting No Churn': '#1F77B4'},
                                   range_y=[0, len(ENSEMBLE_MODELS)],
                                   text_auto=True,
                                   height=400
                                   )
                fig_votes.update_layout(xaxis_title="Ensemble Vote Outcome", yaxis_title="Number of Models")
                st.plotly_chart(fig_votes, use_container_width=True) # use_container_width will make it fit the column


            st.markdown("---")
            st.subheader("AI-Powered Recommendations from Google Gemini")

            ai_recommendations = mp.get_gemini_recommendations( # Call from mp module
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
                st.info("No AI recommendations generated (Gemini might be unavailable or errored). Ensure your API key is correctly configured).")

        else:
            st.warning("No models were loaded to perform predictions.")