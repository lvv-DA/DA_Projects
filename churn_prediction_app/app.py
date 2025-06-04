import streamlit as st
import pandas as pd
import os
import sys

# --- Path Setup ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import your custom modules (assuming these are well-optimized internally)
# Ensure these modules (data_loader.py, model_predictor.py, preprocessor.py)
# are present in your 'src' directory.
from data_loader import load_data
from model_predictor import load_all_models, predict_churn
from preprocessor import preprocess_data

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction App")
st.markdown("""
    This application predicts customer churn based on various customer attributes.
    You can search for an an existing customer to auto-fill their details, or enter new details manually.
    The prediction is made using an **ensemble of four machine learning models** to provide a more robust and insightful result.
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
            X_full_for_cols = df_full_for_cols.drop('Churn', axis=1)
            # IMPORTANT: This assumes your preprocessor.py or model training
            # also applies pd.get_dummies(..., drop_first=True).
            # For maximum robustness, save the exact X_train.columns during model training.
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
    st.stop() # Stop the app execution if critical assets are missing

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
        # Ensure ID/Phone/Name are string type for consistent searching
        if 'CustomerID' in df_customers.columns:
            df_customers['CustomerID'] = df_customers['CustomerID'].astype(str).fillna('')
        if 'PhoneNumber' in df_customers.columns:
            df_customers['PhoneNumber'] = df_customers['PhoneNumber'].astype(str).fillna('')
        if 'CustomerName' in df_customers.columns:
            df_customers['CustomerName'] = df_customers['CustomerName'].astype(str).fillna('')
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
# FIX: Added "0 (No Charge/Undefined)" to handle cases where ChargeAmount might be 0
CHARGE_AMOUNT_OPTIONS = {
    "0 (No Charge/Undefined)": 0,
    "1 (Lowest Charge)": 1, "2": 2, "3": 3, "4": 4, "5": 5,
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

# Initialize default values only if they don't exist in session state
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Initialize search-specific session state variables
if 'search_results_display' not in st.session_state:
    st.session_state['search_results_display'] = []
    st.session_state['search_results_data'] = pd.DataFrame()
    st.session_state['last_search_query'] = ""
    st.session_state['show_search_results'] = False # Control visibility

# --- Customer Search Section ---
st.header("Search for an Existing Customer")

if all_customers_df is not None:
    # Use a direct session state key for the search input value
    if 'search_bar_input' not in st.session_state: # Changed key for clarity
        st.session_state['search_bar_input'] = ''

    search_query_input = st.text_input(
        "Enter Customer Name, ID, or Phone Number to search:",
        value=st.session_state['search_bar_input'],
        key="search_bar", # The actual key for the widget
        placeholder="e.g., Vinu, 100001, 9092777925",
        on_change=lambda: st.session_state.__setitem__('last_search_query_cleared', False) # Reset flag when text changes
    )

    # Update the tracking session state variable with the current input value
    st.session_state['search_bar_input'] = search_query_input

    search_button = st.button("Search Customers")

    # Flag to check if search results were displayed after a button click
    if 'performed_search_with_button' not in st.session_state:
        st.session_state['performed_search_with_button'] = False

    # --- Conditional Search Execution Logic ---
    if search_button: # Only perform search on button click
        st.session_state['performed_search_with_button'] = True # Mark that search button was pressed
        st.session_state['last_search_query'] = search_query_input # Store the query that triggered the search

        if search_query_input:
            with st.spinner(f"Searching for '{search_query_input}'..."):
                search_results = all_customers_df[
                    all_customers_df['CustomerName'].str.contains(search_query_input, case=False, na=False) |
                    all_customers_df['CustomerID'].str.contains(search_query_input, case=False, na=False) |
                    all_customers_df['PhoneNumber'].str.contains(search_query_input, case=False, na=False)
                ]

            if not search_results.empty:
                display_options = []
                for idx, row in search_results.iterrows():
                    address_info = f" | Address: {row.get('Address', 'N/A')}" # Use .get() for safety
                    display_options.append(f"ID: {row['CustomerID']} | Name: {row['CustomerName']} | Phone: {row['PhoneNumber']}{address_info}")

                st.session_state['search_results_display'] = display_options
                st.session_state['search_results_data'] = search_results
                st.session_state['show_search_results'] = True
                st.subheader(f"Found {len(search_results)} matching customers:")
            else:
                st.info(f"No customers found matching '{search_query_input}'.")
                st.session_state['search_results_display'] = []
                st.session_state['search_results_data'] = pd.DataFrame()
                st.session_state['show_search_results'] = False
        else: # Search button clicked with empty query
            st.session_state['search_results_display'] = []
            st.session_state['search_results_data'] = pd.DataFrame()
            st.session_state['show_search_results'] = False
            st.info("Enter a query in the search bar and click 'Search' to find customers.")

    # --- Clear search results if query is empty and was previously searched ---
    # This prevents stale results from showing if the user clears the search box manually
    if search_query_input == "" and st.session_state['performed_search_with_button'] and st.session_state['show_search_results']:
        st.session_state['search_results_display'] = []
        st.session_state['search_results_data'] = pd.DataFrame()
        st.session_state['show_search_results'] = False
        st.session_state['performed_search_with_button'] = False # Reset this flag

    # --- Display Search Results (if show_search_results is True) ---
    if st.session_state['show_search_results'] and st.session_state['search_results_display']:
        # Define a callback function for st.radio to update session state
        def on_customer_select():
            # Use the actual key of the radio button to get its selected value
            selected_option_label = st.session_state[f"customer_select_radio_{st.session_state['search_bar_input']}"]
            selected_customer_id_from_label = selected_option_label.split(' | ')[0].replace('ID: ', '')

            # Use .loc for potentially faster lookup if 'CustomerID' is unique
            selected_customer_row_filtered = st.session_state['search_results_data'][
                st.session_state['search_results_data']['CustomerID'] == selected_customer_id_from_label
            ]

            if not selected_customer_row_filtered.empty:
                selected_customer_row = selected_customer_row_filtered.iloc[0]

                for feature in default_values.keys():
                    if feature in selected_customer_row and pd.notna(selected_customer_row[feature]):
                        # Ensure type consistency for numerical values
                        # Special handling for 'ChargeAmount' to map value to key for selectbox index
                        if feature == 'ChargeAmount':
                            value_from_data = selected_customer_row[feature]
                            # Find the key (label) corresponding to the value
                            try:
                                # Find the key (label) for the value
                                found_label = next(key for key, val in CHARGE_AMOUNT_OPTIONS.items() if val == value_from_data)
                                st.session_state[feature] = CHARGE_AMOUNT_OPTIONS[found_label]
                            except StopIteration:
                                st.warning(f"ChargeAmount '{value_from_data}' from customer data not found in options. Falling back to default.")
                                st.session_state[feature] = default_values[feature]
                        elif feature == 'Complains':
                            value_from_data = selected_customer_row[feature]
                            try:
                                found_label = next(key for key, val in COMPLAINS_OPTIONS.items() if val == value_from_data)
                                st.session_state[feature] = COMPLAINS_OPTIONS[found_label]
                            except StopIteration:
                                st.warning(f"Complains value '{value_from_data}' from customer data not found in options. Falling back to default.")
                                st.session_state[feature] = default_values[feature]
                        elif feature == 'AgeGroup':
                            value_from_data = selected_customer_row[feature]
                            try:
                                found_label = next(key for key, val in AGE_GROUP_OPTIONS.items() if val == value_from_data)
                                st.session_state[feature] = AGE_GROUP_OPTIONS[found_label]
                            except StopIteration:
                                st.warning(f"AgeGroup value '{value_from_data}' from customer data not found in options. Falling back to default.")
                                st.session_state[feature] = default_values[feature]
                        elif feature == 'TariffPlan':
                            value_from_data = selected_customer_row[feature]
                            try:
                                found_label = next(key for key, val in TARIFF_PLAN_OPTIONS.items() if val == value_from_data)
                                st.session_state[feature] = TARIFF_PLAN_OPTIONS[found_label]
                            except StopIteration:
                                st.warning(f"TariffPlan value '{value_from_data}' from customer data not found in options. Falling back to default.")
                                st.session_state[feature] = default_values[feature]
                        elif feature == 'Status':
                            value_from_data = selected_customer_row[feature]
                            try:
                                found_label = next(key for key, val in STATUS_OPTIONS.items() if val == value_from_data)
                                st.session_state[feature] = STATUS_OPTIONS[found_label]
                            except StopIteration:
                                st.warning(f"Status value '{value_from_data}' from customer data not found in options. Falling back to default.")
                                st.session_state[feature] = default_values[feature]
                        else: # For numerical inputs (number_input)
                            try:
                                target_type = type(default_values[feature])
                                st.session_state[feature] = target_type(selected_customer_row[feature])
                            except (ValueError, TypeError):
                                # Fallback to default if conversion fails
                                st.session_state[feature] = default_values[feature]
                    else: # If feature not in row or is NaN, fall back to default
                        st.session_state[feature] = default_values[feature]

                st.success(f"Details for Customer ID {selected_customer_row.get('CustomerID', 'N/A')} ({selected_customer_row.get('CustomerName', 'N/A')}) loaded!")
                st.rerun() # Ensure the input fields update immediately
            else:
                st.error("Selected customer data not found in search results. Please try again.")

        # Use a unique key for the radio button that depends on the current search query
        st.radio(
            "Select a customer to auto-fill details:",
            st.session_state['search_results_display'],
            key=f"customer_select_radio_{st.session_state['search_bar_input']}", # Use actual input value for key
            on_change=on_customer_select # Trigger callback on selection
        )
    elif not st.session_state['show_search_results'] and not search_query_input and not st.session_state['performed_search_with_button']:
        st.info("Enter a query in the search bar and click 'Search' to find customers.")
    elif st.session_state['show_search_results'] and not st.session_state['search_results_display'] and st.session_state['performed_search_with_button']:
        st.info(f"No customers found for your last search: '{st.session_state['last_search_query']}'.")

else: # If all_customers_df is None
    st.warning("Customer data not available for search. Please ensure 'customer_data_with_identifiers.csv' is in your `/data/` folder.")


# --- Customer Details for Prediction ---
st.header("Customer Details for Prediction")
st.markdown("**(Auto-filled if customer searched, or enter manually)**")

# --- Input Features (Now read from st.session_state via key) ---
col1, col2, col3 = st.columns(3)

with col1:
    st.number_input(
        "Call Failure (Number of failed calls)",
        min_value=0, max_value=50,
        key="CallFailure"
    )
    st.number_input(
        "Subscription Length (months)",
        min_value=0, max_value=100,
        key="SubscriptionLength"
    )
    st.number_input(
        "Seconds Use (total seconds in past year)",
        min_value=0, max_value=20000,
        key="SecondsUse"
    )
    st.number_input(
        "Distinct Calls (number of distinct numbers called in past year)",
        min_value=0, max_value=100,
        key="DistinctCalls"
    )

    # For selectboxes, the `index` needs to be the index of the *selected label*
    # within the `options` list.
    # Get the label from the value stored in session state
    try:
        current_tariff_label = next(key for key, val in TARIFF_PLAN_OPTIONS.items() if val == st.session_state['TariffPlan'])
        tariff_index = list(TARIFF_PLAN_OPTIONS.keys()).index(current_tariff_label)
    except StopIteration:
        # Fallback if the value in session state isn't found in options
        tariff_index = list(TARIFF_PLAN_OPTIONS.values()).index(default_values['TariffPlan'])

    selected_tariff_label = st.selectbox(
        "Tariff Plan",
        options=list(TARIFF_PLAN_OPTIONS.keys()),
        index=tariff_index,
        key="TariffPlan_selectbox"
    )
    # Update numerical value in session state based on selectbox label
    st.session_state['TariffPlan'] = TARIFF_PLAN_OPTIONS[selected_tariff_label]


with col2:
    try:
        current_complains_label = next(key for key, val in COMPLAINS_OPTIONS.items() if val == st.session_state['Complains'])
        complains_index = list(COMPLAINS_OPTIONS.keys()).index(current_complains_label)
    except StopIteration:
        complains_index = list(COMPLAINS_OPTIONS.values()).index(default_values['Complains'])

    selected_complains_label = st.selectbox(
        "Complains (Has the customer filed a complaint?)",
        options=list(COMPLAINS_OPTIONS.keys()),
        index=complains_index,
        key="Complains_selectbox"
    )
    st.session_state['Complains'] = COMPLAINS_OPTIONS[selected_complains_label]

    # FIX APPLIED HERE: Adjusting selectbox index for ChargeAmount
    try:
        current_charge_label = next(key for key, val in CHARGE_AMOUNT_OPTIONS.items() if val == st.session_state['ChargeAmount'])
        charge_index = list(CHARGE_AMOUNT_OPTIONS.keys()).index(current_charge_label)
    except StopIteration:
        # If the value in session state isn't in the options (e.g., if it was 0 before the fix),
        # gracefully fall back to the default value's index.
        st.warning(f"Charge Amount value {st.session_state['ChargeAmount']} not found in options. Resetting to default.")
        st.session_state['ChargeAmount'] = default_values['ChargeAmount']
        charge_index = list(CHARGE_AMOUNT_OPTIONS.values()).index(default_values['ChargeAmount'])


    selected_charge_label = st.selectbox(
        "Charge Amount (Categorical)",
        options=list(CHARGE_AMOUNT_OPTIONS.keys()),
        index=charge_index, # This is now correctly handled
        key="ChargeAmount_selectbox"
    )
    st.session_state['ChargeAmount'] = CHARGE_AMOUNT_OPTIONS[selected_charge_label]

    st.number_input(
        "Frequency Use (total calls in past year)",
        min_value=0, max_value=200,
        key="FrequencyUse"
    )

    try:
        current_age_group_label = next(key for key, val in AGE_GROUP_OPTIONS.items() if val == st.session_state['AgeGroup'])
        age_group_index = list(AGE_GROUP_OPTIONS.keys()).index(current_age_group_label)
    except StopIteration:
        age_group_index = list(AGE_GROUP_OPTIONS.values()).index(default_values['AgeGroup'])

    selected_age_group_label = st.selectbox(
        "Age Group Category",
        options=list(AGE_GROUP_OPTIONS.keys()),
        index=age_group_index,
        key="AgeGroup_selectbox"
    )
    st.session_state['AgeGroup'] = AGE_GROUP_OPTIONS[selected_age_group_label]

    st.number_input(
        "Age (Customer age)",
        min_value=15, max_value=90,
        key="Age"
    )


with col3:
    st.number_input(
        "Frequency SMS (total SMS in past year)",
        min_value=0, max_value=500,
        key="FrequencySMS"
    )

    try:
        current_status_label = next(key for key, val in STATUS_OPTIONS.items() if val == st.session_state['Status'])
        status_index = list(STATUS_OPTIONS.keys()).index(current_status_label)
    except StopIteration:
        status_index = list(STATUS_OPTIONS.values()).index(default_values['Status'])

    selected_status_label = st.selectbox(
        "Status",
        options=list(STATUS_OPTIONS.keys()),
        index=status_index,
        key="Status_selectbox"
    )
    st.session_state['Status'] = STATUS_OPTIONS[selected_status_label]

    st.number_input(
        "Customer Value (projected for next year)",
        min_value=0.0, max_value=5000.0,
        format="%.2f",
        key="CustomerValue"
    )

# Collect inputs into a DataFrame from session state
input_data = pd.DataFrame({
    'CallFailure': [st.session_state['CallFailure']], 'Complains': [st.session_state['Complains']],
    'SubscriptionLength': [st.session_state['SubscriptionLength']], 'ChargeAmount': [st.session_state['ChargeAmount']],
    'SecondsUse': [st.session_state['SecondsUse']], 'FrequencyUse': [st.session_state['FrequencyUse']],
    'FrequencySMS': [st.session_state['FrequencySMS']], 'DistinctCalls': [st.session_state['DistinctCalls']],
    'AgeGroup': [st.session_state['AgeGroup']], 'TariffPlan': [st.session_state['TariffPlan']],
    'Status': [st.session_state['Status']], 'Age': [st.session_state['Age']],
    'CustomerValue': [st.session_state['CustomerValue']]
})

st.subheader("Ensemble Prediction")

if st.button("Predict Churn with Ensemble"):
    if not all([m['model'] for m in ENSEMBLE_MODELS.values()]):
        st.error("One or more ensemble models could not be loaded. Cannot make prediction.")
    else:
        with st.spinner("Making predictions with the ensemble models..."):
            individual_predictions = []
            prediction_results = [] # To store probabilities for display

            for model_name, model_info in ENSEMBLE_MODELS.items():
                model = model_info['model']
                model_type = model_info['type']

                # Predict churn for the current input data using the specific model
                prediction, probability = predict_churn(
                    model,
                    input_data,
                    scaler,
                    X_train_columns,
                    model_type
                )
                individual_predictions.append(prediction[0])
                prediction_results.append({
                    "Model": model_name,
                    "Predicted Churn (0/1)": prediction[0],
                    "Churn Probability": f"{probability[0]:.2%}"
                })

            # Convert results to a DataFrame for display
            df_predictions = pd.DataFrame(prediction_results)

            # Calculate majority vote
            churn_votes = sum(individual_predictions)
            no_churn_votes = len(individual_predictions) - churn_votes

            st.subheader("Individual Model Predictions")
            st.dataframe(df_predictions, hide_index=True)

            st.subheader("Ensemble Decision (Majority Vote)")
            if churn_votes >= no_churn_votes: # If 2 or more models predict churn (out of 4)
                st.error(f"The ensemble predicts the customer will **CHURN** (based on {churn_votes} out of {len(ENSEMBLE_MODELS)} models).")
                st.snow()
            else:
                st.success(f"The ensemble predicts the customer will **NOT CHURN** (based on {no_churn_votes} out of {len(ENSEMBLE_MODELS)} models).")
                st.balloons()

        st.markdown(f"**Interpretation:**")
        st.info(f"""
            - The table above shows the individual prediction and churn probability from each of the four models in the ensemble.
            - A higher 'Churn Probability' indicates a greater likelihood of the customer churning according to that specific model.
            - The final **Ensemble Decision** is determined by a majority vote: if more than half of the models predict churn, the ensemble predicts churn.
            - This ensemble approach provides a more robust and reliable prediction by aggregating insights from diverse models.
        """)

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit")