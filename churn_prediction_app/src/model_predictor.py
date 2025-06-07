import pandas as pd
import joblib
import os
from tf_keras.models import load_model
from tf_keras import backend as K
import tensorflow as tf
import numpy as np
import google.generativeai as genai

# Import preprocess_data from preprocessor.py
from preprocessor import preprocess_data

# Re-define focal_loss for loading model if it was used in training
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1. - eps)
        pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
        pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def load_all_models():
    """
    Loads all trained models and the scaler.
    It determines the models directory relative to the current file.
    Includes verbose logging for debugging deployment issues.
    Returns:
        dict: A dictionary containing loaded models and the scaler.
    """
    loaded_assets = {}
    
    # Get the directory of the current script (model_predictor.py)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Assume 'models' directory is one level up from 'src' (at the project root)
    project_root = os.path.dirname(current_script_dir)
    models_dir = os.path.join(project_root, 'models')

    print(f"DEBUG: Attempting to load models from calculated path: {models_dir}")
    print(f"DEBUG: Current working directory (from os.getcwd()): {os.getcwd()}")
    print(f"DEBUG: Contents of current_script_dir ({current_script_dir}): {os.listdir(current_script_dir) if os.path.exists(current_script_dir) else 'Path does not exist'}")
    print(f"DEBUG: Contents of project_root ({project_root}): {os.listdir(project_root) if os.path.exists(project_root) else 'Path does not exist'}")
    print(f"DEBUG: Contents of models_dir ({models_dir}): {os.listdir(models_dir) if os.path.exists(models_dir) else 'Path does not exist'}")


    # Define paths to models and scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    xgb_path = os.path.join(models_dir, 'xgb_model.joblib') # Assumes model saved as xgb_model.joblib
    
    # --- FIX APPLIED HERE: Changed .h5 to .keras for ANN models ---
    ann_class_weights_path = os.path.join(models_dir, 'ann_class_weights_model.keras')
    ann_smote_path = os.path.join(models_dir, 'ann_smote_model.keras')
    ann_focal_loss_path = os.path.join(models_dir, 'ann_focal_loss_model.keras')
    # --- END FIX ---

    custom_objects = {'focal_loss_fixed': focal_loss()} # Required for loading ANN models with custom loss

    # Load scaler
    try:
        if os.path.exists(scaler_path):
            loaded_assets['scaler'] = joblib.load(scaler_path)
            print(f"DEBUG: Successfully loaded scaler from {scaler_path}")
        else:
            print(f"ERROR: Scaler file not found at expected path: {scaler_path}")
    except Exception as e:
        print(f"ERROR: Exception loading scaler: {e}")

    # Load XGBoost model
    try:
        if os.path.exists(xgb_path):
            loaded_assets['xgb_smote'] = joblib.load(xgb_path)
            print(f"DEBUG: Successfully loaded XGBoost model from {xgb_path}")
        else:
            print(f"ERROR: XGBoost model file not found at expected path: {xgb_path}")
    except Exception as e:
        print(f"ERROR: Exception loading XGBoost model: {e}")

    # Load ANN models
    ann_models_to_load = {
        'ann_class_weights': ann_class_weights_path,
        'ann_smote': ann_smote_path,
        'ann_focal_loss': ann_focal_loss_path
    }

    for key, path in ann_models_to_load.items():
        try:
            if os.path.exists(path):
                # Ensure the Keras session is cleared before loading models
                K.clear_session()
                loaded_assets[key] = load_model(path, custom_objects=custom_objects)
                print(f"DEBUG: Successfully loaded {key} model from {path}")
            else:
                print(f"ERROR: {key} model file not found at expected path: {path}")
        except Exception as e:
            print(f"ERROR: Exception loading {key} model from {path}: {e}")
            # If a Keras model fails to load, clearing session might help for next attempt
            K.clear_session()
            
    # Ensure models are loaded and compiled (if ANN)
    for model_key in ['ann_class_weights', 'ann_smote', 'ann_focal_loss']:
        if model_key in loaded_assets and loaded_assets[model_key] is not None:
            try:
                if not hasattr(loaded_assets[model_key], 'optimizer') or loaded_assets[model_key].optimizer is None:
                    # Compile with dummy optimizer and loss if not already compiled
                    loaded_assets[model_key].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    print(f"DEBUG: Re-compiled {model_key} for deployment stability.")
            except Exception as e:
                print(f"WARNING: Could not re-compile {model_key}: {e}")

    return loaded_assets


# Helper for Gemini recommendations
def get_gemini_recommendations(gemini_model, churn_risk_level, customer_details, pre_listed_offers, company_name):
    """
    Generates AI-powered recommendations using the Google Gemini model.
    """
    try:
        # Construct the prompt based on churn risk and customer details
        prompt_parts = [
            f"As an AI assistant for {company_name}, analyze the following customer details and churn risk:\n\n",
            f"Customer Churn Risk: {churn_risk_level}\n",
            "Customer Details:\n",
            f"- Subscription Length: {customer_details.get('SubscriptionLength', 'N/A')} months\n",
            f"- Monthly Charge: ${customer_details.get('MonthlyCharges', 'N/A')}\n",
            f"- Total Usage (Seconds): {customer_details.get('SecondsUse', 'N/A')}\n",
            f"- Frequency of Use: {customer_details.get('FrequencyUse', 'N/A')}\n",
            f"- Call Failures: {customer_details.get('CallFailure', 'N/A')}\n",
            f"- Complaints: {'Yes' if customer_details.get('Complains', 0) == 1 else 'No'}\n",
            f"- Age: {customer_details.get('Age', 'N/A')}\n",
            f"- Customer Value: ${customer_details.get('CustomerValue', 'N/A')}\n",
            "\nBased on this, provide concise, actionable, and empathetic recommendations for churn prevention.\n",
            "If the risk is LOW, suggest ways to enhance loyalty. If the risk is HIGH, suggest retention strategies.\n",
            "Keep the recommendations brief (1-3 sentences) and bullet points are preferred. Do not use markdown headers."
        ]

        # Add pre-listed offers if available and risk is high
        if churn_risk_level == "HIGH CHURN RISK" and pre_listed_offers:
            prompt_parts.append("\nConsider these potential offers to prevent churn:\n")
            for offer in pre_listed_offers:
                prompt_parts.append(f"- {offer}\n")
            prompt_parts.append("\nIncorporate these offers naturally if relevant.\n")

        response = gemini_model.generate_content("".join(prompt_parts))
        return response.text
    except Exception as e:
        print(f"Error generating Gemini recommendations: {e}")
        return None

def predict_churn(model, customer_df, scaler, X_train_columns, model_type='xgb', gemini_model=None):
    """
    Makes a churn prediction for a single customer using the given model.
    Also generates AI recommendations for churn prevention.
    Args:
        model: The trained churn prediction model (XGBoost or Keras ANN).
        customer_df (pd.DataFrame): DataFrame with a single customer's data.
        scaler: The fitted StandardScaler.
        X_train_columns (list): List of columns from the training data.
        model_type (str): Type of model ('xgb' or 'ann').
        gemini_model: The Google Gemini GenerativeModel object.
    Returns:
        tuple: (prediction (0 or 1), probability, AI recommendations string)
    """
    # Fix 1: Check if input customer_df is empty as a DataFrame
    if customer_df.empty:
        return 0, 0.0, "Customer data is empty."

    # Preprocess the single customer data
    # Ensure customer_df has the same columns as X_train_columns after dummy encoding
    customer_processed, _, _, _, _, _ = preprocess_data(customer_df, is_training=False, scaler=scaler, X_train_columns=X_train_columns)

    # Fix 2: Check for empty NumPy array
    if customer_processed is None or customer_processed.size == 0: # Changed .empty to .size == 0
        return 0, 0.0, "Preprocessing failed or resulted in empty data."

    prediction = 0
    probability = 0.0

    if model_type == 'xgb':
        # XGBoost prediction
        probability = model.predict_proba(customer_processed)[:, 1][0]
        prediction = 1 if probability >= 0.5 else 0
    elif model_type == 'ann':
        # ANN prediction (Keras models)
        probability = model.predict(customer_processed)[0][0]
        prediction = 1 if probability >= 0.5 else 0
    else:
        return 0, 0.0, "Unsupported model type."

    # Generate AI recommendations
    churn_risk_level = "HIGH CHURN RISK" if prediction == 1 else "LOW CHURN RISK"
    customer_details = customer_df.iloc[0].to_dict()

    ai_recommendations = get_gemini_recommendations(
        gemini_model,
        churn_risk_level,
        customer_details,
        [], # PRE_LISTED_OFFERS is defined in app.py, so pass an empty list here or adjust if you move it
        "ABC Telecom" # COMPANY_NAME is defined in app.py, so pass string here or adjust if you move it
    )

    return prediction, probability, ai_recommendations

if __name__ == '__main__':
    # This block is for testing purposes only
    print("Running model_predictor.py as a standalone script for testing...")

    # Set up dummy environment for testing
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN optimizations
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Hide GPUs from TensorFlow

    # Configure Gemini (for testing purposes, replace with your actual key or env var)
    # Be cautious with hardcoding API keys; use environment variables or Streamlit secrets in production.
    # genai.configure(api_key="YOUR_GEMINI_API_KEY") # Uncomment and set your API key if testing Gemini here

    # Load models
    print("\nAttempting to load all models...")
    loaded_assets = load_all_models()
    
    scaler = loaded_assets.get('scaler')
    xgb_model = loaded_assets.get('xgb_smote')
    ann_model = loaded_assets.get('ann_class_weights') # Using class weights model for example
    
    # Dummy X_train_columns for testing - you should get this from your preprocessor
    # In a real app, this would be loaded or derived from training data
    X_train_columns = ['CallFailure', 'Complains', 'SubscriptionLength', 'ChargeAmount', 
                       'SecondsUse', 'FrequencyUse', 'FrequencySMS', 'DistinctCalls', 
                       'AgeGroup', 'TariffPlan', 'Status', 'Age', 'CustomerValue']

    if all([scaler, xgb_model, ann_model, X_train_columns]):
        print("All critical assets loaded for testing.")

        # Create a dummy DataFrame for a single customer
        df_new_single = pd.DataFrame({
            'CallFailure': [5], 'Complains': [0], 'SubscriptionLength': [30],
            'ChargeAmount': [1], 'SecondsUse': [1000], 'FrequencyUse': [20],
            'FrequencySMS': [10], 'DistinctCalls': [5], 'AgeGroup': [2],
            'TariffPlan': [1], 'Status': [1], 'Age': [25],
            'CustomerValue': [500.0]
        })

        # Initialize Gemini model for testing if API key is configured
        gemini_model_test = None
        if os.getenv("GEMINI_API_KEY"):
            try:
                gemini_model_test = genai.GenerativeModel('gemini-1.5-flash')
                print("Gemini model initialized for testing.")
            except Exception as e:
                print(f"Could not initialize Gemini model for testing: {e}")
        else:
            print("GEMINI_API_KEY not set in environment variables. Skipping Gemini testing.")


        # Predict with XGBoost
        xgb_preds, xgb_probs, xgb_recs = predict_churn(xgb_model, df_new_single, scaler, X_train_columns, model_type='xgb', gemini_model=gemini_model_test)
        print(f"\n--- XGBoost Test Prediction ---")
        print(f"Prediction: {'Churn' if xgb_preds == 1 else 'No Churn'}, Probability: {xgb_probs:.4f}") # Adjusted for single value
        print(f"Recommendations: {xgb_recs}")

        # Predict with ANN
        ann_preds, ann_probs, ann_recs = predict_churn(ann_model, df_new_single, scaler, X_train_columns, model_type='ann', gemini_model=gemini_model_test)
        print(f"\n--- ANN Test Prediction ---")
        print(f"Prediction: {'Churn' if ann_preds == 1 else 'No Churn'}, Probability: {ann_probs:.4f}") # Adjusted for single value
        print(f"Recommendations: {ann_recs}")

    else:
        print("Not all critical assets could be loaded for testing. Please check paths and file existence.")