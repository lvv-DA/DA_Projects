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

def load_all_models(models_dir): # Added models_dir as parameter
    """
    Loads all trained models and the scaler.
    It determines the models directory relative to the current file.
    Includes verbose logging for debugging deployment issues.
    Args:
        models_dir (str): The path to the directory containing the model files.
    Returns:
        dict: A dictionary containing loaded scaler, XGBoost, and ANN models.
              Returns None for any asset that fails to load.
    """
    loaded_assets = {
        'scaler': None,
        'xgb_smote': None,
        'ann_class_weights': None,
        'ann_smote': None,
        'ann_focal_loss': None
    }
    
    # Define custom objects for Keras model loading
    custom_objects = {'focal_loss_fixed': focal_loss(gamma=2., alpha=.25)}

    # Load Scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    try:
        loaded_assets['scaler'] = joblib.load(scaler_path)
        print(f"Successfully loaded scaler from {scaler_path}")
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {scaler_path}")
    except Exception as e:
        print(f"Error loading scaler from {scaler_path}: {e}")

    # Load XGBoost model
    xgb_smote_path = os.path.join(models_dir, 'xgb_smote_model.joblib')
    try:
        loaded_assets['xgb_smote'] = joblib.load(xgb_smote_path)
        print(f"Successfully loaded XGBoost model from {xgb_smote_path}")
    except FileNotFoundError:
        print(f"Error: XGBoost model file not found at {xgb_smote_path}")
    except Exception as e:
        print(f"Error loading XGBoost model from {xgb_smote_path}: {e}")

    # Load ANN models (HDF5 files)
    ann_class_weights_path = os.path.join(models_dir, 'ann_class_weights.h5')
    try:
        # Load ANN models with custom_objects
        loaded_assets['ann_class_weights'] = load_model(ann_class_weights_path, custom_objects=custom_objects)
        print(f"Successfully loaded ANN Class Weights model from {ann_class_weights_path}")
    except FileNotFoundError:
        print(f"Error: ANN Class Weights model file not found at {ann_class_weights_path}")
    except Exception as e:
        print(f"Error loading ANN Class Weights model from {ann_class_weights_path}: {e}")

    ann_smote_path = os.path.join(models_dir, 'ann_smote.h5')
    try:
        # Load ANN models with custom_objects
        loaded_assets['ann_smote'] = load_model(ann_smote_path, custom_objects=custom_objects)
        print(f"Successfully loaded ANN SMOTE model from {ann_smote_path}")
    except FileNotFoundError:
        print(f"Error: ANN SMOTE model file not found at {ann_smote_path}")
    except Exception as e:
        print(f"Error loading ANN SMOTE model from {ann_smote_path}: {e}")

    ann_focal_loss_path = os.path.join(models_dir, 'ann_focal_loss.h5')
    try:
        # Load ANN models with custom_objects
        loaded_assets['ann_focal_loss'] = load_model(ann_focal_loss_path, custom_objects=custom_objects)
        print(f"Successfully loaded ANN Focal Loss model from {ann_focal_loss_path}")
    except FileNotFoundError:
        print(f"Error: ANN Focal Loss model file not found at {ann_focal_loss_path}")
    except Exception as e:
        print(f"Error loading ANN Focal Loss model from {ann_focal_loss_path}: {e}")

    return loaded_assets


def predict_churn(model, customer_df, scaler, X_train_columns, model_type='xgb'):
    """
    Predicts churn for a single customer using the loaded model.
    Args:
        model: The loaded trained model (XGBoost or Keras ANN).
        customer_df (pd.DataFrame): DataFrame containing a single customer's data.
        scaler: The fitted StandardScaler.
        X_train_columns (list): List of columns from the training data to ensure consistency.
        model_type (str): 'xgb' for XGBoost, 'ann' for Keras ANN.
    Returns:
        tuple: (prediction, probability)
    """
    # Preprocess the single customer data
    # The preprocess_data function now handles aligning columns and scaling
    X_processed, _, _, _, _, _ = preprocess_data(
        customer_df, is_training=False, scaler=scaler, X_train_columns=X_train_columns, target_column=None # No target_column for prediction
    )
    
    prediction = 0
    probability = 0.0

    if model_type == 'xgb':
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[:, 1][0]
    elif model_type == 'ann':
        # Keras models return probabilities directly (or logits that need sigmoid)
        # Assuming your ANN models output a single probability for class 1
        raw_probability = model.predict(X_processed, verbose=0)[0][0] # Get the single probability
        probability = float(raw_probability)
        prediction = 1 if probability >= 0.5 else 0
    else:
        raise ValueError("Invalid model_type. Must be 'xgb' or 'ann'.")

    return prediction, probability


def get_gemini_recommendations(gemini_model, churn_risk_level, customer_details, pre_listed_offers, company_name):
    """
    Generates AI-powered recommendations using the Google Gemini model.
    Args:
        gemini_model: The configured Google Gemini generative model.
        churn_risk_level (str): "HIGH CHURN RISK" or "LOW CHURN RISK".
        customer_details (dict): Dictionary of customer's input features.
        pre_listed_offers (list): A list of pre-defined offers.
        company_name (str): The name of the company.
    Returns:
        str: AI-generated recommendations, or None if an error occurs.
    """
    if gemini_model is None:
        return "AI recommendations are unavailable due to an uninitialized Gemini model."

    try:
        customer_info_str = "\n".join([f"- {k}: {v}" for k, v in customer_details.items()])
        offers_str = "\n".join([f"- {offer}" for offer in pre_listed_offers])

        prompt = f"""
        You are an AI assistant for {company_name}, a telecom company. Your goal is to provide concise and actionable recommendations for customer retention based on their churn risk and profile.

        Customer Churn Risk: {churn_risk_level}

        Customer Profile:
        {customer_info_str}

        Pre-listed Offers:
        {offers_str}

        If the customer has 'HIGH CHURN RISK':
        1. Analyze the customer's profile to identify potential reasons for churn (e.g., high call failures, low usage, low subscription length, complaints).
        2. Suggest which of the `Pre-listed Offers` would be most suitable for this specific customer's profile to reduce churn.
        3. Provide additional, creative, and personalized retention strategies (e.g., proactive check-ins, personalized discounts, exclusive access) that are NOT in the `Pre-listed Offers`.

        If the customer has 'LOW CHURN RISK':
        1. Suggest strategies to maintain their loyalty and increase their engagement.
        2. Recommend potential upselling opportunities or new services based on their profile.
        3. Briefly mention a relevant pre-listed offer that could enhance their experience.

        Ensure the recommendations are:
        - Actionable and specific.
        - Presented in clear bullet points or numbered lists.
        - Tailored to the customer's profile.
        - Professional and empathetic in tone.
        """
        
        response = gemini_model.generate_content(prompt)
        # Access the text attribute if it exists, otherwise handle potential errors
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            # Fallback for models that might return candidates directly
            for candidate in response.candidates:
                if hasattr(candidate, 'text'):
                    return candidate.text
        return "AI recommendations could not be generated (no text in response)."

    except Exception as e:
        print(f"Error generating Gemini recommendations: {e}")
        return f"AI recommendations could not be generated due to an error: {e}"


# Example usage for testing (only runs if model_predictor.py is executed directly)
if __name__ == '__main__':
    print("Running model_predictor.py as a standalone script for testing...")
    
    # Set GEMINI_API_KEY for local testing (replace with your actual key or load from .env)
    # os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY" # Uncomment for local testing

    # Define paths relative to the project root for testing
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_test = os.path.dirname(current_script_dir) # Go up one level from 'src'

    models_dir_test = os.path.join(project_root_test, 'models')
    data_dir_test = os.path.join(project_root_test, 'data')
    
    # Ensure models and data directories exist
    os.makedirs(models_dir_test, exist_ok=True)
    os.makedirs(data_dir_test, exist_ok=True)

    # Dummy data for X_train_columns inference
    dummy_churn_data_path = os.path.join(data_dir_test, 'customer_churn.xlsx - Sheet1.csv')
    if not os.path.exists(dummy_churn_data_path):
        dummy_df_churn = pd.DataFrame({
            'CallFailure': [8,0,10], 'Complains': [0,0,0], 'SubscriptionLength': [38,39,37],
            'ChargeAmount': [0,0,0], 'SecondsUse': [4370,318,2453], 'FrequencyUse': [71,5,60],
            'FrequencySMS': [5,7,359], 'DistinctCalls': [17,6,20], 'AgeGroup': ['Group2','Group3','Group2'],
            'TariffPlan': ['PlanA','PlanB','PlanA'], 'Status': ['Active','Inactive','Active'], 'Age': [30,45,28],
            'CustomerValue': [800.0, 1200.0, 700.0], 'Churn': [0, 1, 0]
        })
        dummy_df_churn.to_csv(dummy_churn_data_path, index=False)
        print(f"Dummy customer_churn.xlsx - Sheet1.csv created for testing at {dummy_churn_data_path}")
    
    initial_df_for_cols = pd.read_csv(dummy_churn_data_path, encoding='latin1')
    categorical_cols_for_inference = initial_df_for_cols.select_dtypes(include=['object', 'category']).columns.tolist()
    X_for_cols_inference = initial_df_for_cols.drop(columns=['Churn'], errors='ignore')
    X_train_columns = pd.get_dummies(X_for_cols_inference, columns=categorical_cols_for_inference, drop_first=True).columns.tolist()


    # Load models and scaler
    loaded_assets = load_all_models(models_dir_test) # Pass models_dir_test
    scaler = loaded_assets.get('scaler')
    xgb_model = loaded_assets.get('xgb_smote')
    ann_class_weights = loaded_assets.get('ann_class_weights') # Use specific ANN model for test
    
    # --- Dummy customer data for prediction test ---
    df_new_single = pd.DataFrame([{
        'CallFailure': 2, 'Complains': 0, 'SubscriptionLength': 24,
        'ChargeAmount': 60, 'SecondsUse': 1500, 'FrequencyUse': 30,
        'FrequencySMS': 15, 'DistinctCalls': 8, 'AgeGroup': 'Group2',
        'TariffPlan': 'PlanA', 'Status': 'Active', 'Age': 35,
        'CustomerValue': 750.0
    }])

    # Initialize Gemini model for testing if API key is configured
    gemini_model_test = None
    if os.getenv("GEMINI_API_KEY"):
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            gemini_model_test = genai.GenerativeModel('gemini-1.5-flash')
            print("Gemini model initialized for testing.")
        except Exception as e:
            print(f"Could not initialize Gemini model for testing: {e}")
    else:
        print("GEMINI_API_KEY not set in environment variables. Skipping Gemini testing.")


    # Predict with XGBoost
    if xgb_model and scaler and X_train_columns:
        xgb_preds, xgb_probs = predict_churn(xgb_model, df_new_single.copy(), scaler, X_train_columns, model_type='xgb')
        print(f"\n--- XGBoost Test Prediction ---")
        print(f"Prediction: {'Churn' if xgb_preds == 1 else 'No Churn'}, Probability: {xgb_probs:.4f}")
        # Test Gemini recommendation for XGBoost if model loaded
        if gemini_model_test:
            risk_level = "HIGH CHURN RISK" if xgb_preds == 1 else "LOW CHURN RISK"
            xgb_recs = get_gemini_recommendations(gemini_model_test, risk_level, df_new_single.iloc[0].to_dict(), [], "ABC Telecom")
            print(f"Recommendations: {xgb_recs}")
    else:
        print("XGBoost model, scaler or X_train_columns not loaded for testing.")

    # Predict with ANN
    if ann_class_weights and scaler and X_train_columns:
        ann_preds, ann_probs = predict_churn(ann_class_weights, df_new_single.copy(), scaler, X_train_columns, model_type='ann')
        print(f"\n--- ANN Test Prediction (Class Weights) ---")
        print(f"Prediction: {'Churn' if ann_preds == 1 else 'No Churn'}, Probability: {ann_probs:.4f}")
        # Test Gemini recommendation for ANN if model loaded
        if gemini_model_test:
            risk_level = "HIGH CHURN RISK" if ann_preds == 1 else "LOW CHURN RISK"
            ann_recs = get_gemini_recommendations(gemini_model_test, risk_level, df_new_single.iloc[0].to_dict(), [], "ABC Telecom")
            print(f"Recommendations: {ann_recs}")
    else:
        print("ANN Class Weights model, scaler or X_train_columns not loaded for testing.")