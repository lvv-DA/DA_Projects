import pandas as pd
import joblib
import os
from tf_keras.models import load_model
from tf_keras import backend as K
import tensorflow as tf
import numpy as np
import google.generativeai as genai

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

def load_all_models(models_dir='models'):
    """
    Loads all trained models and the scaler.
    Args:
        models_dir (str): Directory where models and scaler are saved.
    Returns:
        dict: A dictionary containing loaded models and the scaler.
    """
    loaded_assets = {}

    # Load scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    try:
        loaded_assets['scaler'] = joblib.load(scaler_path)
        print(f"Scaler loaded from {scaler_path}")
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {scaler_path}")
        loaded_assets['scaler'] = None
    except Exception as e:
        print(f"An error occurred loading scaler: {e}")
        loaded_assets['scaler'] = None

    # Load XGBoost model
    xgb_path = os.path.join(models_dir, 'xgb_smote_model.pkl')
    try:
        loaded_assets['xgb_smote'] = joblib.load(xgb_path)
        print(f"XGBoost model loaded from {xgb_path}")
    except FileNotFoundError:
        print(f"Error: XGBoost model file not found at {xgb_path}")
        loaded_assets['xgb_smote'] = None
    except Exception as e:
        print(f"An error occurred loading XGBoost model: {e}")
        loaded_assets['xgb_smote'] = None

    # Load Keras models (need custom_objects for focal_loss)
    custom_objects = {'focal_loss_fixed': focal_loss()} # Only if focal_loss is used
    keras_models = {
        'ann_class_weights': 'ann_class_weights_model.keras',
        'ann_smote': 'ann_smote_model.keras',
        'ann_focal_loss': 'ann_focal_loss_model.keras'
    }
    for name, filename in keras_models.items():
        model_path = os.path.join(models_dir, filename)
        try:
            # compile=False is often used for prediction to avoid recompilation overhead
            # and potential issues with custom objects if not strictly needed for prediction logic.
            # However, if focal_loss is part of the model graph that needs to be active for prediction,
            # you might need to compile=True and provide custom_objects.
            # For simplicity, we'll try with compile=False first.
            loaded_model = load_model(model_path, custom_objects=custom_objects if name == 'ann_focal_loss' else None, compile=False)
            loaded_assets[name] = loaded_model
            print(f"{name.replace('_', ' ').title()} model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Error: {name} model file not found at {model_path}")
            loaded_assets[name] = None
        except Exception as e:
            print(f"An error occurred loading {name} model: {e}")
            loaded_assets[name] = None
            
    return loaded_assets


def predict_churn(model, df_input, scaler, X_train_columns, model_type='xgb', gemini_model=None):
    """
    Makes churn predictions using a given model and preprocessed input data,
    and optionally generates churn prevention recommendations using a Gemini model.

    Args:
        model: The trained machine learning model (XGBoost or Keras ANN).
        df_input (pd.DataFrame): The raw input DataFrame for prediction.
        scaler (StandardScaler): The fitted scaler.
        X_train_columns (list): List of column names used during training.
        model_type (str): Type of model ('xgb' or 'ann').
        gemini_model: The initialized Google Gemini generative model (optional).

    Returns:
        tuple: (predictions (np.array), probabilities (np.array), recommendations (str or None))
    """
    if model is None:
        print(f"Error: Model is not loaded for type {model_type}.")
        return np.array([]), np.array([]), None

    # Ensure input matches training columns (handle dummy variables, missing columns)
    df_processed = pd.get_dummies(df_input.copy(), drop_first=True)

    # Add missing columns (from training data) and reorder
    for col in X_train_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
    df_processed = df_processed[X_train_columns]

    if model_type == 'xgb':
        probs = model.predict_proba(df_processed)[:, 1]
    elif model_type == 'ann':
        # Scale for ANN models
        X_scaled = scaler.transform(df_processed)
        probs = model.predict(X_scaled).flatten()
    else:
        raise ValueError("model_type must be 'xgb' or 'ann'")

    preds = (probs > 0.5).astype(int)

    recommendations = None
    if gemini_model is not None and len(preds) > 0:
        # Assuming you're predicting for a single customer for simplicity in Streamlit app
        customer_data = df_input.iloc[0].to_dict() if not df_input.empty else {}
        churn_prediction = "Likely to Churn" if preds[0] == 1 else "Unlikely to Churn"
        churn_probability = probs[0] * 100
        
        prompt = (
            f"A customer with the following profile has been predicted as '{churn_prediction}' "
            f"with a probability of {churn_probability:.2f}%. "
            f"Customer details: {customer_data}. "
            "Please provide concise, actionable recommendations for a customer service representative "
            "to either retain this customer (if high churn risk) or enhance loyalty (if low churn risk). "
            "Suggest concrete steps, considering the customer's attributes."
        )
        try:
            # For a production app, consider using genai.ChatSession or more structured prompts
            # For now, a direct generate_content call will suffice.
            ai_response = gemini_model.generate_content(prompt)
            recommendations = ai_response.text
        except Exception as e:
            print(f"Error generating Gemini recommendations: {e}")
            recommendations = "Could not generate AI recommendations."

    return preds, probs, recommendations

if __name__ == '__main__':
    from preprocessor import preprocess_data # Import preprocess_data
    # Dummy data for demonstration
    df_new_single = pd.DataFrame({
        'CallFailure': [5], 'Complains': [0], 'SubscriptionLength': [30],
        'ChargeAmount': [1], 'SecondsUse': [1000], 'FrequencyUse': [20],
        'FrequencySMS': [10], 'DistinctCalls': [5], 'AgeGroup': [2],
        'TariffPlan': [1], 'Status': [1], 'Age': [25],
        'CustomerValue': [500.0]
    })

    # Adjust project_root for script execution context
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, 'models')

    # Ensure models directory exists
    if not os.path.exists(models_dir):
        print(f"Models directory not found at {models_dir}. Please run model_trainer.py first.")
        exit()

    # Load all models and scaler
    loaded_assets = load_all_models(models_dir)
    xgb_model = loaded_assets.get('xgb_smote')
    ann_model = loaded_assets.get('ann_class_weights') # Or 'ann_smote', 'ann_focal_loss'
    scaler = loaded_assets.get('scaler')

    # Dummy X_train_columns for demonstration (in a real scenario, these would be saved/loaded)
    # This list must precisely match the columns the model was trained on AFTER get_dummies.
    # For robust production, save these from model_trainer.py
    X_train_columns = ['CallFailure', 'Complains', 'SubscriptionLength', 'ChargeAmount',
                       'SecondsUse', 'FrequencyUse', 'FrequencySMS', 'DistinctCalls',
                       'AgeGroup', 'TariffPlan', 'Status', 'Age', 'CustomerValue'] # Example, update with actual trained columns

    if xgb_model and scaler:
        # Initialize Gemini model (if API key is available)
        try:
            # Assuming GEMINI_API_KEY is an environment variable for local testing
            genai.configure(api_key=os.getenv("GEMINI_API_KEY")) # Ensure your API key is set
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Use gemini-1.5-flash
        except Exception as e:
            print(f"Gemini API initialization failed: {e}. AI recommendations will be unavailable.")
            gemini_model = None

        # Predict with XGBoost
        xgb_preds, xgb_probs, xgb_recs = predict_churn(xgb_model, df_new_single, scaler, X_train_columns, model_type='xgb', gemini_model=gemini_model)
        print(f"\nXGBoost Prediction: {xgb_preds[0]}, Probability: {xgb_probs[0]:.4f}, Recommendations: {xgb_recs}")

        # Predict with ANN + Class Weights
        ann_preds, ann_probs, ann_recs = predict_churn(ann_model, df_new_single, scaler, X_train_columns, model_type='ann', gemini_model=gemini_model)
        print(f"ANN + Class Weights Prediction: {ann_preds[0]}, Probability: {ann_probs[0]:.4f}, Recommendations: {ann_recs}")

        # Predict with ANN + SMOTE
        ann_sm_preds, ann_sm_probs, ann_sm_recs = predict_churn(ann_sm_model, df_new_single, scaler, X_train_columns, model_type='ann', gemini_model=gemini_model)
        print(f"ANN + SMOTE Prediction: {ann_sm_preds[0]}, Probability: {ann_sm_probs[0]:.4f}, Recommendations: {ann_sm_recs}")

        # Predict with ANN + Focal Loss
        ann_fl_preds, ann_fl_probs, ann_fl_recs = predict_churn(ann_focal_loss_model, df_new_single, scaler, X_train_columns, model_type='ann', gemini_model=gemini_model)
        print(f"ANN + Focal Loss Prediction: {ann_fl_preds[0]}, Probability: {ann_fl_probs[0]:.4f}, Recommendations: {ann_fl_recs}")

    else:
        print("Models or scaler not loaded. Cannot perform prediction.")