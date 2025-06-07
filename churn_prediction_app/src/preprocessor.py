import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
# IMPORTANT: There should be NO 'from preprocessor import ...' line here!

def preprocess_data(df, target_column='Churn', is_training=True, scaler=None, X_train_columns=None):
    """
    Preprocesses the data by separating features and target, encoding categoricals,
    and optionally scaling features and applying SMOTE.
    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        is_training (bool): If True, fits and transforms scaler/SMOTE. If False,
                            only transforms using provided scaler.
        scaler (StandardScaler, optional): Fitted scaler object for transforming new data.
        X_train_columns (list, optional): List of columns from training data to ensure
                                           consistent columns for new data.
    Returns:
        tuple: (X_processed, y, scaler, smote_X, smote_y, X_train_columns_final)
               X_processed will be scaled and/or dummified.
               For prediction, smote_X and smote_y will be None.
    """
    X = df.drop(columns=[target_column], errors='ignore')
    y = df[target_column] if target_column in df.columns else None

    # Encode categoricals (assuming get_dummies handles non-existent columns gracefully for prediction)
    # Ensure all categorical columns are treated consistently
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Apply one-hot encoding
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


    if is_training:
        # Fit scaler on training data
        scaler = StandardScaler()
        X_scaled_np = scaler.fit_transform(X_encoded)
        
        # Convert X_scaled_np back to DataFrame to preserve column names
        X_scaled = pd.DataFrame(X_scaled_np, columns=X_encoded.columns, index=X_encoded.index)

        # Store the columns of the processed training data
        X_train_columns_final = X_scaled.columns.tolist()

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        # SMOTE expects numpy arrays, so convert X_scaled back to numpy for SMOTE
        X_sm, y_sm = smote.fit_resample(X_scaled_np, y)
        
        # Convert SMOTE output back to DataFrame, crucial for XGBoost and ANN
        X_sm = pd.DataFrame(X_sm, columns=X_train_columns_final)


        return X_sm, y_sm, scaler, X_sm, y_sm, X_train_columns_final
    else: # is_training is False (prediction mode)
        if X_train_columns is None:
            raise ValueError("X_train_columns must be provided for preprocessing in prediction mode.")
        
        # Ensure all training columns are present, fill with 0 if not, and reorder
        # This handles cases where new data might miss some columns or have extra ones
        X_padded = X_encoded.reindex(columns=X_train_columns, fill_value=0)
        
        # Ensure the order is exactly as in X_train_columns
        X_padded = X_padded[X_train_columns] # Reorder to match the training columns

        if scaler:
            X_scaled_np = scaler.transform(X_padded)
            # Convert back to DataFrame to maintain column names and consistency
            X_processed = pd.DataFrame(X_scaled_np, columns=X_train_columns, index=X_padded.index)
        else:
            X_processed = X_padded # No scaling, just padding

        # Return 6 values for consistency with training mode, even if some are None
        return X_processed, y, scaler, None, None, X_train_columns # X_train_columns is now the 6th return value

def save_scaler(scaler, path):
    """Saves the fitted scaler."""
    joblib.dump(scaler, path)

def load_scaler(path):
    """Loads a fitted scaler."""
    return joblib.load(path)

# Example usage for testing (only runs if preprocessor.py is executed directly)
if __name__ == '__main__':
    print("Running preprocessor.py as a standalone script for testing...")

    # Create dummy data for testing
    df = pd.DataFrame({
        'CallFailure': [5, 8, 2],
        'Complains': [0, 1, 0],
        'SubscriptionLength': [12, 24, 6],
        'ChargeAmount': [50, 75, 30],
        'SecondsUse': [1000, 2000, 500],
        'FrequencyUse': [20, 40, 10],
        'FrequencySMS': [10, 20, 5],
        'DistinctCalls': [5, 10, 3],
        'AgeGroup': ['Group2', 'Group3', 'Group1'], # Example of categorical
        'TariffPlan': ['PlanA', 'PlanB', 'PlanA'],   # Example of categorical
        'Status': ['Active', 'Inactive', 'Active'],   # Example of categorical
        'Age': [30, 45, 25],
        'CustomerValue': [500.0, 1200.0, 200.0],
        'Churn': [0, 1, 0] # Target variable
    })

    # --- Test Training Preprocessing ---
    print("\n--- Preprocessing for Training (Preprocessor Test) ---")
    X_train_processed, y_train_processed, trained_scaler, X_sm, y_sm, X_train_cols = preprocess_data(df, is_training=True)

    print("X_train_processed shape:", X_train_processed.shape)
    print("y_train_processed shape:", y_train_processed.shape)
    print("X_train_columns_final:", X_train_cols)
    print("X_train_processed (first 3 rows):\n", X_train_processed.head(3))
    print("y_train_processed (first 3 rows):\n", y_train_processed.head(3))

    # Save scaler for prediction test
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, 'scaler_test.pkl')
    joblib.dump(trained_scaler, scaler_path)
    print(f"Test scaler saved to {scaler_path}")


    # --- Test Prediction Preprocessing ---
    print("\n--- Preprocessing for Prediction (Preprocessor Test) ---")
    df_new = pd.DataFrame({
        'CallFailure': [5], 'Complains': [0], 'SubscriptionLength': [30],
        'ChargeAmount': [1], 'SecondsUse': [1000], 'FrequencyUse': [20],
        'FrequencySMS': [10], 'DistinctCalls': [5], 'AgeGroup': ['Group2'], # Match categorical format
        'TariffPlan': ['PlanA'], 'Status': ['Active'], 'Age': [25],
        'CustomerValue': [500.0]
    })

    loaded_scaler = joblib.load(scaler_path)

    X_new_processed, _, _, _, _, _ = preprocess_data(
        df_new, is_training=False, scaler=loaded_scaler, X_train_columns=X_train_cols
    )

    print("X_new_processed shape:", X_new_processed.shape)
    print("X_new_processed (first row):\n", X_new_processed.head(1))
    print("Type of X_new_processed:", type(X_new_processed)) # Should be <class 'pandas.DataFrame'>
    assert isinstance(X_new_processed, pd.DataFrame), "X_new_processed is not a DataFrame!"
    print("Prediction preprocessing test successful!")

    # Clean up test scaler
    # os.remove(scaler_path)
    # print(f"Cleaned up test scaler: {scaler_path}")