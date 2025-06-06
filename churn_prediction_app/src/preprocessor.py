import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

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
        tuple: (X, y, scaler, smote_X, smote_y) if is_training=True,
               (X_processed, scaler) if is_training=False.
               smote_X and smote_y are None if is_training=False.
    """
    X = df.drop(columns=[target_column], errors='ignore')
    y = df[target_column] if target_column in df.columns else None

    # Encode categoricals (assuming get_dummies handles non-existent columns gracefully for prediction)
    X = pd.get_dummies(X, drop_first=True)

    if is_training:
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index) # Convert back to DataFrame

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_sm, y_sm = smote.fit_resample(X_scaled, y)
        X_sm = pd.DataFrame(X_sm, columns=X_scaled.columns) # Convert back to DataFrame
        return X_scaled, y, scaler, X_sm, y_sm, X.columns.tolist()
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for preprocessing new data.")
        if X_train_columns is None:
            raise ValueError("X_train_columns must be provided for preprocessing new data.")

        # Ensure columns match the training data
        for col in X_train_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[X_train_columns] # Reorder columns to match training data

        X_scaled = scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        return X_scaled, y, scaler, None, None, X.columns.tolist() # y is still needed for evaluation

def save_scaler(scaler, path):
    """Saves the trained StandardScaler."""
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")

def load_scaler(path):
    """Loads a trained StandardScaler."""
    try:
        scaler = joblib.load(path)
        print(f"Scaler loaded from {path}")
        return scaler
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {path}")
        return None
    except Exception as e:
        print(f"An error occurred loading scaler: {e}")
        return None

if __name__ == '__main__':
    # This block is for testing the preprocessor functions independently
    from data_loader import load_data # Assuming data_loader is in the same directory or accessible
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, 'data', 'customer_churn.xlsx') # Adjust as per your data path

    df = load_data(data_path)
    if df is not None:
        X_scaled, y, scaler, X_sm, y_sm, X_cols = preprocess_data(df, is_training=True)
        print("\n--- Preprocessing for Training ---")
        print("X_scaled shape:", X_scaled.shape)
        print("y shape:", y.shape)
        print("X_sm shape:", X_sm.shape)
        print("y_sm shape:", y_sm.shape)

        # Save the scaler for later use
        scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')
        if not os.path.exists(os.path.join(project_root, 'models')):
            os.makedirs(os.path.join(project_root, 'models'))
        save_scaler(scaler, scaler_path)

        # Example usage for prediction (using a new dummy df)
        print("\n--- Preprocessing for Prediction ---")
        df_new = pd.DataFrame({
            'CallFailure': [5], 'Complains': [0], 'SubscriptionLength': [30],
            'ChargeAmount': [1], 'SecondsUse': [1000], 'FrequencyUse': [20],
            'FrequencySMS': [10], 'DistinctCalls': [5], 'AgeGroup': [2],
            'TariffPlan': [1], 'Status': [1], 'Age': [25],
            'CustomerValue': [500.0]
        })
        # Load the saved scaler
        loaded_scaler = load_scaler(scaler_path)
        if loaded_scaler:
            X_new_scaled, _, _, _, _, _ = preprocess_data(df_new, is_training=False, scaler=loaded_scaler, X_train_columns=X_cols)
            print("X_new_scaled shape:", X_new_scaled.shape)
            print("X_new_scaled head:\n", X_new_scaled.head())
        else:
            print("Could not load scaler for prediction example.")