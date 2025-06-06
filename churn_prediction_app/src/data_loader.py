import pandas as pd
import os

def load_data(data_path):
    """
    Loads the dataset from the specified path.
    Args:
        data_path (str): The path to the Excel data file.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_excel(data_path)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None

if __name__ == '__main__':
    # Example usage (assuming DATA_PATH is defined somewhere, e.g., in config.py)
    # For this example, let's create a dummy file or adjust the path
    # You would typically have a config.py to define DATA_PATH
    # For now, let's assume it's in the parent directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dummy_data_path = os.path.join(project_root,'customer_churn.xlsx') # Adjust as per your actual data path

    # Create a dummy Excel file for testing if it doesn't exist
    if not os.path.exists(os.path.join(project_root, 'data')):
        os.makedirs(os.path.join(project_root, 'data'))
    if not os.path.exists(dummy_data_path):
        dummy_df = pd.DataFrame({
            'CallFailure': [8,0,10], 'Complains': [0,0,0], 'SubscriptionLength': [38,39,37],
            'ChargeAmount': [0,0,0], 'SecondsUse': [4370,318,2453], 'FrequencyUse': [71,5,60],
            'FrequencySMS': [5,7,359], 'DistinctCalls': [17,4,24], 'AgeGroup': [3,2,3],
            'TariffPlan': [1,1,1], 'Status': [1,2,1], 'Age': [30,25,30],
            'CustomerValue': [197.64,46.035,1536.52], 'Churn': [0,0,1]
        })
        dummy_df.to_excel(dummy_data_path, index=False)
        print(f"Dummy data created at {dummy_data_path}")

    df = load_data(dummy_data_path)
    if df is not None:
        print(df.head())