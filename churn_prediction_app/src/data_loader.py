import pandas as pd
import os

def load_data(data_path):
    """
    Loads the dataset from the specified path, typically for training column inference.
    Args:
        data_path (str): The path to the CSV data file.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        # Try reading with a common alternative encoding like 'latin1'
        df = pd.read_csv(data_path, encoding='latin1')
        print(f"Data for column inference loaded successfully from: {data_path}")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None
    except UnicodeDecodeError:
        print(f"Error: Could not decode data from {data_path} with 'latin1' encoding. Trying 'ISO-8859-1'...")
        try:
            df = pd.read_csv(data_path, encoding='ISO-8859-1')
            print(f"Data for column inference loaded successfully with 'ISO-8859-1' encoding from: {data_path}")
            return df
        except UnicodeDecodeError:
            print(f"Error: Could not decode data from {data_path} with 'ISO-8859-1' encoding. Trying 'cp1252'...")
            try:
                df = pd.read_csv(data_path, encoding='cp1252')
                print(f"Data for column inference loaded successfully with 'cp1252' encoding from: {data_path}")
                return df
            except Exception as e:
                print(f"Error: Could not load data from {data_path} with 'cp1252' encoding. Error: {e}")
                return None
    except Exception as e:
        print(f"An unexpected error occurred while loading data from {data_path}: {e}")
        return None

def load_customer_identifiers_data(data_path):
    """
    Loads customer data with identifiers (name, phone, address) from the specified path.
    Args:
        data_path (str): The path to the CSV data file.
    Returns:
        pd.DataFrame: The loaded DataFrame, or None if an error occurs.
    """
    try:
        # Assuming customer_data_with_identifiers.csv is encoded as UTF-8 or latin1
        df = pd.read_csv(data_path, encoding='latin1')
        print(f"Customer identifiers data loaded successfully from: {data_path}")
        return df
    except FileNotFoundError:
        print(f"Error: Customer identifiers data file not found at {data_path}")
        return None
    except Exception as e:
        print(f"Error loading customer identifiers data from {data_path}: {e}")
        return None

# Example usage for testing (only runs if data_loader.py is executed directly)
if __name__ == '__main__':
    print("Running data_loader.py as a standalone script for testing...")

    # Define a dummy path for testing the main load_data function
    project_root_test = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dummy_churn_data_path = os.path.join(project_root_test, 'data', 'customer_churn.xlsx - Sheet1.csv')
    
    # Create a dummy CSV file for testing if it doesn't exist
    os.makedirs(os.path.join(project_root_test, 'data'), exist_ok=True)
    if not os.path.exists(dummy_churn_data_path):
        dummy_df_churn = pd.DataFrame({
            'CallFailure': [8,0,10], 'Complains': [0,0,0], 'SubscriptionLength': [38,39,37],
            'ChargeAmount': [0,0,0], 'SecondsUse': [4370,318,2453], 'FrequencyUse': [71,5,60],
            'FrequencySMS': [5,7,359], 'DistinctCalls': [17,6,20], 'AgeGroup': ['Group2','Group3','Group2'],
            'TariffPlan': ['PlanA','PlanB','PlanA'], 'Status': ['Active','Inactive','Active'], 'Age': [30,45,28],
            'CustomerValue': [800.0, 1200.0, 700.0], 'Churn': [0, 1, 0]
        })
        dummy_df_churn.to_csv(dummy_churn_data_path, index=False)
        print(f"Dummy customer_churn.xlsx - Sheet1.csv created at {dummy_churn_data_path}")

    df_churn_test = load_data(dummy_churn_data_path)
    if df_churn_test is not None:
        print("Loaded churn data head:\n", df_churn_test.head())

    # Define a dummy path for testing the load_customer_identifiers_data function
    dummy_identifiers_data_path = os.path.join(project_root_test, 'data', 'customer_data_with_identifiers.csv')

    # Create a dummy CSV file for testing if it doesn't exist
    if not os.path.exists(dummy_identifiers_data_path):
        dummy_identifiers_df = pd.DataFrame({
            'CustomerID': [1,2,3],
            'CustomerName': ['Alice Smith', 'Bob Johnson', 'Charlie Brown'],
            'PhoneNumber': ['07712345678', '07898765432', '07955512345'],
            'Address': ['1 High St, London', '2 Oak Ave, Manchester', '3 Pine Ln, Birmingham'],
            'CallFailure': [5, 8, 2], 'Complains': [0, 1, 0], 'SubscriptionLength': [12, 24, 6],
            'ChargeAmount': [50, 75, 30], 'SecondsUse': [1000, 2000, 500], 'FrequencyUse': [20, 40, 10],
            'FrequencySMS': [10, 20, 5], 'DistinctCalls': [5, 10, 3], 'AgeGroup': ['Group2', 'Group3', 'Group1'],
            'TariffPlan': ['PlanA', 'PlanB', 'PlanA'], 'Status': ['Active', 'Inactive', 'Active'], 'Age': [30, 45, 25],
            'CustomerValue': [500.0, 1200.0, 200.0], 'Churn': [0, 1, 0]
        })
        dummy_identifiers_df.to_csv(dummy_identifiers_data_path, index=False)
        print(f"Dummy customer_data_with_identifiers.csv created at {dummy_identifiers_data_path}")

    df_identifiers_test = load_customer_identifiers_data(dummy_identifiers_data_path)
    if df_identifiers_test is not None:
        print("Loaded identifiers data head:\n", df_identifiers_test.head())