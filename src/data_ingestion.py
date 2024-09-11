import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(file_path):
    """
    Load the raw data from a CSV file.
    :param file_path: Path to the CSV file containing raw data
    :return: Pandas DataFrame containing the raw data
    """
    data = pd.read_csv(file_path)
    print(f"Data loaded successfully from {file_path}")
    return data

def preprocess_data(data, scaler_path='models/scaler.pkl'):
    """
    Preprocess the data by scaling 'Amount' and 'Time' features.
    :param data: Raw Pandas DataFrame
    :return: Preprocessed Pandas DataFrame
    """
    # Check for missing values
    if data.isnull().sum().any():
        print("Warning: Missing values detected. Consider handling them before further processing.")
    
    # Scale 'Amount' and 'Time'
    scaler = StandardScaler()
    
    data['scaled_amount'] = scaler.fit_transform(data[['Amount']])
    data['scaled_time'] = scaler.fit_transform(data[['Time']])
    
    # Drop the original 'Amount' and 'Time' columns
    data = data.drop(['Amount', 'Time'], axis=1)
    
    # Save the scaler to a .pkl file
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    print("Data preprocessing completed: 'Amount' and 'Time' features scaled.")
    
    return data

def load_and_preprocess(file_path):
    """
    Load and preprocess data in a single function.
    :param file_path: Path to the CSV file containing raw data
    :return: Preprocessed Pandas DataFrame
    """
    data = load_data(file_path)
    processed_data = preprocess_data(data)
    return processed_data

if __name__ == "__main__":
    # Example usage: loading and preprocessing the data
    processed_data = load_and_preprocess('data/raw/creditcard.csv')
    # Save the processed data (optional)
    processed_data.to_csv('data/processed/creditcard_preprocessed.csv', index=False)
    print("Preprocessed data saved to 'data/processed/creditcard_preprocessed.csv'.")
