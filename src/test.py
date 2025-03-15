import os
import joblib
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
from models.nn import PVNN  # Import NN class for model loading

# Define paths
MODEL_SAVE_PATH = "./saved_models"
TEST_DATA_PATH = "./data/pv_weather_test.csv"

# Check if the test dataset exists
if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(f"Test dataset not found at {TEST_DATA_PATH}")

# Load test dataset
df_test = pd.read_csv(TEST_DATA_PATH)

def preprocess_data(df):
    """Preprocess dataset for testing: feature extraction, encoding, and scaling."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['weather'], prefix='weather', dtype=int)
    weather_columns = [col for col in df.columns if col.startswith('weather_')]

    # Select relevant features
    #X = df[['temperature', 'humidity', 'wind_speed', 'rain', 'min_temperature', 'max_temperature', 'weather', 'month', 'day']]
    X = df[['temperature', 'humidity', 'wind_speed', 'rain', 'min_temperature', 'max_temperature'] + weather_columns]
    y = df[['pv_energy']]

    # Add missing columns
    for col in expected_columns:
        if col not in X.columns:
            X[col] = 0

    # Reorder columns
    X = X[expected_columns]
    
    return X, y

# Load preprocessing data
preprocessing_data = joblib.load(os.path.join(MODEL_SAVE_PATH, "preprocessing.pkl"))
scaler = preprocessing_data["scaler"]
expected_columns = preprocessing_data["columns"]

# Process test data
X_test, y_test = preprocess_data(df_test)
X_test_scaled = scaler.fit_transform(X_test)

# Define models to evaluate
models = ["xgboost", "nn", "rf"]
results = {}

for model_name in models:
    model_filename = os.path.join(MODEL_SAVE_PATH, f"{model_name}_model.pkl")

    if not os.path.exists(model_filename):
        print(f"Skipping {model_name}: Model not found in {MODEL_SAVE_PATH}")
        continue

    print(f"Evaluating model: {model_name}")

    # Load the model and make predictions
    if model_name == "nn":
        model = PVNN(input_size=X_test_scaled.shape[1])
        model.load_state_dict(torch.load(model_filename))  # Load PyTorch model
        model.eval()
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        predictions = model(X_tensor).detach().numpy().flatten()
    else:
        model = joblib.load(model_filename)  # Load other models trained with joblib
        predictions = model.predict(X_test)

    # Compute Mean Absolute Error
    mae = mean_absolute_error(y_test, predictions)
    results[model_name] = mae
    print(f"Model: {model_name} | MAE: {mae:.4f}")


