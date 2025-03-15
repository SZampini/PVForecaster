import os
import pandas as pd
import joblib
import torch
import importlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Define the directory to save trained models
MODEL_SAVE_PATH = "./saved_models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Load the dataset
df = pd.read_csv("./data/pv_weather.csv")

def preprocess_data(df):
    """Preprocess dataset: feature extraction, encoding, and scaling."""
    df = df.copy()
    print(f"Original dataset example:\n  {df.head(1)}")

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
    
    return X, y

# Split dataset into train/test sets
X, y = preprocess_data(df)
print(f"Features example:\n  {X.head(1)}")
print(f"Labels example:\n  {y.head(1)}")
print(f"Dataset dimension:\n  {X.shape[0]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
#scaler = StandardScaler()
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessing data
expected_columns = X.columns.tolist()
joblib.dump({"scaler": scaler, "columns": expected_columns}, os.path.join(MODEL_SAVE_PATH, "preprocessing.pkl"))

# Prompt user to select a model
model_choice = input("Select a model (xgboost, nn, rf): ").strip().lower()

model_modules = {
    'xgboost': 'models.xgboost',
    'nn': 'models.nn',
    'rf': 'models.rf'
}

if model_choice not in model_modules:
    raise ValueError("Invalid model choice. Select from 'xgboost', 'nn', or 'rf'.")

# Dynamically import the selected model's module
model_module = importlib.import_module(model_modules[model_choice])
model = model_module.train_model(X_train, y_train, X_train_scaled, X_test, y_test, X_test_scaled)

# Save the trained model
model_filename = os.path.join(MODEL_SAVE_PATH, f"{model_choice}_model.pkl")

if model_choice == "nn":
    torch.save(model.state_dict(), model_filename)  # Save PyTorch model
else:
    joblib.dump(model, model_filename)  # Save other models using joblib

print(f"Model '{model_choice}' saved to {model_filename}")