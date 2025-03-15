import os
import joblib
import torch
import requests
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from models.nn import PVNN  # Import NN model structure

def get_historical_weather(date, api_key):
    """Fetch historical weather data from OpenWeather for a past date."""
    start_timestamp = int(pd.Timestamp(date).timestamp())
    end_timestamp = start_timestamp + 86400  # 24 hours later

    url = f"https://history.openweathermap.org/data/2.5/history/city?lat={LATITUDE}&lon={LONGITUDE}&type=hour&start={start_timestamp}&end={end_timestamp}&appid={api_key}&units=metric"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "list" in data and data["list"]:
            return process_weather_data(data["list"], is_forecast=False)  # Pass is_forecast=False
    
    print(f"Error fetching historical data for {date} (status: {response.status_code})")
    return None

def get_forecast_weather(date, api_key):
    """Fetch weather forecast data from OpenWeather for a future date (up to 5 days ahead)."""
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LATITUDE}&lon={LONGITUDE}&appid={api_key}&units=metric"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "list" in data and data["list"]:
            return process_weather_data(data["list"], target_date=date, is_forecast=True)  # Explicitly pass is_forecast=True
    
    print(f"Error fetching forecast data for {date} (status: {response.status_code})")
    return None

def process_weather_data(weather_list, target_date=None, is_forecast=True):
    """Extract relevant weather features (average, min/max values) for a specific date.

    Args:
        weather_list (list): List of weather data entries.
        target_date (str, optional): Date for filtering forecast data (YYYY-MM-DD).
        is_forecast (bool): True if processing forecast data, False for historical data.

    Returns:
        tuple: Averaged weather features (temperature, humidity, wind_speed, rain, min_temp, max_temp, weather_condition)
    """
    filtered_data = []
    
    for entry in weather_list:
        if is_forecast:
            # Forecast API -> dt_txt is available
            dt_txt = entry["dt_txt"]
            if target_date and not dt_txt.startswith(target_date):
                continue
        else:
            # Historical API -> dt (timestamp) is available, convert to YYYY-MM-DD
            dt_txt = datetime.utcfromtimestamp(entry["dt"]).strftime('%Y-%m-%d')

        temp = entry["main"]["temp"]
        humidity = entry["main"]["humidity"]
        wind_speed = entry["wind"]["speed"]
        rain = entry.get("rain", {}).get("3h", 0)  # Rain data (in 3-hour intervals)
        min_temp = entry["main"]["temp_min"]
        max_temp = entry["main"]["temp_max"]
        weather = entry["weather"][0]["main"]

        filtered_data.append((temp, humidity, wind_speed, rain, min_temp, max_temp, weather))

    if not filtered_data:
        print(f"No weather data available for {target_date if target_date else 'selected date'}")
        return None

    # Compute averages and most common weather condition
    avg_temp = np.mean([x[0] for x in filtered_data])
    avg_humidity = np.mean([x[1] for x in filtered_data])
    avg_wind_speed = np.mean([x[2] for x in filtered_data])
    avg_rain = np.mean([x[3] for x in filtered_data])
    min_temp = min(x[4] for x in filtered_data)
    max_temp = max(x[5] for x in filtered_data)
    most_common_weather = max(set([x[6] for x in filtered_data]), key=[x[6] for x in filtered_data].count)

    return avg_temp, avg_humidity, avg_wind_speed, avg_rain, min_temp, max_temp, most_common_weather

def preprocess_weather_data(date, temperature, humidity, wind_speed, rain, min_temp, max_temp, weather):
    """Preprocess the weather data for model inference."""

    if None in [temperature, humidity, wind_speed, rain, min_temp, max_temp, weather]:
        raise ValueError("Incomplete weather data retrieved. Unable to proceed with inference.")

    # Extract weather categories from expected columns
    weather_categories = [col.replace("weather_", "") for col in expected_columns if col.startswith("weather_")]

    # If the weather category is not in the expected columns, raise an error
    if weather not in weather_categories:
        raise ValueError(f"Unknown weather category: {weather}. Expected one of {weather_categories}")

    # Define input features as a dictionary
    X_dict = {
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "rain": rain,
        "min_temperature": min_temp,
        "max_temperature": max_temp,
        **{f"weather_{cat}": 1 if cat == weather else 0 for cat in weather_categories}  # One-Hot Encoding
    }

    # Convert to DataFrame
    X_df = pd.DataFrame([X_dict])
    X_df = X_df.reindex(columns=expected_columns, fill_value=0)

    return X_df

# Argument Parser
parser = argparse.ArgumentParser(description="Solar Forecaster Inference")
parser.add_argument("--model", type=str, required=False, help="Model to use (xgboost, nn, rf)", default="xgboost")
parser.add_argument("--date", type=str, required=False, help="Date for prediction (YYYY-MM-DD)", default="2025-03-10")
parser.add_argument("--api_key", type=str, required=True, help="OpenWeather API key")
args = parser.parse_args()

# Assign arguments
model_choice = args.model.lower()
date_input = args.date
API_KEY = args.api_key

# OpenWeather API Configuration
LATITUDE = "xx"
LONGITUDE = "xx"

MODEL_SAVE_PATH = "saved_models"

# Validate Date Input
try:
    selected_date = datetime.strptime(date_input, "%Y-%m-%d")
    today = datetime.today()
    max_future_date = today + timedelta(days=5)

    if selected_date > max_future_date:
        raise ValueError("Date cannot be more than 5 days in the future.")

except ValueError as e:
    print(f"Invalid date format or range: {e}")
    exit()

# Fetch Weather Data
if selected_date < today:
    print(f"Fetching historical weather data for {date_input}...")
    weather_data = get_historical_weather(date_input, API_KEY)
else:
    print(f"Fetching forecast weather data for {date_input}...")
    weather_data = get_forecast_weather(date_input, API_KEY)

if weather_data is None:
    print("Weather data not available. Exiting...")
    exit()

# Load preprocessing data
preprocessing_data = joblib.load(os.path.join(MODEL_SAVE_PATH, "preprocessing.pkl"))
scaler = preprocessing_data["scaler"]
expected_columns = preprocessing_data["columns"]

# Preprocess Data
X_df = preprocess_weather_data(date_input, *weather_data)
X_scaled = scaler.transform(X_df)
print(f"Preprocessed data:\n{X_df}")

# Convert to NumPy array
X = X_df.to_numpy()

# Load Trained Model
model_filename = os.path.join(MODEL_SAVE_PATH, f"{model_choice}_model.pkl")

if not os.path.exists(model_filename):
    print(f"Trained model '{model_choice}' not found. Run 'train.py' first.")
    exit()

print(f"Using model: {model_choice}")

# Make Prediction
if model_choice == "nn":
    model = PVNN(input_size=X_scaled.shape[1])
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    prediction = model(X_tensor).detach().numpy().flatten()[0]
else:
    model = joblib.load(model_filename)
    prediction = model.predict(X)[0]

print(f"Predicted PV energy for {date_input}: {prediction:.4f} kWh")