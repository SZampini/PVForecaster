import pandas as pd
import requests
import time
import os

# Set your OpenWeather API key
API_KEY = "xx"  # Replace with your API key
LATITUDE = "xx"  # Latitude
LONGITUDE = "xx"   # Longitude

# Function to fetch weather data for a specific date
def get_weather_data(date):
    start_timestamp = int(pd.Timestamp(date).timestamp())
    end_timestamp = start_timestamp + 86400  # Add 24 hours (1 day)
    url = f"https://history.openweathermap.org/data/2.5/history/city?lat={LATITUDE}&lon={LONGITUDE}&type=hour&start={start_timestamp}&end={end_timestamp}&appid={API_KEY}&units=metric"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "list" in data and data["list"]:
            avg_temp = sum(entry["main"]["temp"] for entry in data["list"]) / len(data["list"])
            avg_humidity = sum(entry["main"]["humidity"] for entry in data["list"]) / len(data["list"])
            avg_wind_speed = sum(entry["wind"]["speed"] for entry in data["list"]) / len(data["list"])
            avg_rain = sum(entry.get("rain", {}).get("1h", 0) for entry in data["list"]) / len(data["list"])
            min_temp = min(entry["main"]["temp_min"] for entry in data["list"])
            max_temp = max(entry["main"]["temp_max"] for entry in data["list"])
            weather_conditions = [entry["weather"][0]["main"] for entry in data["list"]]
            most_common_weather = max(set(weather_conditions), key=weather_conditions.count)
            return avg_temp, avg_humidity, avg_wind_speed, avg_rain, min_temp, max_temp, most_common_weather
    
    print(f"No data found for {date} or request failed with status code {response.status_code}")
    return None, None, None, None, None, None, None

# Load the CSV file
test = False

if not test:
    csv_file = "./data/raw/pv_merged.csv"  # Input CSV file path
    output_file = "./data/pv_weather.csv"  # Output CSV file path
else:
    csv_file = "./data/raw/pv_merged_test.csv"
    output_file = "./data/pv_weather_test.csv"
    
df = pd.read_csv(csv_file)

data_list = []

# Iterate through all dates to fetch weather data
for index, row in df.iterrows():
    date = row["date"]
    pv_energy = row["pv_energy"]
    temp, humidity, wind_speed, rain, min_temp, max_temp, weather = get_weather_data(date)
    if temp is not None:
        data_list.append([date, pv_energy, temp, humidity, wind_speed, rain, min_temp, max_temp, weather])
    
    print(f"Processed {date}: Temp={temp}, Min Temp={min_temp}, Max Temp={max_temp}, Humidity={humidity}, Wind Speed={wind_speed}, Rain={rain}, Weather={weather}")
    #time.sleep(1)  # Sleep 1 second to avoid exceeding API rate limits

# Create a new DataFrame with the obtained data
df_weather = pd.DataFrame(data_list, columns=["date", "pv_energy", "temperature", "humidity", "wind_speed", "rain", "min_temperature", "max_temperature", "weather"])

# Save the new CSV
df_weather.to_csv(output_file, index=False)
print(f"Weather data saved in {output_file}")