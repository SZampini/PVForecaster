# PVForecaster

PVForecaster is a tool designed to predict photovoltaic energy production using meteorological data and machine learning models. 
This project aims to provide forecasts to optimize energy consumption and improve the efficiency of photovoltaic installations.

## Features

- **OpenWeather API Integration**: Fetches real-time weather data from OpenWeather to enhance forecast accuracy.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/SZampini/PVForecaster.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd PVForecaster
   ```

3. **Install required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the forecasting process, run:

```bash
python inference.py --date YYYY-MM-DD --api_key YOUR_OPENWEATHER_API_KEY --latitude YOUR_LATITUDE --longitude YOUR_LONGITUDE
```

## Configuration

- **An OpenWeather API Key**: Register at [OpenWeather](https://openweathermap.org/) to obtain an API key.
- **Latitude and Longitude**: Specify the geographic coordinates for accurate predictions.
- **Model Type**: Choose between XGBoost, RF, and NN

## Models training

- **Pre-trained models**: Pre-trained models are provided, trained on a small dataset of approximately 300 samples.
- **Custom Training**: Users can train the model from scratch using their own dataset containing daily photovoltaic production data. The script allows downloading weather data and retraining the model accordingly.

## Contributing

Contributions to this project are welcome. You can contribute in several ways:

- **Reporting bugs**: Open an issue describing the problem encountered.
- **Suggesting new features**: Propose improvements or new functionalities.
- **Development**: Submit pull requests with fixes or new features.

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

---

This README provides a general overview of the PVForecaster project, including its purpose, features, installation and usage instructions, configuration, contribution guidelines, and licensing information.

