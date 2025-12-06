# DriftOpt-UAV: Spray Drift Prediction and Spraying-Parameter Optimization Platform

A desktop application for predicting spray drift rate (%) for Plant-Protection Unmanned Aerial Vehicles (UAVs) using a trained XGBoost model with SHAP explanations.

## Features

- **Drift Rate Prediction**: Predict spray drift rate based on operational parameters
- **SHAP Visualization**: Generate SHAP force plots for each prediction to understand feature contributions
- **Interactive Interface**: User-friendly web-based interface built with Streamlit
- **Model Inference**: Standalone inference script for batch predictions

## Input Parameters

The application accepts the following input parameters:

1. **Operating altitude (m)**: UAV operating altitude in meters
2. **Operating speed (m/s)**: UAV operating speed in meters per second
3. **Wind speed (m/s)**: Wind speed in meters per second
4. **Wind deviation (°)**: Wind deviation angle in degrees
5. **Temperature (℃)**: Ambient temperature in Celsius
6. **Relative humidity (%)**: Relative humidity percentage
7. **Droplet size (μm)**: Droplet size in micrometers
8. **Distance (m)**: Distance from spray source in meters

## Output

- **Predicted Drift Rate (%)**: The predicted spray drift rate percentage
- **SHAP Force Plot (PNG)**: Visual explanation showing how each feature contributes to the prediction
- **SHAP Force Plot (HTML)**: Interactive HTML version of the SHAP plot
- **Feature Contributions Table**: Detailed breakdown of each feature's contribution

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure the model file `Xgboost.h5` is in the project directory.

## Usage

### Running the Desktop Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Inference Script

For programmatic use or batch predictions:

```python
from inference import DriftRatePredictor

# Initialize predictor
predictor = DriftRatePredictor("Xgboost.h5")

# Make a prediction
prediction, shap_values, input_data = predictor.predict(
    operating_altitude=4.0,
    operating_speed=7.0,
    wind_speed=1.9,
    wind_deviation=5.0,
    temperature=31.75,
    relative_humidity=39.3,
    droplet_size=250.0,
    distance=10.0
)

print(f"Predicted Drift Rate: {prediction:.2f} %")

# Generate SHAP plots
predictor.generate_shap_plot(shap_values, input_data, "shap_plot.png")
predictor.generate_shap_html(shap_values, input_data, "shap_plot.html")
```

## File Structure

- `app.py`: Main Streamlit application
- `inference.py`: Standalone inference script with DriftRatePredictor class
- `train_and_export_model.py`: Model training and export utilities
- `Xgboost.h5`: Trained XGBoost model file
- `requirements.txt`: Python package dependencies

## Model Information

The model is trained using XGBoost regression with hyperparameters optimized via Sparrow Search Algorithm (SSA). The model is saved in HDF5 format for efficient loading and inference.

## SHAP Explanations

Each prediction includes SHAP (SHapley Additive exPlanations) values that explain how each input feature contributes to the predicted drift rate. The SHAP plots are generated automatically for each prediction and can be downloaded as PNG or HTML files.

## Requirements

- Python 3.8+
- See `requirements.txt` for detailed package versions

## License

This project is for research and educational purposes.


