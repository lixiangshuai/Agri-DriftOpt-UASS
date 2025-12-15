# 🚁 Agri-DriftOpt-UASS: Spray Drift Prediction & Spraying-Parameter Optimization Platform

An explainable artificial intelligence–based decision-support tool for spray drift control in unmanned aerial spraying systems (UASS). A desktop application for predicting spray drift rate (%) for Plant-Protection Unmanned Aerial Vehicles (UAVs) using a trained XGBoost model with SHAP explanations.

## ✨ Features

- **Drift Rate Prediction**: Predict spray drift rate based on operational parameters
- **SHAP Visualization**: Generate SHAP force plots for each prediction to understand feature contributions
- **Parameter Optimization Recommendations**: Automatic suggestions for adjusting parameters to reduce drift rate based on SHAP values
- **Feature Contributions Analysis**: Detailed breakdown of how each parameter affects the prediction
- **Interactive Interface**: User-friendly web-based interface built with Streamlit
- **Model Inference**: Standalone inference script for batch predictions

## 📝 Input Parameters

The application accepts the following input parameters:

1. **Operating altitude (m)**: UAV operating altitude in meters
2. **Operating speed (m/s)**: UAV operating speed in meters per second
3. **Wind speed (m/s)**: Wind speed in meters per second
4. **Wind deviation (°)**: Wind deviation angle in degrees
5. **Temperature (℃)**: Ambient temperature in Celsius
6. **Relative humidity (%)**: Relative humidity percentage
7. **Droplet size (μm)**: Droplet size in micrometers
8. **Distance (m)**: Distance from spray source in meters

## 📊 Output

- **Predicted Drift Rate (%)**: The predicted spray drift rate percentage displayed prominently
- **SHAP Force Plot (HTML)**: Interactive HTML visualization showing how each feature contributes to the prediction (automatically displayed in the interface)
- **SHAP Force Plot Download**: Downloadable HTML version of the SHAP plot
- **SHAP Plot Interpretation Guide**: Explanation of how to interpret the SHAP plot (green parameters increase drift, yellow parameters decrease drift)
- **Parameter Adjustment Recommendations**: Automatic suggestions for parameters with positive SHAP values that should be adjusted to reduce drift rate
- **Feature Contributions Table**: Detailed table showing each feature's SHAP value, input value, and contribution percentage (expandable section)

## 📦 Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure the model file `Xgboost.h5` is in the project directory.

## 🚀 Usage

### Running the Desktop Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

**Usage Steps:**
1. Fill in the 8 input parameters in the form
2. Click the **"🔮 Drift Prediction and Parameter Optimization"** button
3. View the predicted drift rate and detailed analysis results
4. Review SHAP visualizations and parameter optimization recommendations
5. Download SHAP plots if needed

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

## 📁 File Structure

- `app.py`: Main Streamlit application with full UI and parameter optimization features
- `inference.py`: Standalone inference script with DriftRatePredictor class
- `train_and_export_model.py`: Model training and export utilities
- `Xgboost.h5`: Trained XGBoost model file
- `requirements.txt`: Python package dependencies
- `T25.xlsx`: Training dataset (for model retraining)
- `run_app.bat`: Windows batch script to launch the application
- `run_app.sh`: Linux/Mac shell script to launch the application

## 🤖 Model Information

The model is trained using XGBoost regression with hyperparameters optimized via Sparrow Search Algorithm (SSA). The model is saved in HDF5 format for efficient loading and inference.

## 📈 SHAP Explanations

Each prediction includes SHAP (SHapley Additive exPlanations) values that explain how each input feature contributes to the predicted drift rate. The platform provides:

- **Interactive SHAP Force Plot**: Automatically displayed in the interface, showing feature contributions with color coding:
  - **Green parameters**: Push the predicted drift rate higher (recommended to adjust)
  - **Yellow parameters**: Push the drift rate lower (generally no adjustment needed)

- **Parameter Optimization Recommendations**: The platform automatically identifies parameters with positive SHAP values (SHAP > 0) and provides specific adjustment suggestions, such as:
  - Reducing operating altitude or speed
  - Delaying operation until wind speed decreases
  - Adjusting flight heading to reduce crosswind influence
  - Selecting larger droplet sizes
  - Operating within suitable temperature and humidity ranges

- **Feature Contributions Table**: Expandable table showing:
  - Feature names
  - SHAP values (positive/negative contributions)
  - Input feature values
  - Contribution percentages
  - Summary statistics (total positive/negative contributions, base value, predicted value)

**Note**: The Distance feature is automatically filtered out from SHAP visualizations and recommendations, as it's a measurement parameter rather than an operational control parameter. Additionally, the platform applies special handling for Temperature and Relative Humidity SHAP values to provide more intuitive recommendations.

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for detailed package versions

## 🏢 Affiliation

State Key Laboratory for Biology of Plant Diseases and Insect Pests, Institute of Plant Protection, Chinese Academy of Agricultural Sciences, Beijing 100193, China

## 📄 License

This project is for research and educational purposes.



