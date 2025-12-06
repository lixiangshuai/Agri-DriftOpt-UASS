"""
Inference Script for XGBoost Drift Rate Prediction Model
Loads the trained model from H5 file and provides prediction functionality
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import h5py
import json
import tempfile
import os
from train_and_export_model import load_xgboost_from_h5


class DriftRatePredictor:
    """Drift Rate Prediction Model Inference Class"""
    
    def __init__(self, model_path="Xgboost.h5"):
        """
        Initialize the predictor with a trained model
        
        Parameters:
        model_path: Path to the H5 model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from {model_path}...")
        self.model = load_xgboost_from_h5(model_path)
        
        # Get feature names
        # Try to get from _feature_names (our custom attribute) first
        if hasattr(self.model, '_feature_names'):
            self.feature_names = list(self.model._feature_names)
        elif hasattr(self.model, 'feature_names_in_'):
            self.feature_names = list(self.model.feature_names_in_)
        elif hasattr(self.model, 'get_booster'):
            try:
                booster = self.model.get_booster()
                if hasattr(booster, 'feature_names') and booster.feature_names:
                    self.feature_names = list(booster.feature_names)
                else:
                    raise AttributeError
            except:
                # Default feature names
                self.feature_names = [
                    'Operating altitude(m)',
                    'Operating speed(m/s)',
                    'Wind speed(m/s)',
                    'Wind deviation(°)',
                    'Temperature(℃)',
                    'Relative humidity(%)',
                    'Droplet size(μm)',
                    'Distance'
                ]
        else:
            # Default feature names
            self.feature_names = [
                'Operating altitude(m)',
                'Operating speed(m/s)',
                'Wind speed(m/s)',
                'Wind deviation(°)',
                'Temperature(℃)',
                'Relative humidity(%)',
                'Droplet size(μm)',
                'Distance'
            ]
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        print("Model loaded successfully!")
    
    def predict(self, 
                operating_altitude,
                operating_speed,
                wind_speed,
                wind_deviation,
                temperature,
                relative_humidity,
                droplet_size,
                distance):
        """
        Predict drift rate for given input parameters
        
        Parameters:
        operating_altitude: Operating altitude (m)
        operating_speed: Operating speed (m/s)
        wind_speed: Wind speed (m/s)
        wind_deviation: Wind deviation (°)
        temperature: Temperature (℃)
        relative_humidity: Relative humidity (%)
        droplet_size: Droplet size (μm)
        distance: Distance (m)
        
        Returns:
        prediction: Predicted drift rate (%)
        shap_values: SHAP values for explanation
        input_data: Formatted input data as DataFrame
        """
        # Create input DataFrame
        input_data = pd.DataFrame({
            self.feature_names[0]: [operating_altitude],
            self.feature_names[1]: [operating_speed],
            self.feature_names[2]: [wind_speed],
            self.feature_names[3]: [wind_deviation],
            self.feature_names[4]: [temperature],
            self.feature_names[5]: [relative_humidity],
            self.feature_names[6]: [droplet_size],
            self.feature_names[7]: [distance]
        })
        
        # Ensure feature order matches model
        input_data = input_data[self.feature_names]
        
        # Make prediction
        prediction = self.model.predict(input_data)[0]
        
        # Compute SHAP values
        shap_values = self.explainer(input_data)
        
        return prediction, shap_values, input_data
    
    def generate_shap_plot(self, shap_values, input_data, output_path="shap_plot.png", 
                          remove_distance=True):
        """
        Generate SHAP force plot and save as PNG
        
        Parameters:
        shap_values: SHAP values object
        input_data: Input data DataFrame
        output_path: Path to save the plot
        remove_distance: Whether to remove Distance feature from plot
        """
        try:
            if remove_distance and 'Distance' in self.feature_names:
                distance_index = self.feature_names.index('Distance')
                
                # Remove Distance feature
                shap_values_filtered = shap_values.values[0].copy()
                shap_values_filtered = np.delete(shap_values_filtered, distance_index)
                
                input_data_filtered = input_data.iloc[0].copy()
                input_data_filtered = input_data_filtered.drop(labels=['Distance'])
                
                # Generate SHAP force plot
                plt.rcParams.update({'font.size': 8})
                plt.figure(figsize=(15, 6))
                shap.force_plot(
                    self.explainer.expected_value, 
                    shap_values_filtered, 
                    input_data_filtered, 
                    matplotlib=True
                )
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
            else:
                # Use all features
                plt.rcParams.update({'font.size': 8})
                plt.figure(figsize=(15, 6))
                shap.force_plot(
                    self.explainer.expected_value,
                    shap_values.values[0],
                    input_data.iloc[0],
                    matplotlib=True
                )
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"SHAP plot saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error generating SHAP plot: {e}")
            return False
    
    def generate_shap_html(self, shap_values, input_data, output_path="shap_plot.html",
                           remove_distance=True):
        """
        Generate SHAP force plot and save as HTML
        
        Parameters:
        shap_values: SHAP values object
        input_data: Input data DataFrame
        output_path: Path to save the HTML file
        remove_distance: Whether to remove Distance feature from plot
        """
        try:
            # Define high contrast color scheme
            color_positive_html = '#2ca02c'  # 绿色 - 正贡献
            color_negative_html = '#ffd700'  # 黄色 - 负贡献
            
            if remove_distance and 'Distance' in self.feature_names:
                distance_index = self.feature_names.index('Distance')
                
                # Remove Distance feature
                shap_values_filtered = shap_values.values[0].copy()
                shap_values_filtered = np.delete(shap_values_filtered, distance_index)
                
                input_data_filtered = input_data.iloc[0].copy()
                input_data_filtered = input_data_filtered.drop(labels=['Distance'])
                
                # Generate SHAP force plot with custom colors
                shap_plot = shap.force_plot(
                    self.explainer.expected_value,
                    shap_values_filtered,
                    input_data_filtered,
                    plot_cmap=[color_positive_html, color_negative_html]
                )
                shap.save_html(output_path, shap_plot)
            else:
                # Use all features
                shap_plot = shap.force_plot(
                    self.explainer.expected_value,
                    shap_values.values[0],
                    input_data.iloc[0],
                    plot_cmap=[color_positive_html, color_negative_html]
                )
                shap.save_html(output_path, shap_plot)
            
            print(f"SHAP HTML saved to: {output_path}")
            return True
        except Exception as e:
            print(f"Error generating SHAP HTML: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = DriftRatePredictor("Xgboost.h5")
    
    # Example prediction
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
    
    print(f"\nPredicted Drift Rate: {prediction:.2f} %")
    
    # Generate SHAP plots
    predictor.generate_shap_plot(shap_values, input_data, "shap_plot.png")
    predictor.generate_shap_html(shap_values, input_data, "shap_plot.html")

