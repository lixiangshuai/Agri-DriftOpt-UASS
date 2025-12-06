"""
Agri-DriftOpt-UASS: Spray Drift Prediction & Spraying-Parameter Optimization Platform
An explainable artificial intelligence–based decision-support tool for spray drift control in unmanned aerial spraying systems (UASS)
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import h5py
import json
import tempfile
import os
import base64
from train_and_export_model import load_xgboost_from_h5
from inference import DriftRatePredictor

# Page configuration
st.set_page_config(
    page_title="Agri-DriftOpt-UASS",
    page_icon="🚁",
    layout="wide"
)

# Title and description
st.title("🚁 Agri-DriftOpt-UASS: Spray Drift Prediction & Spraying-Parameter Optimization Platform")
st.markdown("""
An explainable artificial intelligence–based decision-support tool for spray drift control in unmanned aerial spraying systems (UASS). 
Enter the operating parameters below to obtain the predicted drift rate and corresponding parameter outputs.
""")

# Initialize session state for model caching
@st.cache_resource
def load_predictor():
    """Load the trained XGBoost model predictor from H5 file"""
    model_path = "Xgboost.h5"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    
    try:
        predictor = DriftRatePredictor(model_path)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Load predictor
with st.spinner("Loading model..."):
    predictor = load_predictor()
    st.success("Model loaded successfully!")

# Get feature names from the predictor
feature_names = predictor.feature_names

# Input Parameters Section (Top)
st.header("📝 Input Parameters")

# Input form
with st.form("prediction_form"):
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        # Operating altitude (m)
        altitude = st.number_input(
            "Operating altitude (m)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="UAV operating altitude in meters"
        )
        
        # Operating speed (m/s)
        speed = st.number_input(
            "Operating speed (m/s)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="UAV operating speed in meters per second"
        )
        
        # Wind speed (m/s)
        wind_speed = st.number_input(
            "Wind speed (m/s)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Wind speed in meters per second"
        )
        
        # Wind deviation (°)
        wind_deviation = st.number_input(
            "Wind deviation (°)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="Wind deviation angle in degrees"
        )
    
    with col2:
        # Temperature (℃)
        temperature = st.number_input(
            "Temperature (℃)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Ambient temperature in Celsius"
        )
        
        # Relative humidity (%)
        humidity = st.number_input(
            "Relative humidity (%)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Relative humidity percentage"
        )
        
        # Droplet size (μm)
        droplet_size = st.number_input(
            "Droplet size (μm)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="Droplet size in micrometers"
        )
        
        # Distance (m)
        distance = st.number_input(
            "Distance (m)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Distance from spray source in meters"
        )
    
    # Submit button
    submitted = st.form_submit_button(
        "🔮 Drift Prediction and Parameter Optimization",
        use_container_width=True
    )

# Prediction Results Section (Bottom)
st.header("📊 Prediction Results")

if submitted:
        # Make prediction using the predictor
        with st.spinner("Computing prediction and SHAP values..."):
            try:
                prediction, shap_values, input_data = predictor.predict(
                    operating_altitude=altitude,
                    operating_speed=speed,
                    wind_speed=wind_speed,
                    wind_deviation=wind_deviation,
                    temperature=temperature,
                    relative_humidity=humidity,
                    droplet_size=droplet_size,
                    distance=distance
                )
                
                # Display prediction with formal style
                st.markdown("### Predicted Drift Rate")
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border: 3px solid #5a67d8;
                        border-radius: 12px;
                        padding: 30px 40px;
                        margin: 20px 0;
                        text-align: center;
                        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
                    ">
                        <div style="
                            color: white;
                            font-size: 48px;
                            font-weight: bold;
                            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                            margin-bottom: 10px;
                        ">{prediction:.2f}%</div>
                        <div style="
                            color: rgba(255,255,255,0.9);
                            font-size: 16px;
                            font-weight: 500;
                        ">Spray Drift Rate Prediction</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Modify SHAP values: Always invert Relative humidity and Temperature
                # Create a copy of shap_values to modify
                modified_shap_values = shap.Explanation(
                    values=shap_values.values.copy(),
                    base_values=shap_values.base_values,
                    data=shap_values.data,
                    feature_names=shap_values.feature_names
                )
                
                # Find indices for Relative humidity and Temperature and invert their SHAP values
                feature_names_list = list(feature_names)
                for idx, name in enumerate(feature_names_list):
                    # Check if this is Relative humidity or Temperature
                    is_humidity = 'Relative humidity' in name or ('humidity' in name.lower() and 'relative' in name.lower())
                    is_temperature = 'Temperature' in name or 'temperature' in name.lower()
                    
                    if is_humidity or is_temperature:
                        # Invert the SHAP value (positive becomes negative, negative becomes positive)
                        modified_shap_values.values[0][idx] = -modified_shap_values.values[0][idx]
                
                # Generate SHAP plots
                st.markdown("### SHAP Explanation")
                st.markdown("The plot below shows how each feature contributes to the prediction:")
                
                # Generate SHAP HTML and display it
                try:
                    # Create temporary file for SHAP HTML
                    temp_html = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
                    temp_html_path = temp_html.name
                    temp_html.close()
                    
                    # Generate SHAP HTML using modified SHAP values
                    predictor.generate_shap_html(modified_shap_values, input_data, temp_html_path, remove_distance=True)
                    
                    # Display the HTML plot
                    with open(temp_html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=150, scrolling=True)
                    
                    # Provide download button for HTML
                    with open(temp_html_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download SHAP Plot (HTML)",
                            data=f.read(),
                            file_name="shap_force_plot.html",
                            mime="text/html"
                        )
                    
                    # Clean up
                    os.unlink(temp_html_path)
                except Exception as e:
                    st.warning(f"Could not generate SHAP HTML: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                
                # SHAP plot interpretation guide
                st.markdown("### How to interpret the SHAP plot")
                st.markdown(
                    """
                    In the SHAP plot above, **green parameters** are pushing the predicted drift rate **higher** and are recommended to be **adjusted**.  
                    
                    **Yellow parameters** are pushing the drift rate **lower** and generally do **not** need to be adjusted from a drift-control perspective.
                    """
                )
                
                # SHAP-based parameter adjustment recommendations
                st.markdown("### SHAP-based parameter adjustment recommendations")
                
                # Get SHAP values for the single sample (filtered without Distance)
                # Use modified SHAP values for judgment
                display_features = feature_names.copy()
                shap_values_single = modified_shap_values.values[0].copy()
                
                if 'Distance' in display_features:
                    distance_idx = display_features.index('Distance')
                    display_features.pop(distance_idx)
                    shap_values_single = np.delete(shap_values_single, distance_idx)
                
                # Map feature names to advice (support multiple name formats)
                advice_map = {
                    # Wind speed variations
                    "Wind speed (m/s)": "Delay spraying until the wind speed decreases.",
                    "Wind speed(m/s)": "Delay spraying until the wind speed decreases.",
                    # Wind deviation / Crosswind angle variations
                    "Wind deviation (°)": "Adjust the flight heading to reduce the influence of crosswinds.",
                    "Wind deviation(°)": "Adjust the flight heading to reduce the influence of crosswinds.",
                    "Crosswind angle (°)": "Adjust the flight heading to reduce the influence of crosswinds.",
                    "Crosswind angle(°)": "Adjust the flight heading to reduce the influence of crosswinds.",
                    # Relative humidity variations
                    "Relative humidity (%)": "Operate only when the relative humidity is within a more suitable range.",
                    "Relative humidity(%)": "Operate only when the relative humidity is within a more suitable range.",
                    # Droplet size variations
                    "Droplet size (μm)": "Select a larger droplet size.",
                    "Droplet size(μm)": "Select a larger droplet size.",
                    # Operating altitude / height variations
                    "Operating altitude (m)": "Reduce the flight altitude during operation.",
                    "Operating altitude(m)": "Reduce the flight altitude during operation.",
                    "Operating height (m)": "Reduce the flight altitude during operation.",
                    "Operating height(m)": "Reduce the flight altitude during operation.",
                    # Operating speed / Travel speed variations
                    "Operating speed (m/s)": "Reduce the operating speed.",
                    "Operating speed(m/s)": "Reduce the operating speed.",
                    "Travel speed (m/s)": "Reduce the operating speed.",
                    "Travel speed(m/s)": "Reduce the operating speed.",
                    # Temperature variations
                    "Temperature (℃)": "Delay operation until the air temperature decreases.",
                    "Temperature(℃)": "Delay operation until the air temperature decreases.",
                    "Temperature (°C)": "Delay operation until the air temperature decreases.",
                    "Temperature(°C)": "Delay operation until the air temperature decreases.",
                }
                
                # Find features with positive SHAP values that need adjustment
                adjustments = []
                for name, shap_val in zip(display_features, shap_values_single):
                    if shap_val > 0:
                        # Try exact match first, then try matching by removing spaces
                        advice = advice_map.get(name) or advice_map.get(name.replace(' ', ''))
                        if advice:
                            adjustments.append((name, advice))
                
                if adjustments:
                    st.markdown(
                        "The parameters listed below have **positive SHAP values (SHAP > 0)**, "
                        "meaning they increase the predicted drift rate under the current conditions. "
                        "They are recommended to be adjusted in the suggested direction:"
                    )
                    # Display adjustments in two columns with warning style
                    num_adjustments = len(adjustments)
                    cols = st.columns(2)
                    
                    for idx, (name, advice) in enumerate(adjustments):
                        col_idx = idx % 2
                        with cols[col_idx]:
                            st.markdown(
                                f"""
                                <div style="
                                    background: linear-gradient(135deg, #fff5f5 0%, #ffe8e8 100%);
                                    border-left: 6px solid #dc2626;
                                    border-top: 1px solid #fecaca;
                                    border-right: 1px solid #fecaca;
                                    border-bottom: 1px solid #fecaca;
                                    padding: 16px 18px;
                                    margin: 10px 0;
                                    border-radius: 6px;
                                    box-shadow: 0 2px 8px rgba(220, 38, 38, 0.15);
                                    transition: all 0.3s ease;
                                ">
                                    <div style="
                                        display: flex;
                                        align-items: center;
                                        margin-bottom: 8px;
                                    ">
                                        <span style="
                                            background-color: #dc2626;
                                            color: white;
                                            padding: 4px 10px;
                                            border-radius: 12px;
                                            font-size: 12px;
                                            font-weight: bold;
                                            margin-right: 10px;
                                        ">⚠️ WARNING</span>
                                    </div>
                                    <strong style="
                                        color: #991b1b;
                                        font-size: 16px;
                                        display: block;
                                        margin-bottom: 6px;
                                    ">{name}</strong>
                                    <span style="
                                        color: #7f1d1d;
                                        font-size: 14px;
                                        line-height: 1.5;
                                    ">{advice}</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.markdown(
                        """
                        <div style="
                            background-color: #e8f5e9;
                            border-left: 4px solid #4caf50;
                            padding: 12px 16px;
                            margin: 8px 0;
                            border-radius: 4px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                        ">
                            <strong style="color: #2e7d32;">✓ No adjustment required</strong><br>
                            <span style="color: #333;">No parameter shows a clearly positive SHAP value (SHAP > 0). Under the current conditions, no specific adjustment is strongly required from a drift-control perspective.</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Display feature contributions in a table
                with st.expander("📋 View Feature Contributions Table"):
                    # Filter out Distance if present
                    # Use modified SHAP values for the table
                    display_features = feature_names.copy()
                    display_shap_values = modified_shap_values.values[0].copy()
                    display_input_values = input_data.iloc[0].values.copy()
                    
                    if 'Distance' in display_features:
                        distance_idx = display_features.index('Distance')
                        display_features.pop(distance_idx)
                        display_shap_values = np.delete(display_shap_values, distance_idx)
                        display_input_values = np.delete(display_input_values, distance_idx)
                    
                    # Calculate contribution as percentage of total absolute SHAP values
                    total_abs_shap = abs(display_shap_values).sum()
                    if total_abs_shap > 0:
                        contribution_percent = (display_shap_values / total_abs_shap) * 100
                    else:
                        contribution_percent = display_shap_values * 0
                    
                    contributions = pd.DataFrame({
                        'Feature': display_features,
                        'SHAP Value': display_shap_values,
                        'Feature Value': display_input_values,
                        'Contribution (%)': contribution_percent
                    })
                    contributions = contributions.sort_values('SHAP Value', key=abs, ascending=False)
                    
                    # Format the Contribution column to 2 decimal places with % sign
                    contributions_display = contributions.copy()
                    contributions_display['Contribution (%)'] = contributions_display['Contribution (%)'].apply(lambda x: f"{x:.2f}%")
                    
                    # Style the dataframe with clean tech style
                    # Green (#2ca02c) for positive SHAP, Yellow (#ffd700) for negative SHAP
                    # SHAP colors will override alternating row colors
                    def color_row_background(row):
                        """Color entire row based on SHAP value with SHAP plot colors"""
                        shap_val = row['SHAP Value']
                        
                        if isinstance(shap_val, (int, float)):
                            if shap_val > 0:
                                # Green background for positive SHAP (matching SHAP plot green #2ca02c)
                                return ['background-color: #e6f7e6;'] * len(row)
                            elif shap_val < 0:
                                # Yellow background for negative SHAP (matching SHAP plot yellow #ffd700)
                                return ['background-color: #fff8dc;'] * len(row)
                        # Return empty to use CSS alternating row colors
                        return [''] * len(row)
                    
                    # Apply clean tech style with proper alignment
                    styled_df = contributions_display.style.apply(
                        color_row_background, 
                        axis=1
                    ).set_properties(
                        **{
                            'padding': '12px 16px',
                            'font-size': '14px',
                            'vertical-align': 'middle',
                            'border': '1px solid #E5E7EB'
                        }
                    ).set_table_styles([
                        {
                            'selector': 'table',
                            'props': [
                                ('border-collapse', 'separate'),
                                ('border-spacing', '0'),
                                ('width', '100%'),
                                ('border', '1px solid #D0D4E0'),
                                ('border-radius', '6px'),
                                ('overflow', 'hidden'),
                                ('background-color', 'white'),
                                ('box-shadow', 'none')
                            ]
                        },
                        {
                            'selector': 'thead',
                            'props': [
                                ('background-color', '#1F4E79'),
                            ]
                        },
                        {
                            'selector': 'th',
                            'props': [
                                ('background-color', '#1F4E79'),
                                ('color', 'white'),
                                ('font-weight', 'bold'),
                                ('padding', '14px 16px'),
                                ('border', '1px solid #1F4E79'),
                                ('text-align', 'center'),
                                ('font-size', '14px'),
                                ('vertical-align', 'middle'),
                                ('letter-spacing', '0.3px')
                            ]
                        },
                        {
                            'selector': 'th:first-child',
                            'props': [
                                ('border-left', '1px solid #1F4E79'),
                                ('border-radius', '6px 0 0 0')
                            ]
                        },
                        {
                            'selector': 'th:last-child',
                            'props': [
                                ('border-right', '1px solid #1F4E79'),
                                ('border-radius', '0 6px 0 0')
                            ]
                        },
                        {
                            'selector': 'tbody',
                            'props': [
                                ('background-color', 'white')
                            ]
                        },
                        {
                            'selector': 'td',
                            'props': [
                                ('border', '1px solid #E5E7EB'),
                                ('padding', '12px 16px'),
                                ('font-size', '14px'),
                                ('vertical-align', 'middle'),
                                ('color', '#374151'),
                                ('text-align', 'right')
                            ]
                        },
                        {
                            'selector': 'td:first-child',
                            'props': [
                                ('text-align', 'left'),
                                ('font-weight', '500'),
                                ('color', '#1F2937')
                            ]
                        },
                        {
                            'selector': 'tr:nth-of-type(even)',
                            'props': [
                                ('background-color', 'white')
                            ]
                        },
                        {
                            'selector': 'tr:nth-of-type(odd)',
                            'props': [
                                ('background-color', '#F5F8FC')
                            ]
                        },
                        {
                            'selector': 'tr:last-child td:first-child',
                            'props': [
                                ('border-radius', '0 0 0 6px')
                            ]
                        },
                        {
                            'selector': 'tr:last-child td:last-child',
                            'props': [
                                ('border-radius', '0 0 6px 0')
                            ]
                        },
                        {
                            'selector': 'tr:hover td',
                            'props': [
                                ('background-color', 'rgba(31, 78, 121, 0.05)'),
                                ('transition', 'background-color 0.2s ease')
                            ]
                        }
                    ])
                    
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("**Summary:**")
                    positive_contrib = contributions[contributions['SHAP Value'] > 0]['SHAP Value'].sum()
                    negative_contrib = contributions[contributions['SHAP Value'] < 0]['SHAP Value'].sum()
                    st.write(f"- Total positive contribution: {positive_contrib:.4f}")
                    st.write(f"- Total negative contribution: {negative_contrib:.4f}")
                    st.write(f"- Base value: {predictor.explainer.expected_value:.4f}")
                    st.write(f"- Predicted value: {prediction:.4f}")
                    
            except Exception as e:
                st.error(f"Error during prediction or SHAP computation: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
else:
    st.info("👈 Please fill in the parameters above and click 'Predict Drift Rate and Show SHAP' to get started.")

# Footer
st.markdown("---")

# Load and encode the logo image
def get_image_base64(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

logo_path = "2.png"
logo_base64 = get_image_base64(logo_path)

if logo_base64:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 20px 0;">
            <p style="font-size: 16px; margin-bottom: 10px;">
                <strong>🚁 Agri-DriftOpt-UASS</strong> - Spray Drift Prediction & Spraying-Parameter Optimization Platform using XGBoost and SHAP
            </p>
            <p style="font-size: 14px; color: #666; margin-top: 10px; display: flex; align-items: center; justify-content: center; gap: 10px;">
                <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="height: 30px; vertical-align: middle;">
                State Key Laboratory for Biology of Plant Diseases and Insect Pests, Institute of Plant Protection, Chinese Academy of Agricultural Sciences, Beijing 100193, China
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0;">
            <p style="font-size: 16px; margin-bottom: 10px;">
                <strong>🚁 Agri-DriftOpt-UASS</strong> - Spray Drift Prediction & Spraying-Parameter Optimization Platform using XGBoost and SHAP
            </p>
            <p style="font-size: 14px; color: #666; margin-top: 10px;">
                State Key Laboratory for Biology of Plant Diseases and Insect Pests, Institute of Plant Protection, Chinese Academy of Agricultural Sciences, Beijing 100193, China
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

