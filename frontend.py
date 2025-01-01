# import streamlit as st
# import plotly.graph_objects as go
# import pandas as pd
# from datetime import datetime

# def create_custom_theme():
#     # Configure page layout and theme
#     st.set_page_config(layout="wide", page_title="Battery Health Analytics")
    
#     # Custom CSS for styling
#     st.markdown("""
#     <style>
#     .main {
#         background-color: #f8f9fa;
#     }
#     .stButton>button {
#         background-color: #0066cc;
#         color: white;
#         border-radius: 20px;
#         padding: 10px 25px;
#     }
#     .css-1d391kg {
#         padding: 1rem 5rem;
#     }
#     .sidebar .sidebar-content {
#         background-color: white;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# def main():
#     create_custom_theme()
    
#     # Header Section with Logo and Navigation
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col1:
#         st.image("assets\logo.jpeg", width=150)  # Add your logo here
#     with col2:
#         st.title("AI-POWERED BATTERY ANALYTICS")
#     with col3:
#         st.button("Start Analysis", type="primary")
    
#     # Navigation Menu
#     menu = ["Home", "Analysis Dashboard", "Predictive Analytics", "Historical Data"]
#     selected = st.tabs(menu)
    
#     # Main Content Area
#     with selected[1]:  # Analysis Dashboard
#         st.subheader("Battery Health Monitoring System")
        
#         # Two-column layout for inputs and real-time visualization
#         left_col, right_col = st.columns([2, 3])
        
#         with left_col:
#             # Stylized input card
#             st.markdown("""
#             <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
#             <h3>Input Parameters</h3>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Group related inputs
#             with st.expander("Voltage Measurements", expanded=True):
#                 terminal_voltage = st.number_input(
#                     "Terminal Voltage (V)",
#                     min_value=3.0,
#                     max_value=4.5,
#                     value=3.8,
#                     format="%.3f"
#                 )
#                 charge_voltage = st.number_input(
#                     "Charge Voltage (V)",
#                     min_value=0.0,
#                     max_value=4.5,
#                     value=4.0,
#                     format="%.3f"
#                 )
            
#             with st.expander("Current Measurements", expanded=True):
#                 terminal_current = st.number_input(
#                     "Terminal Current (A)",
#                     min_value=-2.5,
#                     max_value=0.5,
#                     value=-0.5,
#                     format="%.3f"
#                 )
#                 charge_current = st.number_input(
#                     "Charge Current (A)",
#                     min_value=-2.5,
#                     max_value=0.5,
#                     value=-0.5,
#                     format="%.3f"
#                 )
            
#             with st.expander("Environmental & Cycle Data", expanded=True):
#                 temperature = st.slider(
#                     "Temperature (°C)",
#                     min_value=20.0,
#                     max_value=30.0,
#                     value=25.0
#                 )
#                 capacity = st.number_input(
#                     "Capacity (Ah)",
#                     min_value=0.0,
#                     max_value=3.0,
#                     value=1.85,
#                     format="%.3f"
#                 )
#                 cycle = st.number_input(
#                     "Cycle Number",
#                     min_value=1,
#                     step=1,
#                     value=1
#                 )
        
#         with right_col:
#             # Results Display Area
#             st.markdown("""
#             <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
#             <h3>Real-time Analysis</h3>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Create three columns for metrics
#             metric_col1, metric_col2, metric_col3 = st.columns(3)
            
#             with metric_col1:
#                 st.markdown("""
#                 <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>
#                 <h4>State of Health</h4>
#                 <h2 style='color: #0066cc;'>92.8%</h2>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#             with metric_col2:
#                 st.markdown("""
#                 <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>
#                 <h4>Predicted Capacity</h4>
#                 <h2 style='color: #0066cc;'>1.856 Ah</h2>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#             with metric_col3:
#                 st.markdown("""
#                 <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px;'>
#                 <h4>Battery Status</h4>
#                 <h2 style='color: #00cc66;'>Normal</h2>
#                 </div>
#                 """, unsafe_allow_html=True)
            
#             # Add interactive plots
#             st.plotly_chart(create_health_trend_plot(), use_container_width=True)
            
#     # Additional features in sidebar
#     with st.sidebar:
#         st.markdown("### Quick Actions")
#         st.button("Export Report")
#         st.button("Schedule Analysis")
        
#         st.markdown("### System Status")
#         st.info("Last Update: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# def create_health_trend_plot():
#     # Create sample data for demonstration
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=[1, 2, 3, 4, 5],
#         y=[0.95, 0.93, 0.92, 0.91, 0.928],
#         name="SOH Trend",
#         line=dict(color="#0066cc", width=2)
#     ))
#     fig.update_layout(
#         title="Battery Health Trend",
#         xaxis_title="Time",
#         yaxis_title="State of Health",
#         template="plotly_white"
#     )
#     return fig

# if __name__ == "__main__":
#     main()

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
from datetime import datetime
from typing import Dict, List

class FrontendFeatureEngineer:
    """
    A simplified version of FeatureEngineer for frontend use.
    Creates all required features from basic input parameters.
    """
    def __init__(self):
        # Define window sizes for time-series features
        self.window_sizes = [5, 10, 20]
        
        # Define temperature ranges for advanced features
        self.temp_ranges = [(0, 25), (25, 40), (40, 60)]
        
        # Store historical data for time series calculations
        self.historical_data = []
        self.max_history_size = 20  # Keep enough history for largest window
    
    def create_feature_vector(self, input_data: Dict[str, float], debug: bool = True) -> np.ndarray:
        """
        Create all required features from input parameters.
        
        Args:
            input_data: Dictionary containing basic battery parameters
            debug: If True, print feature counts for debugging
            
        Returns:
            numpy array containing all 88 features in the correct order
        """
        """
        Create all required features from input parameters.
        
        Args:
            input_data: Dictionary containing basic battery parameters
        
        Returns:
            numpy array containing all 88 features in the correct order
        """
        # Store current data point in history
        self.historical_data.append(input_data)
        if len(self.historical_data) > self.max_history_size:
            self.historical_data.pop(0)
            
        features = []
        
        # 1. Add basic features
        basic_features = self._create_basic_features(input_data)
        features.extend(basic_features.values())
        
        # 2. Add time series features
        time_series_features = self._create_time_series_features()
        features.extend(time_series_features.values())
        
        # 3. Add statistical features
        statistical_features = self._create_statistical_features()
        features.extend(statistical_features.values())
        
        # 4. Add advanced features
        advanced_features = self._create_advanced_features(input_data)
        features.extend(advanced_features.values())
        
        # 5. Add base parameters (excluding SOH)
        base_params = self._get_base_parameters(input_data)
        features.extend(base_params.values())
        
        if debug:
            feature_array = np.array(features)
            print(f"Total features created: {len(feature_array)}")
            print(f"Basic features: {len(basic_features)}")
            print(f"Time series features: {len(time_series_features)}")
            print(f"Statistical features: {len(statistical_features)}")
            print(f"Advanced features: {len(advanced_features)}")
            print(f"Base parameters: {len(base_params)}")
        
        return np.array(features)
    
    def _create_basic_features(self, data: Dict[str, float]) -> Dict[str, float]:
        """Create basic engineered features."""
        features = {}
        
        # Power features
        features['power'] = data['terminal_voltage'] * data['terminal_current']
        features['charge_power'] = data['charge_voltage'] * data['charge_current']
        
        # Efficiency features
        features['voltage_efficiency'] = (data['terminal_voltage'] / data['charge_voltage'] 
                                        if data['charge_voltage'] != 0 else 0)
        features['current_efficiency'] = (data['terminal_current'] / data['charge_current']
                                        if data['charge_current'] != 0 else 0)
        
        # Resistance features
        features['internal_resistance'] = ((data['charge_voltage'] - data['terminal_voltage']) 
                                         / data['terminal_current'] if data['terminal_current'] != 0 else 0)
        
        # Energy features (using 1.0 as time diff since we don't have real time series)
        features['energy'] = features['power'] * 1.0
        features['charge_energy'] = features['charge_power'] * 1.0
        
        # Overall efficiency
        features['energy_efficiency'] = (features['energy'] / features['charge_energy'] 
                                       if features['charge_energy'] != 0 else 0)
        
        return features
    
    def _create_time_series_features(self) -> Dict[str, float]:
        """
        Creates time series based features using historical battery measurements.
        
        This function calculates rolling statistics and rates of change for key battery parameters
        across different time windows. It handles missing or insufficient data gracefully by
        using appropriate filling strategies.
        
        Returns:
            Dict[str, float]: Dictionary containing all time series features
        """
        # Initialize empty dictionary to store our calculated features
        features = {}
        
        # Define the core parameters we want to analyze
        # These are the key parameters that indicate battery health and performance
        params = [
            'terminal_voltage',    # Basic measurement of battery voltage
            'terminal_current',    # Basic measurement of current flow
            'temperature',         # Environmental parameter affecting battery performance
            'power',              # Derived from voltage and current, indicates energy transfer
            'internal_resistance'  # Key battery health indicator
        ]
        
        # Convert our historical data into a DataFrame for easier manipulation
        history_df = pd.DataFrame(self.historical_data)
        
        # Handle cases where we don't have enough historical data
        if len(history_df) < self.max_history_size:
            for param in params:
                # Calculate derived parameters if they don't exist
                if param not in history_df.columns:
                    if param == 'power':
                        # Power is the product of voltage and current
                        history_df[param] = history_df['terminal_voltage'] * history_df['terminal_current']
                    elif param == 'internal_resistance':
                        # Internal resistance is calculated from voltage and current differences
                        history_df[param] = np.where(
                            history_df['terminal_current'] != 0,
                            (history_df['charge_voltage'] - history_df['terminal_voltage']) /
                            history_df['terminal_current'],
                            0  # Use 0 as fallback when current is 0
                        )
                
                # Fill any missing values using forward fill, then backward fill
                # This ensures we have continuous data for our calculations
                history_df[param] = (history_df[param]
                                .fillna(method='ffill')  # First try to fill with previous values
                                .fillna(method='bfill')  # Then fill any remaining NaNs with next values
                                .fillna(0))              # Finally, use 0 for any still-missing values
        
        # Calculate features for each window size and parameter
        for window in self.window_sizes:  # window_sizes is typically [5, 10, 20]
            for param in params:
                if param in history_df.columns:
                    # 1. Calculate rolling mean (average over the window)
                    mean_val = history_df[param].rolling(
                        window=min(window, len(history_df)),  # Don't use window larger than our data
                        min_periods=1  # Allow calculation even with single value
                    ).mean().iloc[-1]  # Get the most recent value
                    
                    # Store the mean, using 0 as fallback if calculation failed
                    features[f'{param}_rolling_mean_{window}'] = mean_val if not pd.isna(mean_val) else 0
                    
                    # 2. Calculate rolling standard deviation (variation over the window)
                    std_val = history_df[param].rolling(
                        window=min(window, len(history_df)),
                        min_periods=1
                    ).std().iloc[-1]
                    
                    # Store the standard deviation, using 0 as fallback
                    features[f'{param}_rolling_std_{window}'] = std_val if not pd.isna(std_val) else 0
                    
                    # 3. Calculate rate of change (trend over the window)
                    diff_val = history_df[param].diff(
                        periods=min(window, len(history_df)-1)  # Use smaller period if not enough data
                    ).iloc[-1]
                    
                    # Calculate rate of change, handling missing values
                    if pd.isna(diff_val):
                        rate = 0
                    else:
                        # Rate is the change divided by the time period
                        rate = diff_val / min(window, len(history_df)-1)
                    
                    # Store the rate of change
                    features[f'{param}_rate_{window}'] = rate
        
        return features
    
    def _create_statistical_features(self) -> Dict[str, float]:
        """Create statistical features using historical data."""
        features = {}
        history_df = pd.DataFrame(self.historical_data)
        
        # Statistical calculations for each parameter
        params_stats = {
            'terminal_voltage': ['mean', 'std', 'max', 'min', 'skew'],
            'terminal_current': ['mean', 'std', 'max', 'min', 'skew'],
            'temperature': ['mean', 'std', 'max', 'min'],
            'power': ['mean', 'max', 'sum'],
            'internal_resistance': ['mean', 'std'],
            'energy_efficiency': ['mean', 'std']
        }
        
        for param, stats in params_stats.items():
            if param not in history_df.columns:
                # Calculate derived parameters first
                if param == 'power':
                    history_df[param] = history_df['terminal_voltage'] * history_df['terminal_current']
                elif param == 'internal_resistance':
                    history_df[param] = np.where(
                        history_df['terminal_current'] != 0,
                        (history_df['charge_voltage'] - history_df['terminal_voltage']) /
                        history_df['terminal_current'],
                        0
                    )
                elif param == 'energy_efficiency':
                    power = history_df['terminal_voltage'] * history_df['terminal_current']
                    charge_power = history_df['charge_voltage'] * history_df['charge_current']
                    history_df[param] = np.where(charge_power != 0, power / charge_power, 0)
            
            # Calculate statistics
            for stat in stats:
                if stat == 'mean':
                    features[f'{param}_mean'] = history_df[param].mean()
                elif stat == 'std':
                    features[f'{param}_std'] = history_df[param].std()
                elif stat == 'max':
                    features[f'{param}_max'] = history_df[param].max()
                elif stat == 'min':
                    features[f'{param}_min'] = history_df[param].min()
                elif stat == 'skew':
                    features[f'{param}_skew'] = history_df[param].skew()
                elif stat == 'sum':
                    features[f'{param}_sum'] = history_df[param].sum()
        
        return features
    
    def _create_advanced_features(self, data: Dict[str, float]) -> Dict[str, float]:
        """Create advanced engineered features."""
        features = {}
        
        # Capacity retention (using 1.0 as initial capacity since we don't have historical data)
        features['capacity_retention'] = data['capacity'] / 1.0
        
        # SOH change rate (using 0 since we don't have historical SOH)
        features['soh_change_rate'] = 0.0
        
        # Temperature stress indicators
        for i, (min_temp, max_temp) in enumerate(self.temp_ranges):
            features[f'temp_range_{i}'] = (
                1.0 if min_temp <= data['temperature'] < max_temp else 0.0
            )
        
        # Voltage stress (using current voltage vs typical range)
        typical_voltage_mean = 3.7  # typical Li-ion cell nominal voltage
        typical_voltage_std = 0.5   # approximate standard deviation
        features['voltage_stress'] = ((data['terminal_voltage'] - typical_voltage_mean) / 
                                    typical_voltage_std)
        
        # Cycle progress (using 0.5 as default since we don't have cycle information)
        features['cycle_progress'] = 0.5
        
        return features
    
    def _get_base_parameters(self, data: Dict[str, float]) -> Dict[str, float]:
        """Return the base parameters (excluding SOH)."""
        return {
            'terminal_voltage': data['terminal_voltage'],
            'terminal_current': data['terminal_current'],
            'temperature': data['temperature'],
            'charge_voltage': data['charge_voltage'],
            'charge_current': data['charge_current'],
            'capacity': data['capacity'],
            'cycle': 1.0  # Default to 1 since we don't track cycles in frontend
        }

class BatteryPredictionInterface:
    def __init__(self):
        """Initialize the prediction interface with necessary models and configurations."""
        st.set_page_config(
            page_title="Battery Health Predictor",
            page_icon="🔋",
            layout="wide"
        )
        
        # Define normal operating ranges for parameters
        self.parameter_ranges = {
            'terminal_voltage': (2.5, 4.2),
            'terminal_current': (-20.0, 20.0),
            'temperature': (0.0, 45.0),
            'charge_current': (0.0, 10.0),
            'charge_voltage': (3.0, 4.2),
            'capacity': (0.5, 2.0)
        }
        
        # Initialize feature engineer
        self.feature_engineer = FrontendFeatureEngineer()
        
        # Load models
        self.load_models()

    def load_models(self):
        """Load the trained machine learning models."""
        try:
            self.soh_model = joblib.load('data/models/soh_model.pkl')
            self.capacity_model = joblib.load('data/models/capacity_model.pkl')
            self.anomaly_model = joblib.load('data/models/anomaly_model.pkl')
            st.session_state.models_loaded = True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.session_state.models_loaded = False

    def create_input_section(self):
        """Create the user input section with consistent float parameters."""
        st.header("Battery Parameter Inputs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Using float values consistently for all inputs
            terminal_voltage = st.number_input(
                "Terminal Voltage (V)",
                min_value=float(self.parameter_ranges['terminal_voltage'][0]),
                max_value=float(self.parameter_ranges['terminal_voltage'][1]),
                value=3.7,
                step=0.1,
                format="%.2f",
                help="The current voltage reading of the battery"
            )
            
            terminal_current = st.number_input(
                "Terminal Current (A)",
                min_value=float(self.parameter_ranges['terminal_current'][0]),
                max_value=float(self.parameter_ranges['terminal_current'][1]),
                value=0.0,
                step=0.1,
                format="%.2f",
                help="Current flowing through the battery (negative for discharge)"
            )
            
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=float(self.parameter_ranges['temperature'][0]),
                max_value=float(self.parameter_ranges['temperature'][1]),
                value=25.0,
                step=0.1,
                format="%.1f",
                help="Current temperature of the battery"
            )
        
        with col2:
            charge_current = st.number_input(
                "Charging Current (A)",
                min_value=float(self.parameter_ranges['charge_current'][0]),
                max_value=float(self.parameter_ranges['charge_current'][1]),
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Current used during charging"
            )
            
            charge_voltage = st.number_input(
                "Charging Voltage (V)",
                min_value=float(self.parameter_ranges['charge_voltage'][0]),
                max_value=float(self.parameter_ranges['charge_voltage'][1]),
                value=4.2,
                step=0.1,
                format="%.2f",
                help="Voltage used during charging"
            )
            
            current_capacity = st.number_input(
                "Current Capacity (Ah)",
                min_value=float(self.parameter_ranges['capacity'][0]),
                max_value=float(self.parameter_ranges['capacity'][1]),
                value=1.0,
                step=0.1,
                format="%.2f",
                help="Current capacity measurement"
            )

        # Create input dictionary
        input_data = {
            'terminal_voltage': float(terminal_voltage),
            'terminal_current': float(terminal_current),
            'temperature': float(temperature),
            'charge_current': float(charge_current),
            'charge_voltage': float(charge_voltage),
            'capacity': float(current_capacity)
        }
        
        return input_data

    def check_parameter_warnings(self, input_data):
        """Check input parameters for potential issues."""
        warnings = []
        
        for param, value in input_data.items():
            param_range = self.parameter_ranges[param]
            range_width = param_range[1] - param_range[0]
            lower_threshold = param_range[0] + (range_width * 0.1)
            upper_threshold = param_range[1] - (range_width * 0.1)
            
            if value < lower_threshold:
                warnings.append(f"Low {param.replace('_', ' ').title()}")
            elif value > upper_threshold:
                warnings.append(f"High {param.replace('_', ' ').title()}")
        
        if warnings:
            st.warning("⚠️ Attention needed:\n" + "\n".join(f"- {w}" for w in warnings))

    def make_predictions(self, input_data):
        """Generate predictions using the loaded models with engineered features."""
        try:
            # Create feature vector using our feature engineer with debug info
            features = self.feature_engineer.create_feature_vector(input_data, debug=True)
            st.write(f"Number of features created: {len(features)}")
            features = features.reshape(1, -1)  # Reshape for model prediction
            
            # Make predictions
            soh_prediction = float(self.soh_model.predict(features)[0])
            capacity_prediction = float(self.capacity_model.predict(features)[0])
            anomaly_score = float(self.anomaly_model.decision_function(features)[0])
            
            return {
                'soh': soh_prediction,
                'capacity': capacity_prediction,
                'anomaly_score': anomaly_score
            }
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

    def display_predictions(self, predictions):
        """Display prediction results with clear metrics."""
        if predictions is None:
            return

        st.header("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "State of Health",
                f"{predictions['soh']:.1%}",
                delta=f"{(predictions['soh'] - 0.8):.1%}"
            )
            self._display_health_status(predictions['soh'])
        
        with col2:
            st.metric(
                "Predicted Capacity",
                f"{predictions['capacity']:.2f} Ah",
                delta=None
            )
            self._display_capacity_status(predictions['capacity'])
        
        with col3:
            st.metric(
                "Anomaly Score",
                f"{abs(predictions['anomaly_score']):.2f}",
                delta=None
            )
            self._display_anomaly_status(predictions['anomaly_score'])

    def _display_health_status(self, soh):
        """Display SOH status with appropriate color coding."""
        if soh > 0.8:
            st.success("Battery health is good")
        elif soh > 0.6:
            st.warning("Battery health needs attention")
        else:
            st.error("Battery health is critical")

    def _display_capacity_status(self, capacity):
        """Display capacity status relative to nominal."""
        relative_capacity = capacity / self.parameter_ranges['capacity'][1]
        if relative_capacity > 0.8:
            st.success("Capacity is normal")
        elif relative_capacity > 0.6:
            st.warning("Capacity is declining")
        else:
            st.error("Capacity is critically low")

    def _display_anomaly_status(self, anomaly_score):
        """Display anomaly detection results."""
        if abs(anomaly_score) < 0.5:
            st.success("Normal operation detected")
        elif abs(anomaly_score) < 1.0:
            st.warning("Minor anomalies detected")
        else:
            st.error("Significant anomalies detected")

    def run(self):
        """Main method to run the prediction interface."""
        st.title("🔋 Battery Health Prediction System")
        st.write("Enter battery parameters below to predict battery health and receive insights.")
        
        input_data = self.create_input_section()
        self.check_parameter_warnings(input_data)
        
        if st.button("Generate Predictions", type="primary"):
            if st.session_state.models_loaded:
                with st.spinner("Analyzing battery parameters..."):
                    predictions = self.make_predictions(input_data)
                    if predictions:
                        self.display_predictions(predictions)
                        
                        # Add recommendations section
                        st.subheader("📋 Recommendations")
                        self._display_recommendations(input_data, predictions)
            else:
                st.error("Models not loaded. Please check the system configuration.")

    def _display_recommendations(self, input_data, predictions):
        """Display actionable recommendations based on predictions."""
        recommendations = []
        
        # SOH-based recommendations
        if predictions['soh'] < 0.7:
            recommendations.append("⚠️ Consider battery replacement in the near future")
        elif predictions['soh'] < 0.8:
            recommendations.append("📊 Increase monitoring frequency")
        
        # Temperature recommendations
        if input_data['temperature'] > 35:
            recommendations.append("🌡️ Consider improving cooling to extend battery life")
        elif input_data['temperature'] < 15:
            recommendations.append("🌡️ Operating temperature is low - may affect performance")
        
        # Charging recommendations
        if input_data['charge_voltage'] > 4.1:
            recommendations.append("⚡ Consider reducing charging voltage to extend battery life")
        if input_data['charge_current'] > 5:
            recommendations.append("⚡ High charging current detected - consider reducing")
        
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("✅ All parameters are within optimal ranges")

def main():
    """Initialize and run the prediction interface."""
    predictor = BatteryPredictionInterface()
    predictor.run()

if __name__ == "__main__":
    main()