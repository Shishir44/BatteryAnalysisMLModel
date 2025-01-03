import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, Dict, List
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class DataPreprocessor:
    def __init__(self, config: Config = None):
        """
        Initialize the DataPreprocessor for comprehensive battery data.
        
        Args:
            config: Configuration object containing preprocessing parameters
        """
        self.config = config or Config()
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.feature_groups = {
            'voltage_features': [
                'voltage_mean', 'voltage_std', 'voltage_max', 'voltage_min',
                'voltage_range', 'voltage_skew', 'voltage_kurtosis',
                'voltage_efficiency', 'voltage_stability'
            ],
            'current_features': [
                'current_mean', 'current_std', 'current_integral',
                'terminal_current_mean', 'terminal_current_std',
                'terminal_current_min', 'terminal_current_max'
            ],
            'temperature_features': [
                'temp_mean', 'temp_max', 'temp_rise', 'temp_integral',
                'temp_std', 'temp_stress', 'temperature_mean',
                'temperature_max', 'temperature_std'
            ],
            'efficiency_features': [
                'voltage_efficiency', 'power_efficiency',
                'coulombic_efficiency'
            ],
            'degradation_features': [
                'capacity_degradation', 'SOH_degradation',
                'capacity_degradation_rate', 'SOH_degradation_rate'
            ]
        }
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocess the comprehensive battery data.
        
        Args:
            df: Raw comprehensive battery data DataFrame
            
        Returns:
            Tuple containing preprocessed DataFrame and preprocessing statistics
        """
        logger.info("Starting data preprocessing...")
        
        stats = {
            'initial_shape': df.shape,
            'missing_values': {},
            'outliers_removed': {},
            'features_scaled': []
        }
        
        try:
            # 1. Handle missing values
            df_processed = self._handle_missing_values(df)
            stats['missing_values'] = {
                col: missing for col, missing in df.isnull().sum().items() if missing > 0
            }
            
            # 2. Remove outliers using robust statistics
            df_processed = self._remove_outliers(df_processed)
            stats['outliers_removed'] = {
                group: len(df) - len(df_processed) 
                for group in self.feature_groups.keys()
            }
            
            # 3. Scale features by group
            df_processed = self._scale_features(df_processed)
            stats['features_scaled'] = list(df_processed.columns)
            
            # 4. Validate processed data
            df_processed = self._validate_processed_data(df_processed)
            
            stats['final_shape'] = df_processed.shape
            logger.info("Preprocessing completed successfully")
            
            return df_processed, stats
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in comprehensive battery data."""
        df_clean = df.copy()
        
        for group, features in self.feature_groups.items():
            group_features = [f for f in features if f in df.columns]
            if group_features:
                # For each feature group, use appropriate imputation strategy
                if group in ['voltage_features', 'current_features', 'temperature_features']:
                    # Use interpolation for time-series related features
                    df_clean[group_features] = df_clean[group_features].interpolate(
                        method='linear', 
                        limit_direction='both'
                    )
                elif group in ['efficiency_features']:
                    # Use median for efficiency metrics
                    df_clean[group_features] = df_clean[group_features].fillna(
                        df_clean[group_features].median()
                    )
                else:
                    # Forward fill for degradation features
                    df_clean[group_features] = df_clean[group_features].fillna(
                        method='ffill'
                    ).fillna(method='bfill')  # backfill any remaining NaNs
        
        return df_clean
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using robust statistics for each feature group."""
        df_clean = df.copy()
        
        for group, features in self.feature_groups.items():
            group_features = [f for f in features if f in df.columns]
            if group_features:
                # Calculate robust statistics for each feature
                Q1 = df_clean[group_features].quantile(0.25)
                Q3 = df_clean[group_features].quantile(0.75)
                IQR = Q3 - Q1
                
                # Set different thresholds for different feature groups
                if group in ['efficiency_features']:
                    # Stricter bounds for efficiency metrics
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                else:
                    # More relaxed bounds for other features
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                
                # Create mask for valid data points
                mask = np.ones(len(df_clean), dtype=bool)
                for feature in group_features:
                    feature_mask = (
                        (df_clean[feature] >= lower_bound[feature]) & 
                        (df_clean[feature] <= upper_bound[feature])
                    )
                    mask = mask & feature_mask
                
                df_clean = df_clean[mask]
        
        return df_clean
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using appropriate scalers for each feature group."""
        df_scaled = df.copy()
        
        # Don't scale certain columns
        exclude_from_scaling = ['cycle', 'SOH', 'capacity']
        
        for group, features in self.feature_groups.items():
            group_features = [
                f for f in features 
                if f in df.columns and f not in exclude_from_scaling
            ]
            
            if group_features:
                if group in ['efficiency_features']:
                    # Use robust scaler for efficiency features
                    df_scaled[group_features] = self.robust_scaler.fit_transform(
                        df_scaled[group_features]
                    )
                else:
                    # Use standard scaler for other features
                    df_scaled[group_features] = self.standard_scaler.fit_transform(
                        df_scaled[group_features]
                    )
        
        return df_scaled
    
    def _validate_processed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the processed data meets quality requirements."""
        # Check for any remaining missing values
        if df.isnull().any().any():
            raise ValueError("Processed data contains missing values")
        
        # Ensure critical columns exist
        critical_columns = ['cycle', 'SOH', 'capacity']
        missing_critical = [col for col in critical_columns if col not in df.columns]
        if missing_critical:
            raise ValueError(f"Missing critical columns: {missing_critical}")
        
        # Validate value ranges for critical metrics
        if (df['SOH'] < 0).any() or (df['SOH'] > 100).any():
            raise ValueError("SOH values out of valid range (0-100)")
        
        if (df['capacity'] < 0).any():
            raise ValueError("Negative capacity values found")
        
        return df

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return the feature groups configuration."""
        return self.feature_groups