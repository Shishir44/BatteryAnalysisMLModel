import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from ..utils.logger import setup_logger
from ..utils.config import Config
from scipy.stats import skew, kurtosis
from scipy.signal import savgol_filter

logger = setup_logger(__name__)

class FeatureEngineer:
    def __init__(self, config: Config):
        """
        Initialize the FeatureEngineer with configuration.
        """
        self.config = config
        self.feature_columns = None
        self.base_features = {
            'voltage_features': [
                'voltage_mean', 'voltage_std', 'voltage_max', 'voltage_min',
                'voltage_stability', 'voltage_efficiency'
            ],
            'current_features': [
                'current_mean', 'current_std', 'current_integral'
            ],
            'temperature_features': [
                'temp_mean', 'temp_max', 'temp_std', 'temp_stress'
            ],
            'power_features': [
                'power_efficiency', 'avg_power', 'max_power'
            ],
            'degradation_features': [
                'capacity_degradation', 'SOH_degradation',
                'capacity_degradation_rate', 'SOH_degradation_rate'
            ]
        }

    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Create advanced features from comprehensive battery data.
        
        Args:
            df: Input DataFrame with battery data
            
        Returns:
            DataFrame with additional engineered features and feature creation stats
        """
        logger.info("Starting advanced feature engineering...")
        
        df_featured = df.copy()
        stats = {'created_features': []}
        
        try:
            # 1. Create trend features
            df_featured = self._create_trend_features(df_featured)
            stats['trend_features'] = self._get_feature_names('trend')
            
            # 2. Create interaction features
            df_featured = self._create_interaction_features(df_featured)
            stats['interaction_features'] = self._get_feature_names('interaction')
            
            # 3. Create ratio features
            df_featured = self._create_ratio_features(df_featured)
            stats['ratio_features'] = self._get_feature_names('ratio')
            
            # 4. Create complexity features
            df_featured = self._create_complexity_features(df_featured)
            stats['complexity_features'] = self._get_feature_names('complexity')
            
            # 5. Create moving window features
            df_featured = self._create_window_features(df_featured)
            stats['window_features'] = self._get_feature_names('window')
            
            # Store all feature columns
            self.feature_columns = list(set(df_featured.columns) - set(df.columns))
            stats['total_new_features'] = len(self.feature_columns)
            
            logger.info(f"Created {stats['total_new_features']} new features")
            return df_featured, stats
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise

    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend-based features."""
        # Calculate trends for key metrics
        key_metrics = ['voltage_mean', 'current_mean', 'temp_mean', 'capacity', 'SOH']
        
        for metric in key_metrics:
            if metric in df.columns:
                # Calculate smoothed values using Savitzky-Golay filter
                smoothed = savgol_filter(df[metric], window_length=5, polyorder=2)
                
                # Calculate trend features
                df[f'{metric}_trend'] = np.gradient(smoothed)
                df[f'{metric}_acceleration'] = np.gradient(df[f'{metric}_trend'])
                
                # Calculate trend stability
                df[f'{metric}_trend_stability'] = df[f'{metric}_trend'].rolling(
                    window=5, min_periods=1
                ).std()
        
        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different metrics."""
        # Define important feature pairs for interaction
        interaction_pairs = [
            ('voltage_efficiency', 'temp_stress'),
            ('current_integral', 'temp_max'),
            ('power_efficiency', 'voltage_stability'),
            ('capacity_degradation_rate', 'temp_stress')
        ]
        
        for feat1, feat2 in interaction_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplication interaction
                df[f'{feat1}_{feat2}_interaction'] = df[feat1] * df[feat2]
                
                # Ratio interaction (avoiding division by zero)
                df[f'{feat1}_{feat2}_ratio'] = df[feat1] / df[feat2].replace(0, np.nan)
                
                # Sum interaction
                df[f'{feat1}_{feat2}_sum'] = df[feat1] + df[feat2]
        
        return df

    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio-based features."""
        if all(col in df.columns for col in ['voltage_max', 'voltage_min']):
            df['voltage_range_ratio'] = df['voltage_max'] / df['voltage_min'].replace(0, np.nan)
        
        if all(col in df.columns for col in ['temp_max', 'temp_mean']):
            df['temp_deviation_ratio'] = df['temp_max'] / df['temp_mean'].replace(0, np.nan)
        
        if all(col in df.columns for col in ['power_efficiency', 'voltage_efficiency']):
            df['efficiency_ratio'] = (
                df['power_efficiency'] / df['voltage_efficiency'].replace(0, np.nan)
            )
        
        # Normalize ratios to [0,1] range
        ratio_features = [col for col in df.columns if 'ratio' in col]
        for feat in ratio_features:
            df[feat] = df[feat].clip(lower=0, upper=df[feat].quantile(0.99))
            df[feat] = (df[feat] - df[feat].min()) / (df[feat].max() - df[feat].min())
        
        return df

    def _create_complexity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create complexity-based features."""
        for group, features in self.base_features.items():
            group_features = [f for f in features if f in df.columns]
            if group_features:
                # Calculate group complexity features
                group_values = df[group_features]
                
                # Entropy-based complexity
                df[f'{group}_entropy'] = -(group_values * np.log2(
                    np.abs(group_values.replace(0, 1))
                )).sum(axis=1)
                
                # Statistical complexity
                df[f'{group}_complexity'] = np.sqrt(
                    (group_values ** 2).sum(axis=1) / len(group_features)
                )
                
                # Feature correlation
                df[f'{group}_correlation'] = group_values.corr().mean().mean()
        
        return df

    def _create_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create moving window features."""
        # Define windows for different timeframes
        windows = [3, 5, 10]
        
        key_metrics = [
            'voltage_stability', 'temp_stress', 'power_efficiency',
            'capacity_degradation_rate', 'SOH_degradation_rate'
        ]
        
        for metric in key_metrics:
            if metric in df.columns:
                for window in windows:
                    # Rolling statistics
                    df[f'{metric}_roll_mean_{window}'] = df[metric].rolling(
                        window=window, min_periods=1
                    ).mean()
                    
                    df[f'{metric}_roll_std_{window}'] = df[metric].rolling(
                        window=window, min_periods=1
                    ).std()
                    
                    # Rate of change
                    df[f'{metric}_roll_rate_{window}'] = df[metric].diff(window) / window
        
        return df

    def _get_feature_names(self, feature_type: str) -> List[str]:
        """Get list of features of a specific type."""
        if not self.feature_columns:
            return []
        
        return [
            col for col in self.feature_columns 
            if any(type_indicator in col for type_indicator in [
                f'_{feature_type}', f'{feature_type}_'
            ])
        ]

    def get_feature_names(self) -> List[str]:
        """Return the list of all engineered feature names."""
        if self.feature_columns is None:
            raise ValueError("No features have been created yet. Run create_features first.")
        return self.feature_columns

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by their type for importance analysis."""
        if not self.feature_columns:
            return {}
            
        feature_groups = {
            'trend_features': self._get_feature_names('trend'),
            'interaction_features': self._get_feature_names('interaction'),
            'ratio_features': self._get_feature_names('ratio'),
            'complexity_features': self._get_feature_names('complexity'),
            'window_features': self._get_feature_names('window')
        }
        
        return {k: v for k, v in feature_groups.items() if v}