import pandas as pd
import numpy as np
from typing import List, Dict
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class FeatureEngineer:
    def __init__(self, config: Config):
        """
        Initialize the FeatureEngineer with configuration.
        
        Args:
            config: Configuration object containing feature engineering parameters
        """
        self.config = config
        self.feature_columns = None
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for battery analysis.
        
        Args:
            df: Input DataFrame with battery data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        df_featured = df.copy()
        
        try:
            # Validate required columns
            self._validate_required_columns(df_featured)
            
            # Create cycle column if it doesn't exist
            df_featured = self._ensure_cycle_column(df_featured)
            
            # Basic features
            logger.info("Creating basic features...")
            df_featured = self._create_basic_features(df_featured)
            
            # Time series features
            logger.info("Creating time series features...")
            df_featured = self._create_time_series_features(df_featured)
            
            # Statistical features
            logger.info("Creating statistical features...")
            df_featured = self._create_statistical_features(df_featured)
            
            # Advanced features
            logger.info("Creating advanced features...")
            df_featured = self._create_advanced_features(df_featured)
            
            # Store feature columns for later use
            self.feature_columns = [col for col in df_featured.columns if col not in df.columns]
            
            logger.info(f"Created {len(self.feature_columns)} new features")
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
            
        return df_featured
    
    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist in the DataFrame."""
        required_columns = [
            'terminal_voltage', 'terminal_current', 'temperature',
            'charge_voltage', 'charge_current', 'time', 'capacity', 'SOH'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def _ensure_cycle_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure cycle column exists, create it if it doesn't."""
        if 'cycle' not in df.columns:
            logger.info("'cycle' column not found, creating it from time series data...")
            
            # Sort by time to ensure proper cycle detection
            df = df.sort_values('time')
            
            # Detect cycles based on time discontinuities
            time_diff = df['time'].diff()
            
            # A new cycle starts when:
            # 1. Time difference is negative (time resets)
            # 2. Time difference is too large (gap between cycles)
            # Use median time difference * 10 as threshold for large gaps
            median_diff = time_diff.median()
            cycle_starts = (time_diff < 0) | (time_diff > median_diff * 10)
            
            # Create cycle numbers
            df['cycle'] = cycle_starts.cumsum()
            
            logger.info(f"Created {df['cycle'].nunique()} cycles")
        
        return df
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic engineered features."""
        try:
            # Power features
            df['power'] = df['terminal_voltage'] * df['terminal_current']
            df['charge_power'] = df['charge_voltage'] * df['charge_current']
            
            # Efficiency features
            df['voltage_efficiency'] = np.where(
                df['charge_voltage'] != 0,
                df['terminal_voltage'] / df['charge_voltage'],
                0
            )
            df['current_efficiency'] = np.where(
                df['charge_current'] != 0,
                df['terminal_current'] / df['charge_current'],
                0
            )
            
            # Resistance features
            df['internal_resistance'] = np.where(
                df['terminal_current'] != 0,
                (df['charge_voltage'] - df['terminal_voltage']) / df['terminal_current'],
                0
            )
            
            # Energy features
            df['energy'] = df['power'] * df['time'].diff()
            df['charge_energy'] = df['charge_power'] * df['time'].diff()
            
            # Overall efficiency
            df['energy_efficiency'] = np.where(
                df['charge_energy'] != 0,
                df['energy'] / df['charge_energy'],
                0
            )
            
        except Exception as e:
            logger.error(f"Error in creating basic features: {str(e)}")
            raise
            
        return df
    
    def _create_time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time series based features."""
        try:
            # Get window sizes from config
            window_sizes = self.config.FEATURE_ENGINEERING.get('window_sizes', [5, 10, 20])
            
            # Key parameters to analyze
            parameters = ['terminal_voltage', 'terminal_current', 'temperature', 
                        'power', 'internal_resistance']
            
            for window in window_sizes:
                for col in parameters:
                    if col in df.columns:
                        # Rolling statistics per cycle
                        df[f'{col}_rolling_mean_{window}'] = df.groupby('cycle')[col].transform(
                            lambda x: x.rolling(window=window, min_periods=1).mean()
                        )
                        df[f'{col}_rolling_std_{window}'] = df.groupby('cycle')[col].transform(
                            lambda x: x.rolling(window=window, min_periods=1).std()
                        )
                        
                        # Rate of change
                        df[f'{col}_rate_{window}'] = df.groupby('cycle')[col].transform(
                            lambda x: x.diff(window) / (window * x.index.to_series().diff(window))
                        )
            
        except Exception as e:
            logger.error(f"Error in creating time series features: {str(e)}")
            raise
            
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        try:
            # Verify cycle column exists and type
            if 'cycle' not in df.columns:
                raise ValueError("'cycle' column missing from DataFrame")
            logger.debug(f"Cycle column dtype: {df['cycle'].dtype}")
            
            # Define aggregation functions
            agg_funcs = {
                'terminal_voltage': ['mean', 'std', 'max', 'min', 'skew'],
                'terminal_current': ['mean', 'std', 'max', 'min', 'skew'],
                'temperature': ['mean', 'std', 'max', 'min'],
                'power': ['mean', 'max', 'sum'],
                'internal_resistance': ['mean', 'std'],
                'energy_efficiency': ['mean', 'std']
            }
            
            # Verify all columns exist before aggregation
            missing_cols = [col for col in agg_funcs.keys() if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns for aggregation: {missing_cols}")
            
            # Cycle-level statistics
            cycle_stats = df.groupby('cycle').agg(agg_funcs)
            
            # Debug information
            logger.debug(f"Shape of cycle_stats before flatten: {cycle_stats.shape}")
            logger.debug(f"Columns before flatten: {cycle_stats.columns.tolist()}")
            
            # Flatten column names
            cycle_stats.columns = [f"{col[0]}_{col[1]}" for col in cycle_stats.columns]
            cycle_stats = cycle_stats.reset_index()
            
            # Debug information
            logger.debug(f"Shape of cycle_stats after flatten: {cycle_stats.shape}")
            logger.debug(f"Columns after flatten: {cycle_stats.columns.tolist()}")
            
            # Merge back with original DataFrame
            df = pd.merge(df, cycle_stats, on='cycle', how='left')
            
            # Verify merge results
            logger.debug(f"Shape after merge: {df.shape}")
            
        except Exception as e:
            logger.error(f"Error in creating statistical features: {str(e)}")
            raise
            
        return df
    
    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced engineered features."""
        try:
            # Capacity retention features
            df['capacity_retention'] = df.groupby('cycle')['capacity'].transform(
                lambda x: x / x.iloc[0] if len(x) > 0 else 1
            )
            
            # SOH change rate
            df['soh_change_rate'] = df.groupby('cycle')['SOH'].transform(
                lambda x: x.diff() / x.shift(1)
            )
            
            # Temperature stress indicator
            temp_ranges = self.config.FEATURE_ENGINEERING.get('temp_ranges', [(0, 25), (25, 40), (40, 60)])
            for i, (min_temp, max_temp) in enumerate(temp_ranges):
                df[f'temp_range_{i}'] = ((df['temperature'] >= min_temp) & 
                                       (df['temperature'] < max_temp)).astype(int)
            
            # Voltage stress features
            df['voltage_stress'] = (df['terminal_voltage'] - df['terminal_voltage'].mean()) / df['terminal_voltage'].std()
            
            # Cycle progress
            df['cycle_progress'] = df.groupby('cycle')['time'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) if len(x) > 1 else 0
            )
            
        except Exception as e:
            logger.error(f"Error in creating advanced features: {str(e)}")
            raise
            
        return df

    def get_feature_names(self) -> List[str]:
        """Return the list of engineered feature names."""
        if self.feature_columns is None:
            raise ValueError("No features have been created yet. Run create_features first.")
        return self.feature_columns