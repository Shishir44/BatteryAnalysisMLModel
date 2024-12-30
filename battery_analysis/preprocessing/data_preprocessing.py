import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class DataPreprocessor:
    def __init__(self):
        """Initialize the DataPreprocessor with default settings."""
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = None
    
    def preprocess_data(self, df: pd.DataFrame, config: Config = None, target: str = 'SOH') -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocess the raw battery data.
        
        Args:
            df: Raw data DataFrame
            config: Configuration object (optional)
            target: Target variable name
            
        Returns:
            Preprocessed DataFrame and preprocessing statistics
        """
        logger.info("Starting data preprocessing...")
        
        # Make a copy of the DataFrame
        df_processed = df.copy()
        stats = {
            'initial_rows': len(df),
            'initial_columns': len(df.columns),
            'missing_values': {},
            'outliers': {}
        }
        
        try:
            # Handle missing values
            df_processed = self._handle_missing_values(df_processed)
            stats['missing_values'] = {
                col: df[col].isna().sum() for col in df.columns
            }
            
            # Remove outliers using config if provided
            if config and hasattr(config, 'PREPROCESSING'):
                iqr_multiplier = config.PREPROCESSING.get('iqr_multiplier', 1.5)
                df_processed = self._remove_outliers(df_processed, iqr_multiplier)
            else:
                df_processed = self._remove_outliers(df_processed)
            
            stats['rows_after_outlier_removal'] = len(df_processed)
            stats['outliers_removed'] = stats['initial_rows'] - len(df_processed)
            
            # Scale features
            df_processed = self._scale_features(df_processed, exclude=[target])
            
            # Store feature columns for later use
            self.feature_columns = [col for col in df_processed.columns if col != target]
            self.target_column = target
            
            # Final statistics
            stats['final_rows'] = len(df_processed)
            stats['final_columns'] = len(df_processed.columns)
            
            logger.info(f"Preprocessing completed. Rows: {stats['initial_rows']} -> {stats['final_rows']}")
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
        
        return df_processed, stats
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        Uses linear interpolation for numeric columns.
        """
        try:
            # For numeric columns, use interpolation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
            
            # Fill any remaining NaNs with forward fill then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, iqr_multiplier: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using IQR method.
        
        Args:
            df: Input DataFrame
            iqr_multiplier: Multiplier for IQR to determine outlier bounds
        """
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            raise
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
        """
        Scale features using StandardScaler.
        
        Args:
            df: Input DataFrame
            exclude: List of columns to exclude from scaling
        """
        try:
            # Select columns to scale (exclude target and specified columns)
            cols_to_scale = [col for col in df.columns if col not in exclude]
            
            # Fit and transform
            scaled_features = self.scaler.fit_transform(df[cols_to_scale])
            
            # Create new DataFrame with scaled features
            df_scaled = pd.DataFrame(scaled_features, columns=cols_to_scale, index=df.index)
            
            # Add back excluded columns
            for col in exclude:
                df_scaled[col] = df[col]
            
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise
        
        return df_scaled
    
    def get_feature_columns(self) -> List[str]:
        """Return the list of feature columns."""
        if self.feature_columns is None:
            raise ValueError("Feature columns not set. Run preprocess_data first.")
        return self.feature_columns
    
    def get_target_column(self) -> str:
        """Return the target column name."""
        if self.target_column is None:
            raise ValueError("Target column not set. Run preprocess_data first.")
        return self.target_column