from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple, List, Optional, Union
from pathlib import Path
from datetime import datetime
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseModel(ABC):
    """Base class for all battery analysis models with enhanced functionality."""
    
    def __init__(self, config: Optional[object] = None):
        """
        Initialize base model.
        
        Args:
            config: Configuration object containing model parameters
        """
        self.model = None
        self.config = config
        self.feature_columns = None
        self.feature_importance = {}
        self.training_history = {}
        self.model_metadata = {
            'training_date': None,
            'model_version': '1.0.0',
            'feature_columns': None,
            'training_metrics': None,
            'data_statistics': None,
            'hyperparameters': None
        }
        self.validation_thresholds = {
            'missing_values_threshold': 0.1,  # 10% missing values allowed
            'outlier_threshold': 3.0,         # IQR multiplier for outlier detection
            'correlation_threshold': 0.95      # Maximum allowed correlation between features
        }
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train the model and return metrics.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Series containing predictions
        """
        pass
    
    def validate_features(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate input features and return validation report.
        
        Args:
            X: Feature matrix to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation_report = {
            'missing_features': [],
            'high_missing_values': [],
            'high_correlation': [],
            'constant_features': [],
            'warnings': []
        }
        
        # Check for required features
        if self.feature_columns is not None:
            missing_features = set(self.feature_columns) - set(X.columns)
            validation_report['missing_features'] = list(missing_features)
            
            if missing_features:
                logger.warning(f"Missing required features: {missing_features}")
        
        # Check for missing values
        missing_pct = X.isnull().mean()
        high_missing = missing_pct[missing_pct > self.validation_thresholds['missing_values_threshold']]
        validation_report['high_missing_values'] = list(high_missing.index)
        
        # Check for constant features
        constant_features = [col for col in X.columns if X[col].nunique() <= 1]
        validation_report['constant_features'] = constant_features
        
        # Check for highly correlated features
        correlation_matrix = X.corr().abs()
        upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        high_corr = [(i, j) for i, j in zip(*np.where(upper > self.validation_thresholds['correlation_threshold']))]
        validation_report['high_correlation'] = [(correlation_matrix.index[i], correlation_matrix.columns[j]) 
                                               for i, j in high_corr]
        
        return validation_report
    
    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the trained model and all metadata.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update metadata before saving
        self._update_metadata()
        
        # Prepare save data
        save_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'model_metadata': self.model_metadata,
            'validation_thresholds': self.validation_thresholds
        }
        
        joblib.dump(save_data, path)
        logger.info(f"Model and metadata saved to {path}")
    
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load a trained model and all metadata.
        
        Args:
            path: Path to load the model from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load model data
        load_data = joblib.load(path)
        
        # Restore model and attributes
        self.model = load_data['model']
        self.feature_columns = load_data['feature_columns']
        self.feature_importance = load_data.get('feature_importance', {})
        self.training_history = load_data.get('training_history', {})
        self.model_metadata = load_data.get('model_metadata', {})
        self.validation_thresholds = load_data.get('validation_thresholds', self.validation_thresholds)
        
        logger.info(f"Model and metadata loaded from {path}")
    
    def _update_metadata(self) -> None:
        """Update model metadata."""
        self.model_metadata.update({
            'last_updated': datetime.now().isoformat(),
            'feature_columns': self.feature_columns,
            'n_features': len(self.feature_columns) if self.feature_columns else None,
            'model_parameters': getattr(self.model, 'get_params', lambda: {})()
        })
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> Dict:
        """
        Get feature importance analysis.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary containing feature importance analysis
        """
        if not self.feature_importance:
            logger.warning("Feature importance not calculated")
            return {}
        
        importance_analysis = {
            'all_features': self.feature_importance.copy()
        }
        
        if top_n is not None:
            sorted_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            importance_analysis['top_features'] = dict(sorted_features[:top_n])
        
        return importance_analysis
    
    def _validate_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Validate input data quality.
        
        Args:
            X: Feature matrix
            y: Optional target values
        """
        # Check basic data quality
        if X.empty:
            raise ValueError("Empty feature matrix")
        
        if y is not None and len(X) != len(y):
            raise ValueError("Feature matrix and target values have different lengths")
        
        # Validate features
        validation_report = self.validate_features(X)
        
        # Log validation warnings
        if any(validation_report.values()):
            logger.warning("Data validation found issues: %s", validation_report)
        
        # Check target if provided
        if y is not None:
            if y.isnull().any():
                raise ValueError("Target values contain missing values")
            if np.isinf(y).any():
                raise ValueError("Target values contain infinite values")
    
    def get_model_info(self) -> Dict:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model information and statistics
        """
        return {
            'metadata': self.model_metadata,
            'feature_information': {
                'n_features': len(self.feature_columns) if self.feature_columns else 0,
                'feature_list': self.feature_columns,
                'feature_importance': self.get_feature_importance()
            },
            'training_history': self.training_history,
            'model_parameters': getattr(self.model, 'get_params', lambda: {})()
        }