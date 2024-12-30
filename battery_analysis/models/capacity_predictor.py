from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from typing import Dict
from .base_model import BaseModel
from ..utils.config import Config
from ..utils.logger import setup_logger

# Initialize logger for this module
logger = setup_logger(__name__)

class CapacityPredictor(BaseModel):
    def __init__(self, config: Config = None):
        super().__init__()
        self.config = config or Config()
        # Filter out incompatible parameters
        gbr_params = {
            'n_estimators': self.config.CAPACITY_MODEL_PARAMS.get('n_estimators', 300),
            'learning_rate': self.config.CAPACITY_MODEL_PARAMS.get('learning_rate', 0.03),
            'max_depth': self.config.CAPACITY_MODEL_PARAMS.get('max_depth', 8),
            'subsample': self.config.CAPACITY_MODEL_PARAMS.get('subsample', 0.8),
            'random_state': self.config.CAPACITY_MODEL_PARAMS.get('random_state', 42)
        }
        self.model = GradientBoostingRegressor(**gbr_params)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train the capacity prediction model."""
        logger.info("Training capacity prediction model...")
        
        # Train model
        self.model.fit(X, y)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }
        
        logger.info("Model training completed.")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        return pd.Series(self.model.predict(X))