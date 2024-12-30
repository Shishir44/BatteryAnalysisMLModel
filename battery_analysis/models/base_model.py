from abc import ABC, abstractmethod
import pandas as pd
import joblib
from typing import Dict, Tuple
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseModel(ABC):
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self.feature_columns = None
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train the model and return metrics."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions using the trained model."""
        pass
    
    def save_model(self, path: str = None):
        """Save the trained model."""
        save_path = path or self.model_path
        if save_path and self.model:
            joblib.dump(self.model, save_path)
            logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str = None):
        """Load a trained model."""
        load_path = path or self.model_path
        if load_path:
            self.model = joblib.load(load_path)
            logger.info(f"Model loaded from {load_path}")