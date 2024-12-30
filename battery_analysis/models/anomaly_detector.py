from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .base_model import BaseModel
from ..utils.config import Config
from ..utils.logger import setup_logger

# Initialize logger for this module
logger = setup_logger(__name__)

class AnomalyDetector(BaseModel):
    def __init__(self, config: Config = None):
        super().__init__()
        self.config = config or Config()
        # Filter parameters for IsolationForest
        isolation_forest_params = {
            'n_estimators': self.config.ANOMALY_DETECTOR_PARAMS.get('n_estimators', 100),
            'contamination': self.config.ANOMALY_DETECTOR_PARAMS.get('contamination', 0.1),
            'random_state': self.config.ANOMALY_DETECTOR_PARAMS.get('random_state', 42),
            'n_jobs': self.config.ANOMALY_DETECTOR_PARAMS.get('n_jobs', -1)
        }
        self.model = IsolationForest(**isolation_forest_params)
        self.scaler = StandardScaler()


    def train(self, X: pd.DataFrame, y: pd.Series = None) -> Dict:
        """Train the anomaly detection model."""
        logger.info("Training anomaly detection model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        anomalies = np.where(predictions == -1, 1, 0)
        
        # Calculate metrics
        metrics = {
            'anomaly_ratio': float(np.mean(anomalies)),
            'total_anomalies': int(np.sum(anomalies))
        }
        
        logger.info("Model training completed.")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict anomalies in new data."""
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return pd.Series(np.where(predictions == -1, 1, 0))
    
    def analyze_anomalies(self, df: pd.DataFrame, predictions: pd.Series) -> Dict:
        """Analyze detected anomalies."""
        anomaly_stats = {
            'total_anomalies': int(np.sum(predictions)),
            'anomaly_percentage': float(np.mean(predictions) * 100),
            'anomalies_by_cycle': df[predictions == 1]['cycle'].value_counts().to_dict()
        }
        return anomaly_stats