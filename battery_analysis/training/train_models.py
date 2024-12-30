# battery_ml/training/train_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
from ..preprocessing import DataPreprocessor, FeatureEngineer
from ..models import SOHPredictor, CapacityPredictor, AnomalyDetector
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class ModelTrainer:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer(self.config)
        self.models = {
            'soh': SOHPredictor(model_type='xgboost', config=self.config),
            'capacity': CapacityPredictor(config=self.config),
            'anomaly': AnomalyDetector(config=self.config)
        }
        self.metrics = {}
    
    def train_all_models(self, data_path: str) -> Dict:
        """Train all models in the pipeline."""
        logger.info("Starting model training pipeline...")
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Preprocess data
        df_processed, _ = self.preprocessor.preprocess_data(df)
        
        # Engineer features
        df_featured = self.feature_engineer.create_features(df_processed)
        
        # Train each model
        self.metrics['soh'] = self._train_soh_model(df_featured)
        self.metrics['capacity'] = self._train_capacity_model(df_featured)
        self.metrics['anomaly'] = self._train_anomaly_model(df_featured)
        
        logger.info("Model training pipeline completed.")
        return self.metrics
    
    def _train_soh_model(self, df: pd.DataFrame) -> Dict:
        """Train SOH prediction model."""
        logger.info("Training SOH prediction model...")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['cycle', 'SOH']]
        X = df[feature_cols]
        y = df['SOH']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.TRAIN_TEST_SPLIT,
            random_state=self.config.RANDOM_STATE
        )
        
        # Train model
        metrics = self.models['soh'].train(X_train, y_train)
        
        # Save model
        self.models['soh'].save_model(self.config.MODELS_DIR / 'soh_model.joblib')
        
        return metrics
    
    def _train_capacity_model(self, df: pd.DataFrame) -> Dict:
        """Train capacity prediction model."""
        logger.info("Training capacity prediction model...")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['cycle', 'capacity']]
        X = df[feature_cols]
        y = df['capacity']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TRAIN_TEST_SPLIT,
            random_state=self.config.RANDOM_STATE
        )
        
        # Train model
        metrics = self.models['capacity'].train(X_train, y_train)
        
        # Save model
        self.models['capacity'].save_model(self.config.MODELS_DIR / 'capacity_model.joblib')
        
        return metrics
    
    def _train_anomaly_model(self, df: pd.DataFrame) -> Dict:
        """Train anomaly detection model."""
        logger.info("Training anomaly detection model...")
        
        # Prepare features
        feature_cols = ['terminal_voltage', 'terminal_current', 'temperature',
                       'power', 'internal_resistance']
        X = df[feature_cols]
        
        # Train model
        metrics = self.models['anomaly'].train(X)
        
        # Save model
        self.models['anomaly'].save_model(self.config.MODELS_DIR / 'anomaly_model.joblib')
        
        return metrics

# Example usage script
if __name__ == "__main__":
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train all models
    metrics = trainer.train_all_models('data/raw/merged_data.csv')
    
    # Print metrics
    print("\nTraining Metrics:")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name.upper()} Model:")
        for metric_name, value in model_metrics.items():
            print(f"{metric_name}: {value}")