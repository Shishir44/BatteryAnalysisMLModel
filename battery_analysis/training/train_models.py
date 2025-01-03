import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional
from pathlib import Path
from ..preprocessing import DataPreprocessor
from ..preprocessing.feature_engineering import FeatureEngineer
from ..models import SOHPredictor, CapacityPredictor, AnomalyDetector
from ..training.model_evaluation import ModelEvaluator
from ..utils.logger import setup_logger
from ..utils.config import Config
import json
import time

logger = setup_logger(__name__)

class ModelTrainer:
    """
    Model training pipeline for comprehensive battery health analysis.
    Handles training, evaluation, and saving of multiple models.
    """
    
    def __init__(self, config: Config = None):
        """Initialize ModelTrainer with configuration."""
        self.config = config or Config()
        self.preprocessor = DataPreprocessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.evaluator = ModelEvaluator(output_dir=self.config.RESULTS_DIR)
        
        # Initialize models
        self.models = {
            'soh': SOHPredictor(model_type='xgboost', config=self.config),
            'capacity': CapacityPredictor(config=self.config),
            'anomaly': AnomalyDetector(config=self.config)
        }
        
        self.training_history = {}
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories for model artifacts."""
        directories = [
            self.config.MODELS_DIR,
            self.config.RESULTS_DIR,
            self.config.PROCESSED_DATA_DIR
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def train_pipeline(
        self,
        data_path: str,
        validation_split: float = 0.2,
        save_results: bool = True
    ) -> Dict:
        """
        Run complete training pipeline for all models.
        
        Args:
            data_path: Path to comprehensive battery dataset
            validation_split: Validation set proportion
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary containing training results and metrics
        """
        start_time = time.time()
        pipeline_results = {}
        
        try:
            # Load and preprocess data
            logger.info("Loading and preprocessing data...")
            df = pd.read_csv(data_path)
            df_processed, preprocessing_stats = self.preprocessor.preprocess_data(df)
            
            # Engineer features
            logger.info("Engineering features...")
            df_featured, feature_stats = self.feature_engineer.create_features(df_processed)
            
            if save_results:
                df_featured.to_csv(
                    self.config.PROCESSED_DATA_DIR / 'processed_features.csv',
                    index=False
                )
            
            # Train and evaluate models
            pipeline_results['preprocessing'] = preprocessing_stats
            pipeline_results['feature_engineering'] = feature_stats
            pipeline_results['models'] = {}
            
            # Train SOH Predictor
            logger.info("Training SOH Predictor...")
            soh_results = self._train_soh_model(df_featured, validation_split)
            pipeline_results['models']['soh'] = soh_results
            
            # Train Capacity Predictor
            logger.info("Training Capacity Predictor...")
            capacity_results = self._train_capacity_model(df_featured, validation_split)
            pipeline_results['models']['capacity'] = capacity_results
            
            # Train Anomaly Detector
            logger.info("Training Anomaly Detector...")
            anomaly_results = self._train_anomaly_model(df_featured)
            pipeline_results['models']['anomaly'] = anomaly_results
            
            # Calculate execution time
            execution_time = time.time() - start_time
            pipeline_results['execution_time'] = execution_time
            
            if save_results:
                self._save_pipeline_results(pipeline_results)
            
            logger.info(f"Training pipeline completed in {execution_time:.2f} seconds")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise

    def _train_soh_model(
        self,
        df: pd.DataFrame,
        validation_split: float
    ) -> Dict:
        """Train and evaluate SOH prediction model."""
        # Prepare features and target
        X = df.drop(['SOH', 'cycle'], axis=1, errors='ignore')
        y = df['SOH']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=self.config.RANDOM_STATE
        )
        
        # Train model
        train_metrics = self.models['soh'].train(X_train, y_train)
        
        # Evaluate model
        evaluation_results = self.evaluator.evaluate_regression_model(
            model=self.models['soh'],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name='SOH_Predictor',
            target_type='SOH'
        )
        
        # Save model
        self.models['soh'].save_model(
            self.config.MODELS_DIR / 'soh_model.joblib'
        )
        
        return {
            'training_metrics': train_metrics,
            'evaluation': evaluation_results,
            'feature_importance': self.models['soh'].get_feature_importance(top_n=10)
        }

    def _train_capacity_model(
        self,
        df: pd.DataFrame,
        validation_split: float
    ) -> Dict:
        """Train and evaluate capacity prediction model."""
        # Prepare features and target
        X = df.drop(['capacity', 'cycle'], axis=1, errors='ignore')
        y = df['capacity']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=self.config.RANDOM_STATE
        )
        
        # Train model
        train_metrics = self.models['capacity'].train(X_train, y_train)
        
        # Evaluate model
        evaluation_results = self.evaluator.evaluate_regression_model(
            model=self.models['capacity'],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name='Capacity_Predictor',
            target_type='capacity'
        )
        
        # Save model
        self.models['capacity'].save_model(
            self.config.MODELS_DIR / 'capacity_model.joblib'
        )
        
        return {
            'training_metrics': train_metrics,
            'evaluation': evaluation_results,
            'feature_importance': self.models['capacity'].get_feature_importance(top_n=10)
        }

    def _train_anomaly_model(self, df: pd.DataFrame) -> Dict:
        """Train and evaluate anomaly detection model."""
        # Prepare features
        X = df.drop(['cycle'], axis=1, errors='ignore')
        
        # Train model
        train_metrics = self.models['anomaly'].train(X)
        
        # Get predictions for evaluation
        predictions = self.models['anomaly'].predict(X)
        
        # Evaluate model
        evaluation_results = self.evaluator.evaluate_anomaly_detector(
            model=self.models['anomaly'],
            X=X,
            predictions=predictions
        )
        
        # Save model
        self.models['anomaly'].save_model(
            self.config.MODELS_DIR / 'anomaly_model.joblib'
        )
        
        return {
            'training_metrics': train_metrics,
            'evaluation': evaluation_results
        }

    def _save_pipeline_results(self, results: Dict):
        """Save pipeline results and metrics."""
        # Save main results
        results_path = self.config.RESULTS_DIR / 'training_results.json'
        
        # Convert numpy types to Python native types
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(i) for i in obj]
            return obj
        
        results = convert_to_native(results)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Training results saved to {results_path}")

def main():
    """Main function to run the training pipeline."""
    try:
        # Initialize configuration
        config = Config()
        
        # Initialize trainer
        trainer = ModelTrainer(config)
        
        # Run training pipeline
        results = trainer.train_pipeline(
            data_path='data/raw/comprehensive_battery_data.csv',
            validation_split=0.2,
            save_results=True
        )
        
        # Print summary metrics
        print("\nTraining Pipeline Results:")
        for model_name, model_results in results['models'].items():
            print(f"\n{model_name.upper()} Model:")
            if 'evaluation' in model_results:
                metrics = model_results['evaluation'].get('metrics', {})
                print(f"Test RMSE: {metrics.get('test_rmse', 'N/A'):.4f}")
                print(f"Test R²: {metrics.get('test_r2', 'N/A'):.4f}")
        
        print(f"\nTotal execution time: {results['execution_time']:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in main training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()