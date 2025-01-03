from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from .base_model import BaseModel
from ..utils.config import Config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class AnomalyDetector(BaseModel):
    """
    Enhanced anomaly detector for battery data using multiple detection methods
    and feature-group-based analysis.
    """
    
    def __init__(self, config: Config = None):
        """Initialize anomaly detector with multiple models."""
        super().__init__()
        self.config = config or Config()
        
        # Initialize feature groups
        self.feature_groups = {
            'voltage_features': [
                'voltage_mean', 'voltage_std', 'voltage_stability',
                'voltage_efficiency', 'voltage_range'
            ],
            'thermal_features': [
                'temp_mean', 'temp_max', 'temp_stress',
                'temp_integral', 'temp_std'
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
        
        # Initialize models for each feature group
        self.models = self._initialize_models()
        self.scalers = {group: StandardScaler() for group in self.feature_groups}
        
    def _initialize_models(self) -> Dict:
        """Initialize anomaly detection models for each feature group."""
        models = {}
        
        # Base parameters for each model type
        iforest_params = {
            'n_estimators': self.config.ANOMALY_DETECTOR_PARAMS.get('n_estimators', 100),
            'contamination': self.config.ANOMALY_DETECTOR_PARAMS.get('contamination', 0.1),
            'random_state': self.config.ANOMALY_DETECTOR_PARAMS.get('random_state', 42),
            'n_jobs': self.config.ANOMALY_DETECTOR_PARAMS.get('n_jobs', -1)
        }
        
        robust_params = {
            'contamination': self.config.ANOMALY_DETECTOR_PARAMS.get('contamination', 0.1),
            'random_state': self.config.ANOMALY_DETECTOR_PARAMS.get('random_state', 42)
        }
        
        ocsvm_params = {
            'kernel': 'rbf',
            'nu': self.config.ANOMALY_DETECTOR_PARAMS.get('contamination', 0.1),
            'random_state': self.config.ANOMALY_DETECTOR_PARAMS.get('random_state', 42)
        }
        
        # Initialize models for each feature group
        for group in self.feature_groups:
            models[group] = {
                'iforest': IsolationForest(**iforest_params),
                'robust': EllipticEnvelope(**robust_params),
                'ocsvm': OneClassSVM(**ocsvm_params)
            }
        
        return models

    def train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict:
        """
        Train anomaly detection models for each feature group.
        
        Args:
            X: Input features
            y: Optional target variable (not used in unsupervised anomaly detection)
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info("Training anomaly detection models...")
        
        metrics = {}
        self.feature_scores = {}
        
        try:
            for group, features in self.feature_groups.items():
                # Get available features for this group
                available_features = [f for f in features if f in X.columns]
                
                if not available_features:
                    logger.warning(f"No features available for group: {group}")
                    continue
                
                # Scale features
                X_group = self.scalers[group].fit_transform(X[available_features])
                
                # Train models for this group
                group_predictions = {}
                for model_name, model in self.models[group].items():
                    model.fit(X_group)
                    group_predictions[model_name] = model.predict(X_group)
                
                # Combine predictions using majority voting
                combined_pred = np.vstack(list(group_predictions.values()))
                majority_vote = np.apply_along_axis(
                    lambda x: -1 if np.sum(x == -1) >= 2 else 1, 
                    axis=0, 
                    arr=combined_pred
                )
                
                # Calculate anomaly scores
                anomaly_scores = self._calculate_anomaly_scores(X_group, group)
                self.feature_scores[group] = anomaly_scores
                
                # Store metrics
                metrics[group] = {
                    'anomaly_ratio': float(np.mean(majority_vote == -1)),
                    'mean_anomaly_score': float(np.mean(anomaly_scores)),
                    'max_anomaly_score': float(np.max(anomaly_scores))
                }
            
            logger.info("Anomaly detection models trained successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training anomaly detection models: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict anomalies using ensemble of models.
        
        Args:
            X: Input features
            
        Returns:
            Series of anomaly predictions (-1 for anomalies, 1 for normal)
        """
        group_predictions = []
        group_weights = []
        
        for group, features in self.feature_groups.items():
            available_features = [f for f in features if f in X.columns]
            
            if not available_features:
                continue
            
            # Scale features
            X_group = self.scalers[group].transform(X[available_features])
            
            # Get predictions from all models in the group
            model_predictions = []
            for model in self.models[group].values():
                pred = model.predict(X_group)
                model_predictions.append(pred)
            
            # Combine predictions for this group
            group_pred = np.apply_along_axis(
                lambda x: -1 if np.sum(x == -1) >= 2 else 1,
                axis=0,
                arr=np.vstack(model_predictions)
            )
            
            group_predictions.append(group_pred)
            # Weight based on feature group importance
            group_weights.append(len(available_features))
        
        # Weighted majority voting across groups
        if not group_predictions:
            raise ValueError("No predictions available from any feature group")
        
        final_predictions = np.average(
            np.vstack(group_predictions),
            axis=0,
            weights=group_weights
        )
        
        return pd.Series(np.where(final_predictions < 0, -1, 1), index=X.index)

    def _calculate_anomaly_scores(self, X: np.ndarray, group: str) -> np.ndarray:
        """Calculate anomaly scores using multiple methods."""
        scores = []
        
        # Get scores from each model
        for model_name, model in self.models[group].items():
            if hasattr(model, 'score_samples'):
                # Normalize scores to [0,1] range
                score = model.score_samples(X)
                score = (score - np.min(score)) / (np.max(score) - np.min(score))
                scores.append(score)
            elif hasattr(model, 'decision_function'):
                score = model.decision_function(X)
                score = (score - np.min(score)) / (np.max(score) - np.min(score))
                scores.append(score)
        
        # Combine scores using average
        return np.mean(scores, axis=0)

    def analyze_anomalies(self, df: pd.DataFrame, predictions: pd.Series) -> Dict:
        """
        Analyze detected anomalies in detail.
        
        Args:
            df: Input DataFrame
            predictions: Anomaly predictions
            
        Returns:
            Dictionary containing anomaly analysis results
        """
        analysis = {
            'total_anomalies': int(np.sum(predictions == -1)),
            'anomaly_percentage': float(np.mean(predictions == -1) * 100),
            'group_analysis': {}
        }
        
        # Analyze anomalies by feature group
        for group, features in self.feature_groups.items():
            available_features = [f for f in features if f in df.columns]
            
            if not available_features:
                continue
                
            group_stats = {
                'feature_importance': {},
                'anomaly_characteristics': {}
            }
            
            # Calculate feature importance for anomaly detection
            if group in self.feature_scores:
                for feature, score in zip(available_features, self.feature_scores[group]):
                    group_stats['feature_importance'][feature] = float(np.mean(score))
            
            # Analyze characteristics of anomalies
            anomaly_data = df[predictions == -1][available_features]
            normal_data = df[predictions == 1][available_features]
            
            for feature in available_features:
                group_stats['anomaly_characteristics'][feature] = {
                    'mean_diff': float(
                        anomaly_data[feature].mean() - normal_data[feature].mean()
                    ),
                    'std_diff': float(
                        anomaly_data[feature].std() - normal_data[feature].std()
                    ),
                    'max_deviation': float(
                        np.max(np.abs(anomaly_data[feature] - normal_data[feature].mean()))
                    )
                }
            
            analysis['group_analysis'][group] = group_stats
        
        return analysis