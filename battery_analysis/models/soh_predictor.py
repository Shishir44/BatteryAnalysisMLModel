from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from .base_model import BaseModel
from ..utils.config import Config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class SOHPredictor(BaseModel):
    """
    State of Health (SOH) predictor optimized for comprehensive battery data.
    Supports multiple model types with automatic feature selection and importance analysis.
    """
    
    def __init__(self, model_type: str = 'xgboost', config: Config = None):
        super().__init__()
        self.model_type = model_type
        self.config = config or Config()
        self.feature_importance = None
        self.selected_features = None
        self._initialize_model()
        
        # Define feature groups for SOH prediction
        self.feature_groups = {
            'voltage_features': [
                'voltage_mean', 'voltage_std', 'voltage_max', 'voltage_min',
                'voltage_range', 'voltage_skew', 'voltage_kurtosis',
                'voltage_efficiency', 'voltage_stability'
            ],
            'current_features': [
                'current_mean', 'current_std', 'current_integral'
            ],
            'temperature_features': [
                'temp_mean', 'temp_max', 'temp_rise', 'temp_integral',
                'temp_std', 'temp_stress'
            ],
            'power_features': [
                'energy_delivered', 'avg_power', 'max_power',
                'power_efficiency'
            ],
            'degradation_features': [
                'capacity_retention', 'capacity_degradation',
                'capacity_degradation_rate'
            ]
        }

    def _initialize_model(self):
        """Initialize the ML model based on configuration."""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=8,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.2,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                early_stopping_rounds=50,
                n_jobs=-1
            )
        elif self.model_type == 'gbm':
            self.model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=5,
                random_state=42
            )
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _select_features(self, X: pd.DataFrame) -> List[str]:
        """
        Select relevant features for SOH prediction based on available columns.
        
        Args:
            X: Input feature DataFrame
            
        Returns:
            List of selected feature names
        """
        selected_features = []
        
        for group, features in self.feature_groups.items():
            # Get available features from each group
            available_features = [f for f in features if f in X.columns]
            selected_features.extend(available_features)
            
            if not available_features:
                logger.warning(f"No features available from group: {group}")
        
        if not selected_features:
            raise ValueError("No relevant features found in the dataset")
        
        return selected_features

    def _validate_features(self, X: pd.DataFrame, selected_features: List[str]):
        """Validate feature quality and log warnings."""
        for feature in selected_features:
            # Check for missing values
            missing_pct = X[feature].isnull().mean() * 100
            if missing_pct > 0:
                logger.warning(f"Feature {feature} has {missing_pct:.2f}% missing values")
            
            # Check for zero variance
            if X[feature].std() == 0:
                logger.warning(f"Feature {feature} has zero variance")
            
            # Check for outliers (using IQR method)
            Q1 = X[feature].quantile(0.25)
            Q3 = X[feature].quantile(0.75)
            IQR = Q3 - Q1
            outlier_pct = (
                ((X[feature] < (Q1 - 1.5 * IQR)) | 
                 (X[feature] > (Q3 + 1.5 * IQR))).mean() * 100
            )
            if outlier_pct > 10:
                logger.warning(
                    f"Feature {feature} has {outlier_pct:.2f}% potential outliers"
                )

    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        validation_size: float = 0.2
    ) -> Dict:
        """
        Train the SOH prediction model.
        
        Args:
            X: Feature matrix
            y: Target SOH values
            validation_size: Size of validation set
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info(f"Training {self.model_type} model for SOH prediction...")
        
        try:
            # Select and validate features
            self.selected_features = self._select_features(X)
            self._validate_features(X, self.selected_features)
            
            X = X[self.selected_features].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=validation_size,
                random_state=42
            )
            
            # Train model
            if self.model_type == 'xgboost':
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric=['rmse', 'mae'],
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
            
            # Calculate predictions
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                y_train, train_pred, y_val, val_pred, X, y
            )
            
            # Calculate feature importance
            self._calculate_feature_importance(X)
            
            logger.info(f"Model training completed. Validation RMSE: {metrics['val_rmse']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def _calculate_metrics(
        self,
        y_train: pd.Series,
        train_pred: np.ndarray,
        y_val: pd.Series,
        val_pred: np.ndarray,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict:
        """Calculate comprehensive model metrics."""
        metrics = {
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
            'train_mae': float(mean_absolute_error(y_train, train_pred)),
            'train_r2': float(r2_score(y_train, train_pred)),
            'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred))),
            'val_mae': float(mean_absolute_error(y_val, val_pred)),
            'val_r2': float(r2_score(y_val, val_pred))
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=5,
            scoring='neg_root_mean_squared_error'
        )
        metrics['cv_rmse_mean'] = float(-cv_scores.mean())
        metrics['cv_rmse_std'] = float(cv_scores.std())
        
        return metrics

    def _calculate_feature_importance(self, X: pd.DataFrame):
        """Calculate and store feature importance scores."""
        if not hasattr(self.model, 'feature_importances_'):
            return
        
        # Get feature importance scores
        importance = self.model.feature_importances_
        
        # Create feature importance dictionary
        self.feature_importance = {
            'individual': dict(zip(X.columns, importance)),
            'group': {}
        }
        
        # Calculate group importance
        for group, features in self.feature_groups.items():
            group_features = [f for f in features if f in X.columns]
            if group_features:
                group_importance = np.mean([
                    importance[list(X.columns).index(f)]
                    for f in group_features
                ])
                self.feature_importance['group'][group] = float(group_importance)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Make SOH predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Series of predicted SOH values
        """
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        
        if self.selected_features is None:
            raise ValueError("No features selected. Train the model first")
        
        try:
            # Select features and handle missing values
            X = X[self.selected_features].copy()
            X = X.fillna(X.mean())
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Ensure predictions are within valid SOH range [0, 100]
            predictions = np.clip(predictions, 0, 100)
            
            return pd.Series(predictions, index=X.index)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def get_feature_importance(self, top_n: Optional[int] = None) -> Dict:
        """
        Get feature importance analysis.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary containing feature importance analysis
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated. Train the model first")
        
        result = {
            'group_importance': self.feature_importance['group'],
            'individual_importance': self.feature_importance['individual']
        }
        
        if top_n is not None:
            sorted_features = sorted(
                result['individual_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            result['top_features'] = dict(sorted_features[:top_n])
        
        return result