from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .base_model import BaseModel
from ..utils.config import Config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class CapacityPredictor(BaseModel):
    """
    Enhanced capacity predictor for battery health monitoring using comprehensive features.
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the capacity predictor with configuration.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config or Config()
        self.feature_groups = {
            'primary_features': [
                'voltage_mean', 'current_mean', 'temp_mean',
                'voltage_efficiency', 'current_integral'
            ],
            'stability_features': [
                'voltage_stability', 'voltage_rate_stability',
                'temp_stability', 'power_stability'
            ],
            'stress_features': [
                'temp_stress', 'voltage_stress',
                'current_stress', 'power_stress'
            ],
            'degradation_features': [
                'SOH', 'SOH_degradation_rate',
                'voltage_degradation', 'temp_degradation'
            ],
            'efficiency_features': [
                'voltage_efficiency', 'power_efficiency',
                'coulombic_efficiency', 'energy_efficiency'
            ]
        }
        
        # Initialize model with optimized parameters
        self.model = self._initialize_model()
        self.selected_features = None
        self.feature_importance = None
        self.scaler = None

    def _initialize_model(self) -> GradientBoostingRegressor:
        """Initialize the gradient boosting model with optimized parameters."""
        params = {
            'n_estimators': self.config.CAPACITY_MODEL_PARAMS.get('n_estimators', 500),
            'learning_rate': self.config.CAPACITY_MODEL_PARAMS.get('learning_rate', 0.01),
            'max_depth': self.config.CAPACITY_MODEL_PARAMS.get('max_depth', 6),
            'min_samples_split': self.config.CAPACITY_MODEL_PARAMS.get('min_samples_split', 5),
            'min_samples_leaf': self.config.CAPACITY_MODEL_PARAMS.get('min_samples_leaf', 4),
            'subsample': self.config.CAPACITY_MODEL_PARAMS.get('subsample', 0.8),
            'random_state': self.config.CAPACITY_MODEL_PARAMS.get('random_state', 42),
            'validation_fraction': self.config.CAPACITY_MODEL_PARAMS.get('validation_fraction', 0.1),
            'n_iter_no_change': self.config.CAPACITY_MODEL_PARAMS.get('n_iter_no_change', 20),
            'tol': self.config.CAPACITY_MODEL_PARAMS.get('tol', 1e-4)
        }
        
        return GradientBoostingRegressor(**params)

    def _select_features(self, X: pd.DataFrame) -> List[str]:
        """
        Select relevant features for capacity prediction.
        
        Args:
            X: Input feature DataFrame
            
        Returns:
            List of selected feature names
        """
        selected_features = []
        
        for group, features in self.feature_groups.items():
            available_features = [f for f in features if f in X.columns]
            if available_features:
                selected_features.extend(available_features)
                logger.info(f"Selected {len(available_features)} features from {group}")
            else:
                logger.warning(f"No features available from {group}")
        
        if not selected_features:
            raise ValueError("No relevant features found in the dataset")
        
        return selected_features

    def _validate_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Validate input data quality.
        
        Args:
            X: Feature matrix
            y: Target capacity values
        """
        # Check for invalid capacity values
        if (y < 0).any():
            raise ValueError("Negative capacity values found in target")
        
        # Check feature validity
        for feature in X.columns:
            # Check for missing values
            missing_pct = X[feature].isnull().mean() * 100
            if missing_pct > 0:
                logger.warning(f"Feature {feature} has {missing_pct:.2f}% missing values")
            
            # Check for constant features
            if X[feature].std() == 0:
                logger.warning(f"Feature {feature} has zero variance")
            
            # Check for outliers using IQR method
            Q1 = X[feature].quantile(0.25)
            Q3 = X[feature].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((X[feature] < (Q1 - 1.5 * IQR)) | 
                       (X[feature] > (Q3 + 1.5 * IQR))).sum()
            if outliers > len(X) * 0.1:  # More than 10% outliers
                logger.warning(f"Feature {feature} has {outliers} potential outliers")

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train the capacity prediction model.
        
        Args:
            X: Feature matrix
            y: Target capacity values
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info("Training capacity prediction model...")
        
        try:
            # Select features and validate data
            self.selected_features = self._select_features(X)
            X = X[self.selected_features].copy()
            self._validate_data(X, y)
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Calculate predictions
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                y_train, train_pred,
                y_val, val_pred,
                X, y
            )
            
            # Calculate feature importance
            self._calculate_feature_importance(X)
            
            logger.info(f"Training completed. Validation RMSE: {metrics['val_rmse']:.4f}")
            
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
        """Calculate comprehensive model performance metrics."""
        metrics = {
            # Training metrics
            'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
            'train_mae': float(mean_absolute_error(y_train, train_pred)),
            'train_r2': float(r2_score(y_train, train_pred)),
            'train_mape': float(np.mean(np.abs((y_train - train_pred) / y_train)) * 100),
            
            # Validation metrics
            'val_rmse': float(np.sqrt(mean_squared_error(y_val, val_pred))),
            'val_mae': float(mean_absolute_error(y_val, val_pred)),
            'val_r2': float(r2_score(y_val, val_pred)),
            'val_mape': float(np.mean(np.abs((y_val - val_pred) / y_val)) * 100)
        }
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=5,
            scoring='neg_root_mean_squared_error'
        )
        
        metrics.update({
            'cv_rmse_mean': float(-cv_scores.mean()),
            'cv_rmse_std': float(cv_scores.std())
        })
        
        return metrics

    def _calculate_feature_importance(self, X: pd.DataFrame) -> None:
        """Calculate and store feature importance scores."""
        if not hasattr(self.model, 'feature_importances_'):
            return
            
        importance = self.model.feature_importances_
        
        # Store individual feature importance
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
        Make capacity predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Series of predicted capacity values
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
            
            # Ensure predictions are non-negative
            predictions = np.maximum(predictions, 0)
            
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