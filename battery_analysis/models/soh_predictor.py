# battery_ml/models/soh_predictor.py
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .base_model import BaseModel
from ..utils.config import Config
from ..utils.logger import setup_logger

# Initialize logger for this module
logger = setup_logger(__name__)

class CustomXGBRegressor(xgb.XGBRegressor, BaseEstimator, RegressorMixin):
    """Custom XGBoost regressor that properly implements sklearn tags."""
    
    def __sklearn_tags__(self):
        """Implement the sklearn tags interface."""
        return {
            'allow_nan': True,
            'requires_fit': True,
            'requires_y': True,
            'requires_positive_y': False,
            '_skip_test': True,
            'preserves_dtype': [np.float64],
            'X_types': ['2darray', 'sparse'],
            'poor_score': False,
            'multioutput': False,
            'binary_only': False,
            'multilabel': False,
            '_xfail_checks': False
        }

class SOHPredictor(BaseModel):
    def __init__(self, model_type: str = 'xgboost', config: Config = None):
        super().__init__()
        self.model_type = model_type
        self.config = config or Config()
        self.model = None
        self.feature_importance = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type."""
        if self.model_type == 'xgboost':
            self.model = CustomXGBRegressor(**self.config.SOH_MODEL_PARAMS['xgboost'])
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(**self.config.SOH_MODEL_PARAMS['random_forest'])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        # Convert data types
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        # Handle any missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series, validation_size: float = 0.2) -> Dict:
        """
        Train the SOH prediction model with validation and early stopping.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target values
            validation_size (float): Size of validation set
            
        Returns:
            Dict: Training metrics
        """
        logger.info(f"Training {self.model_type} model for SOH prediction...")
        
        try:
            # Prepare data
            X, y = self._prepare_data(X, y)
            
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=validation_size,
                random_state=self.config.RANDOM_STATE
            )
            
            # Train the model with early stopping
            if self.model_type == 'xgboost':
                eval_set = [(X_train, y_train), (X_val, y_val)]
                early_stopping_rounds = self.config.SOH_MODEL_PARAMS['xgboost'].get('early_stopping_rounds', 50)
                
                fit_params = {
                    'eval_set': eval_set,
                    'verbose': True,
                    'callbacks': [xgb.callback.EarlyStopping(
                        rounds=early_stopping_rounds,
                        save_best=True
                    )]
                }
                self.model.fit(X_train, y_train, **fit_params)
                
                # Store feature importance
                self.feature_importance = pd.Series(
                    self.model.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False)
            else:
                self.model.fit(X_train, y_train)
            
            # Make predictions on training and validation sets
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            # Calculate metrics
            metrics = {
                'train': {
                    'mse': float(mean_squared_error(y_train, train_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
                    'mae': float(mean_absolute_error(y_train, train_pred)),
                    'r2': float(r2_score(y_train, train_pred))
                },
                'validation': {
                    'mse': float(mean_squared_error(y_val, val_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_val, val_pred))),
                    'mae': float(mean_absolute_error(y_val, val_pred)),
                    'r2': float(r2_score(y_val, val_pred))
                }
            }
            
            # Perform cross-validation on full dataset
            cv_scores = cross_val_score(self.model, X, y, cv=self.config.CV_FOLDS)
            metrics['cross_validation'] = {
                'mean_score': float(cv_scores.mean()),
                'std_score': float(cv_scores.std())
            }
            
            logger.info("Model training completed successfully.")
            logger.info(f"Training metrics: {metrics}")
            
            # Log feature importance if available
            if self.feature_importance is not None:
                logger.info("Top 10 important features:")
                logger.info(self.feature_importance.head(10).to_string())
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        
        try:
            # Prepare data
            X = X.astype(np.float32)
            X = X.fillna(X.mean())
            
            predictions = self.model.predict(X)
            return pd.Series(predictions, index=X.index)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def get_feature_importance(self, top_n: int = None) -> pd.Series:
        """
        Get feature importance ranking.
        
        Args:
            top_n (int, optional): Number of top features to return
            
        Returns:
            pd.Series: Feature importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Model needs to be trained first")
        
        if top_n is not None:
            return self.feature_importance.head(top_n)
        return self.feature_importance