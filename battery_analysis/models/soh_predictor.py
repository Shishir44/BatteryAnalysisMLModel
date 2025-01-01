# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.base import BaseEstimator, RegressorMixin
# import xgboost as xgb
# import numpy as np
# import torch
# from xgboost.callback import EarlyStopping
# import pandas as pd
# from typing import Dict, Tuple
# from .base_model import BaseModel
# from ..utils.config import Config
# from ..utils.logger import setup_logger

# # Initialize logger for this module
# logger = setup_logger(__name__)

# class CustomXGBRegressor(BaseEstimator, RegressorMixin):
#     """
#     Custom XGBoost regressor with modern scikit-learn compatibility and GPU support.
#     """
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs
#         self.model = xgb.XGBRegressor(**kwargs)
    
#     @property
#     def __sklearn_tags__(self):
#         """Define estimator tags using the modern scikit-learn approach."""
#         return {
#             'allow_nan': True,
#             'requires_y': True,
#             'requires_positive_y': False,
#             '_skip_test': True,
#             'preserves_dtype': [np.float64],
#             'X_types': ['2darray', 'sparse'],
#             'poor_score': False,
#             'multioutput': False
#         }
    
#     def fit(self, X, y, **fit_params):
#         self.model.fit(X, y, **fit_params)
#         return self
    
#     def predict(self, X):
#         return self.model.predict(X)
    
#     def set_params(self, **params):
#         self.kwargs.update(params)
#         self.model.set_params(**params)
#         return self
    
#     def get_params(self, deep=True):
#         return self.kwargs
    
#     @property
#     def feature_importances_(self):
#         return self.model.feature_importances_

# class SOHPredictor(BaseModel):
#     """
#     State of Health (SOH) predictor for battery analysis using machine learning.
#     Supports both GPU and CPU execution with automatic hardware detection.
#     """
    
#     def __init__(self, model_type: str = 'xgboost', config: Config = None):
#         """
#         Initialize the SOH predictor.
        
#         Args:
#             model_type (str): Type of model to use ('xgboost' by default)
#             config (Config): Configuration object containing model parameters
#         """
#         super().__init__()
#         self.model_type = model_type
#         self.config = config or Config()
#         self.gpu_status = self._check_gpu_status()
#         self.feature_importance = None
#         self._initialize_model()

#     def _check_gpu_status(self) -> Dict[str, bool | str]:
#         """
#         Perform comprehensive check of GPU status including both PyTorch and XGBoost capabilities.
        
#         Returns:
#             Dict containing detailed GPU status information
#         """
#         status = {
#             'gpu_available': False,
#             'cuda_available': False,
#             'gpu_name': None,
#             'cuda_version': None,
#             'gpu_memory_total': None,
#             'xgboost_gpu_support': False
#         }
        
#         try:
#             if torch.cuda.is_available():
#                 status['cuda_available'] = True
#                 status['gpu_available'] = True
#                 status['gpu_name'] = torch.cuda.get_device_name(0)
#                 status['cuda_version'] = torch.version.cuda
                
#                 gpu_properties = torch.cuda.get_device_properties(0)
#                 status['gpu_memory_total'] = f"{gpu_properties.total_memory / 1e9:.2f} GB"
                
#                 try:
#                     test_tensor = torch.cuda.FloatTensor([1.0])
#                     status['cuda_working'] = True
#                 except Exception as e:
#                     logger.warning(f"CUDA test failed: {str(e)}")
#                     status['cuda_working'] = False
            
#             # Verify XGBoost GPU support
#             try:
#                 X = np.random.rand(10, 2)
#                 y = np.random.rand(10)
#                 test_model = xgb.XGBRegressor(
#                     tree_method='gpu_hist',
#                     n_estimators=2,
#                     max_depth=2
#                 )
#                 test_model.fit(X, y)
#                 status['xgboost_gpu_support'] = True
#             except Exception as e:
#                 logger.warning(f"XGBoost GPU test failed: {str(e)}")
#                 status['xgboost_gpu_support'] = False
                
#         except Exception as e:
#             logger.error(f"Error checking GPU status: {str(e)}")
        
#         logger.info("GPU Status Check Results:")
#         for key, value in status.items():
#             logger.info(f"{key}: {value}")
            
#         return status
    
#     def _initialize_model(self):
#         """Initialize the ML model based on hardware availability and type."""
#         if self.model_type == 'xgboost':
#             try:
#                 # Get XGBoost parameters with safe dictionary access
#                 default_params = {
#                     'n_estimators': 100,
#                     'learning_rate': 0.1,
#                     'max_depth': 6,
#                     'min_child_weight': 1,
#                     'subsample': 0.8,
#                     'colsample_bytree': 0.8,
#                     'objective': 'reg:squarederror',
#                     'early_stopping_rounds': 10,
#                     'eval_metric': 'rmse'
#                 }
                
#                 # Safely get config parameters if they exist
#                 base_params = (self.config.SOH_MODEL_PARAMS['xgboost'] 
#                             if isinstance(self.config.SOH_MODEL_PARAMS, dict) 
#                             and 'xgboost' in self.config.SOH_MODEL_PARAMS 
#                             else {})
                
#                 if not isinstance(base_params, dict):
#                     logger.warning("Invalid SOH_MODEL_PARAMS format. Using defaults.")
#                     base_params = {}
                
#                 # Merge parameters
#                 model_params = {**default_params, **base_params}
                
#                 # Add hardware-specific parameters
#                 if (self.gpu_status['gpu_available'] and 
#                     self.gpu_status['xgboost_gpu_support']):
#                     model_params.update({
#                         'tree_method': 'gpu_hist',
#                         'gpu_id': 0,
#                         'predictor': 'gpu_predictor'
#                     })
#                     logger.info("Initializing XGBoost with GPU support")
#                 else:
#                     model_params.update({
#                         'tree_method': 'hist',
#                         'predictor': 'cpu_predictor'
#                     })
#                     logger.info("Initializing XGBoost with CPU support")

#                 self.model = CustomXGBRegressor(**model_params)
#                 logger.info(f"XGBoost version: {xgb.__version__}")
#                 logger.info(f"Model parameters: {model_params}")
                
#             except Exception as e:
#                 logger.error(f"Error initializing XGBoost model: {str(e)}")
#                 raise DetailedModelError(f"Model initialization failed: {str(e)}")
#         else:
#             raise ValueError(f"Unsupported model type: {self.model_type}")
            

#     def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
#         """
#         Prepare data for training by handling missing values and type conversion.
        
#         Args:
#             X (pd.DataFrame): Feature matrix
#             y (pd.Series): Target values
            
#         Returns:
#             Tuple[pd.DataFrame, pd.Series]: Processed features and target
#         """
#         X = X.astype(np.float32)
#         y = y.astype(np.float32)
        
#         X = X.fillna(X.mean())
#         y = y.fillna(y.mean())
        
#         return X, y

#     def _calculate_metrics(self, X_train, X_val, y_train, y_val) -> Dict:
#         """
#         Calculate comprehensive training and validation metrics.
        
#         Args:
#             X_train, X_val: Training and validation feature matrices
#             y_train, y_val: Training and validation target values
            
#         Returns:
#             Dict: Dictionary containing various performance metrics
#         """
#         train_pred = self.model.predict(X_train)
#         val_pred = self.model.predict(X_val)
        
#         metrics = {
#             'train': {
#                 'mse': float(mean_squared_error(y_train, train_pred)),
#                 'rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
#                 'mae': float(mean_absolute_error(y_train, train_pred)),
#                 'r2': float(r2_score(y_train, train_pred))
#             },
#             'validation': {
#                 'mse': float(mean_squared_error(y_val, val_pred)),
#                 'rmse': float(np.sqrt(mean_squared_error(y_val, val_pred))),
#                 'mae': float(mean_absolute_error(y_val, val_pred)),
#                 'r2': float(r2_score(y_val, val_pred))
#             }
#         }
        
#         cv_scores = cross_val_score(
#             self.model, 
#             pd.concat([X_train, X_val]), 
#             pd.concat([y_train, y_val]),
#             cv=self.config.CV_FOLDS,
#             scoring='r2'
#         )
        
#         metrics['cross_validation'] = {
#             'mean_r2': float(cv_scores.mean()),
#             'std_r2': float(cv_scores.std())
#         }
        
#         return metrics

#     def train(self, X: pd.DataFrame, y: pd.Series, validation_size: float = 0.2) -> Dict:
#         """
#         Train the SOH prediction model with comprehensive monitoring and validation.
        
#         This implementation properly handles early stopping in the scikit-learn API context.
#         Early stopping is configured through model parameters rather than fit parameters,
#         which is the correct approach for the scikit-learn API wrapper.
        
#         Args:
#             X (pd.DataFrame): Feature matrix containing battery parameters
#             y (pd.Series): Target values (State of Health measurements)
#             validation_size (float): Proportion of data to use for validation
            
#         Returns:
#             Dict: Dictionary containing training metrics and cross-validation results
#         """
#         logger.info(f"Training {self.model_type} model for SOH prediction...")
        
#         try:
#             # Prepare the training and validation data
#             X, y = self._prepare_data(X, y)
#             X_train, X_val, y_train, y_val = train_test_split(
#                 X, y, test_size=validation_size, random_state=self.config.RANDOM_STATE
#             )
            
#             # Monitor GPU memory if available
#             if self.gpu_status.get('gpu_available', False):
#                 initial_memory = torch.cuda.memory_allocated()
#                 logger.info(f"Initial GPU memory allocated: {initial_memory / 1e6:.2f} MB")
            
#             # Configure early stopping through model parameters
            
#             early_stopping_params = {
#                 'n_estimators': 1000,  # Set a high number as maximum iterations
#                 'early_stopping_rounds': 10,  # Stop if no improvement for 10 rounds
#                 'eval_metric': 'rmse'  # Metric to monitor for early stopping
#             }
            
#             # Update model parameters with early stopping configuration
#             self.model.set_params(**early_stopping_params)
            
#             # Train the model with validation set for monitoring
#             # Note: early_stopping_rounds is not passed to fit() directly
#             self.model.fit(
#                 X_train, 
#                 y_train,
#                 eval_set=[(X_val, y_val)],
#                 verbose=True  # Enable progress reporting
#             )
            
#             # Clean up GPU memory if used
#             if self.gpu_status.get('gpu_available', False):
#                 final_memory = torch.cuda.memory_allocated()
#                 memory_used = (final_memory - initial_memory) / 1e6
#                 logger.info(f"GPU Memory used for training: {memory_used:.2f} MB")
#                 torch.cuda.empty_cache()
            
#             # Store feature importance if available
#             if hasattr(self.model, 'feature_importances_'):
#                 self.feature_importance = pd.Series(
#                     self.model.feature_importances_,
#                     index=X.columns
#                 ).sort_values(ascending=False)
            
#             # Calculate and log performance metrics
#             metrics = self._calculate_metrics(X_train, X_val, y_train, y_val)
            
#             # Log training results
#             logger.info("Model training completed successfully")
#             logger.info(f"Validation RMSE: {metrics['validation']['rmse']:.4f}")
#             logger.info(f"Validation R²: {metrics['validation']['r2']:.4f}")
            
#             if self.feature_importance is not None:
#                 logger.info("\nTop 5 important features:")
#                 for feat, imp in self.feature_importance.head().items():
#                     logger.info(f"{feat}: {imp:.4f}")
            
#             return metrics
            
#         except Exception as e:
#             logger.error(f"Error during model training: {str(e)}")
#             raise

#     def predict(self, X: pd.DataFrame) -> pd.Series:
#         """
#         Make predictions using the trained model.
        
#         Args:
#             X (pd.DataFrame): Feature matrix for prediction
            
#         Returns:
#             pd.Series: Predicted SOH values
#         """
#         if self.model is None:
#             raise ValueError("Model needs to be trained first")
        
#         try:
#             X = X.astype(np.float32)
#             X = X.fillna(X.mean())
            
#             predictions = self.model.predict(X)
#             return pd.Series(predictions, index=X.index)
#         except Exception as e:
#             logger.error(f"Error during prediction: {str(e)}")
#             raise

#     def get_feature_importance(self, top_n: int = None) -> pd.Series:
#         """
#         Get feature importance ranking.
        
#         Args:
#             top_n (int, optional): Number of top features to return
            
#         Returns:
#             pd.Series: Feature importance scores
#         """
#         if self.feature_importance is None:
#             raise ValueError("Model needs to be trained first")
        
#         if top_n is not None:
#             return self.feature_importance.head(top_n)
#         return self.feature_importance

# # Custom exception for more detailed error reporting
# class DetailedModelError(Exception):
#     """Custom exception class for model-related errors with detailed information."""
#     pass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import numpy as np
import torch
import pandas as pd
from typing import Dict, Tuple
from .base_model import BaseModel
from ..utils.config import Config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class CustomXGBRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs
        
    def fit(self, X, y, **fit_params):
        if self.model is None:
            self.model = xgb.XGBRegressor(**self.kwargs)
            
        # Ensure eval_set is provided for early stopping
        if 'early_stopping_rounds' in self.kwargs and 'eval_set' not in fit_params:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            fit_params['eval_set'] = [(X_val, y_val)]
            
        self.model.fit(X, y, **fit_params)
        return self
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)
    
    def set_params(self, **params):
        self.kwargs.update(params)
        if self.model is not None:
            self.model.set_params(**params)
        return self
    
    def get_params(self, deep=True):
        return self.kwargs
    
    @property
    def feature_importances_(self):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        return self.model.feature_importances_

class SOHPredictor(BaseModel):
    def __init__(self, model_type: str = 'xgboost', config: Config = None):
        super().__init__()
        self.model_type = model_type
        self.config = config or Config()
        self.gpu_status = self._check_gpu_status()
        self.feature_importance = None
        self._initialize_model()

    def _check_gpu_status(self) -> Dict[str, bool | str]:
        status = {
            'gpu_available': False,
            'xgboost_gpu_support': False
        }
        
        try:
            if torch.cuda.is_available():
                status['gpu_available'] = True
                
                try:
                    X = np.random.rand(10, 2)
                    y = np.random.rand(10)
                    test_model = xgb.XGBRegressor(tree_method='gpu_hist', n_estimators=2)
                    test_model.fit(X, y)
                    status['xgboost_gpu_support'] = True
                except:
                    status['xgboost_gpu_support'] = False
                    
        except Exception as e:
            logger.error(f"Error checking GPU status: {str(e)}")
        
        return status

    def _initialize_model(self):
        if self.model_type == 'xgboost':
            try:
                default_params = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'min_child_weight': 1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'reg:squarederror',
                    'early_stopping_rounds': 10,
                    'eval_metric': 'rmse'
                }
                
                config_params = getattr(self.config, 'SOH_MODEL_PARAMS', {}).get('xgboost', {})
                model_params = {**default_params, **(config_params if isinstance(config_params, dict) else {})}
                
                if self.gpu_status.get('xgboost_gpu_support'):
                    model_params.update({'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'})
                else:
                    model_params.update({'tree_method': 'hist'})

                self.model = CustomXGBRegressor(**model_params)
                logger.info(f"Model parameters: {model_params}")
                
            except Exception as e:
                logger.error(f"Error initializing XGBoost model: {str(e)}")
                raise DetailedModelError(f"Model initialization failed: {str(e)}")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        return X, y

    def _calculate_metrics(self, X_train, X_val, y_train, y_val) -> Dict:
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
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
        
        # Create a new instance for cross-validation without early stopping
        cv_params = {k: v for k, v in self.model.get_params().items() 
                    if k not in ['early_stopping_rounds', 'eval_metric']}
        cv_model = CustomXGBRegressor(**cv_params)
        
        cv_scores = cross_val_score(
            cv_model,
            pd.concat([X_train, X_val]),
            pd.concat([y_train, y_val]),
            cv=getattr(self.config, 'CV_FOLDS', 5),
            scoring='r2'
        )
        
        metrics['cross_validation'] = {
            'mean_r2': float(cv_scores.mean()),
            'std_r2': float(cv_scores.std())
        }
        
        return metrics

    def train(self, X: pd.DataFrame, y: pd.Series, validation_size: float = 0.2) -> Dict:
        logger.info(f"Training {self.model_type} model for SOH prediction...")
        
        try:
            X, y = self._prepare_data(X, y)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=validation_size,
                random_state=getattr(self.config, 'RANDOM_STATE', 42)
            )
            
            # Always provide validation dataset for early stopping
            eval_set = [(X_val, y_val)]
            
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=True
            )

            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.Series(
                    self.model.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False)
            
            metrics = self._calculate_metrics(X_train, X_val, y_train, y_val)
            logger.info(f"Training completed - Validation R²: {metrics['validation']['r2']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("Model needs to be trained first")
        
        try:
            X = X.astype(np.float32)
            X = X.fillna(X.mean())
            return pd.Series(self.model.predict(X), index=X.index)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def get_feature_importance(self, top_n: int = None) -> pd.Series:
        if self.feature_importance is None:
            raise ValueError("Model needs to be trained first")
        return self.feature_importance.head(top_n) if top_n else self.feature_importance

class DetailedModelError(Exception):
    pass