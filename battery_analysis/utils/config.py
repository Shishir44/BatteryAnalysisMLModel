
from pathlib import Path
from typing import Dict, Any

class Config:
    def __init__(self):
        # Project paths (keeping your existing paths)
        self.PROJECT_ROOT = Path(__file__).parent.parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.MODELS_DIR = self.DATA_DIR / "models"
        self.RESULTS_DIR = self.PROJECT_ROOT / "results"
        self.VISUALIZATION_DIR = self.RESULTS_DIR / "visualizations"
        
        # Model parameters
        self.SOH_MODEL_PARAMS = {
            'xgboost': {
                # Tree Parameters
                'n_estimators': 500,          # Increased for better accuracy
                'max_depth': 8,               # Slightly increased but controlled to prevent overfitting
                'min_child_weight': 3,        # Helps prevent overfitting
                'gamma': 0.1,                 # Minimum loss reduction for partition
                
                # Learning Parameters
                'learning_rate': 0.03,        # Reduced for better generalization
                'subsample': 0.8,             # Prevents overfitting
                'colsample_bytree': 0.8,      # Feature sampling per tree
                'colsample_bylevel': 0.8,     # Feature sampling per level
                
                # Regularization Parameters
                'reg_alpha': 0.1,             # L1 regularization
                'reg_lambda': 1.0,            # L2 regularization
                
                # Performance Parameters
                'tree_method': 'hist',        # Faster histogram-based algorithm
                'random_state': 42,           # For reproducibility
                
                # Early Stopping Parameters
                'early_stopping_rounds': 50,   # Stop if no improvement
                'eval_metric': ['rmse', 'mae'],# Multiple evaluation metrics
                
                # Additional Parameters
                'n_jobs': -1,                 # Use all CPU cores
                'verbosity': 1,               # Logging level
                'booster': 'gbtree',          # Using tree-based model
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 12,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        }
        
        # Updated Capacity Model Parameters
        self.CAPACITY_MODEL_PARAMS = {
            'n_estimators': 300,
            'learning_rate': 0.03,
            'max_depth': 8,
            'subsample': 0.8,
            'random_state': 42
        }
        
        # Updated Anomaly Detector Parameters
        self.ANOMALY_DETECTOR_PARAMS = {
            'contamination': 0.05,    # Reduced from 0.1 to be more selective
            'n_estimators': 200,
            'max_samples': 'auto',
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Feature engineering parameters (keeping your existing parameters)
        self.FEATURE_ENGINEERING = {
            'window_sizes': [3, 5, 10],
            'polynomial_degree': 2,
            'temp_ranges': [(0, 25), (25, 40), (40, 60)],
            'statistical_features': [
                'mean', 'std', 'min', 'max', 'skew'
            ]
        }
        
        # Training parameters
        self.TRAIN_TEST_SPLIT = 0.2
        self.RANDOM_STATE = 42
        self.CV_FOLDS = 5
        
        # Preprocessing parameters
        self.PREPROCESSING = {
            'outlier_threshold': 3,
            'smoothing_window': 5,
            'min_measurements_per_cycle': 10
        }

    # Keeping your existing methods
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration values from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values to update
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                if isinstance(current_value, dict) and isinstance(value, dict):
                    current_value.update(value)
                else:
                    setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value
        """
        return getattr(self, key, default)
    
    def ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODELS_DIR,
            self.RESULTS_DIR,
            self.VISUALIZATION_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)