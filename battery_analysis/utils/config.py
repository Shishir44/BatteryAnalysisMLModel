from pathlib import Path
from typing import Dict, Any
import logging

class Config:
    """Enhanced configuration class for battery analysis system."""
    
    def __init__(self):
        # Project Directory Structure
        self.PROJECT_ROOT = Path(__file__).parent.parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.MODELS_DIR = self.DATA_DIR / "models"
        self.RESULTS_DIR = self.PROJECT_ROOT / "results"
        self.VISUALIZATION_DIR = self.RESULTS_DIR / "visualizations"
        self.LOGS_DIR = self.PROJECT_ROOT / "logs"
        
        # Logging Configuration
        self.LOGGING = {
            'level': logging.INFO,
            'file_logging': True,
            'console_logging': True,
            'detailed_format': False,
            'log_rotation': {
                'max_bytes': 10 * 1024 * 1024,  # 10 MB
                'backup_count': 5
            },
            'separate_error_log': True,
            'log_format': {
                'basic': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'detailed': '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
            }
        }
        
        # Data Processing Configuration
        self.DATA_PROCESSING = {
            'validation': {
                'missing_values_threshold': 0.1,
                'outlier_threshold': 3.0,
                'correlation_threshold': 0.95
            },
            'preprocessing': {
                'voltage_bounds': (2.5, 4.2),   # Voltage range in V
                'current_bounds': (-20, 20),    # Current range in A
                'temperature_bounds': (15, 60),  # Temperature range in °C
                'capacity_bounds': (0, None),    # Capacity range
                'smoothing_window': 5,
                'min_measurements': 10,
                'scaling_method': 'robust'
            },
            'feature_engineering': {
                'window_sizes': [3, 5, 10],
                'voltage_ranges': [(2.5, 3.0), (3.0, 3.5), (3.5, 4.2)],
                'temp_ranges': [(15, 25), (25, 35), (35, 45), (45, 60)],
                'statistical_features': [
                    'mean', 'std', 'min', 'max',
                    'skew', 'kurtosis', 'q25', 'q75'
                ]
            }
        }
        
        # Battery-Specific Parameters
        self.BATTERY = {
            'chemistry': 'Li-ion',
            'nominal_capacity': 2.5,  # Ah
            'nominal_voltage': 3.7,   # V
            'cycle_definition': {
                'discharge_threshold': 2.5,  # V
                'charge_threshold': 4.2,     # V
                'min_cycle_time': 600,       # seconds
                'max_cycle_time': 7200       # seconds
            },
            'health_thresholds': {
                'soh_warning': 80,  # %
                'soh_critical': 60, # %
                'temperature_warning': 45,  # °C
                'temperature_critical': 55  # °C
            }
        }
        
        # Model Configuration
        self.MODELS = {
            # SOH Prediction Model
            'soh_predictor': {
                'xgboost': {
                    # Tree Parameters
                    'n_estimators': 500,
                    'max_depth': 8,
                    'min_child_weight': 3,
                    'gamma': 0.1,
                    
                    # Learning Parameters
                    'learning_rate': 0.03,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'colsample_bylevel': 0.8,
                    
                    # Regularization
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    
                    # Training Control
                    'early_stopping_rounds': 50,
                    'eval_metric': ['rmse', 'mae'],
                    
                    # Performance
                    'tree_method': 'hist',
                    'n_jobs': -1,
                    'random_state': 42
                },
                'training': {
                    'train_test_split': 0.2,
                    'validation_split': 0.1,
                    'cv_folds': 5,
                    'early_stopping_patience': 20
                }
            },
            
            # Capacity Prediction Model
            'capacity_predictor': {
                'gbm': {
                    'n_estimators': 300,
                    'learning_rate': 0.03,
                    'max_depth': 8,
                    'subsample': 0.8,
                    'min_samples_split': 5,
                    'min_samples_leaf': 4,
                    'validation_fraction': 0.1,
                    'n_iter_no_change': 20,
                    'tol': 1e-4,
                    'random_state': 42
                }
            },
            
            # Anomaly Detection Model
            'anomaly_detector': {
                'isolation_forest': {
                    'n_estimators': 200,
                    'contamination': 0.05,
                    'max_samples': 'auto',
                    'random_state': 42,
                    'n_jobs': -1,
                    'bootstrap': True
                },
                'thresholds': {
                    'voltage_anomaly': 0.1,
                    'current_anomaly': 0.1,
                    'temperature_anomaly': 0.1
                }
            }
        }
        
        # Feature Groups for Analysis
        self.FEATURE_GROUPS = {
            'voltage_features': [
                'voltage_mean', 'voltage_std', 'voltage_max', 'voltage_min',
                'voltage_range', 'voltage_skew', 'voltage_kurtosis',
                'voltage_efficiency', 'voltage_stability'
            ],
            'current_features': [
                'current_mean', 'current_std', 'current_integral',
                'terminal_current_mean', 'terminal_current_std',
                'terminal_current_min', 'terminal_current_max'
            ],
            'temperature_features': [
                'temp_mean', 'temp_max', 'temp_rise', 'temp_integral',
                'temp_std', 'temp_stress'
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
        
        # Visualization Configuration
        self.VISUALIZATION = {
            'style': 'seaborn',
            'figure_size': (12, 8),
            'dpi': 100,
            'color_palette': 'viridis',
            'save_format': 'png',
            'plot_types': {
                'capacity_fade': {
                    'size': (14, 7),
                    'style': 'scatter+line'
                },
                'voltage_curves': {
                    'size': (12, 6),
                    'style': 'multi_line'
                },
                'temperature_analysis': {
                    'size': (15, 5),
                    'style': 'heatmap'
                }
            }
        }
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration values from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                if isinstance(current_value, dict) and isinstance(value, dict):
                    current_value.update(value)
                else:
                    setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return getattr(self, key, default)
    
    def ensure_directories(self) -> None:
        """Create all necessary directories."""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODELS_DIR,
            self.RESULTS_DIR,
            self.VISUALIZATION_DIR,
            self.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration settings."""
        validation_results = {
            'warnings': [],
            'errors': []
        }
        
        # Validate directory permissions
        for directory in [self.DATA_DIR, self.LOGS_DIR]:
            if not directory.parent.exists():
                validation_results['errors'].append(
                    f"Parent directory does not exist: {directory.parent}"
                )
        
        # Validate model parameters
        for model_name, params in self.MODELS.items():
            if 'random_state' not in str(params):
                validation_results['warnings'].append(
                    f"Random state not set for {model_name}"
                )
        
        # Validate battery parameters
        if self.BATTERY['cycle_definition']['min_cycle_time'] >= \
           self.BATTERY['cycle_definition']['max_cycle_time']:
            validation_results['errors'].append(
                "Invalid cycle time definition: min_cycle_time >= max_cycle_time"
            )
        
        return validation_results