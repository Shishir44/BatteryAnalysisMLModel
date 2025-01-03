import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json

class BatteryLogger:
    """Enhanced logger for battery analysis system with specialized formatting and handlers."""
    
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
    
    def __init__(
        self,
        name: str,
        log_dir: Optional[Union[str, Path]] = None,
        level: int = logging.INFO,
        log_to_file: bool = True,
        log_to_console: bool = True,
        detailed_format: bool = False,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5
    ):
        """
        Initialize enhanced logger with customizable configuration.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            detailed_format: Use detailed logging format
            max_file_size: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Select format based on detail level
        log_format = self.DETAILED_FORMAT if detailed_format else self.DEFAULT_FORMAT
        formatter = logging.Formatter(log_format)
        
        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler if requested
        if log_to_file and log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Regular log file with rotation
            log_file = log_dir / f"{name}.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Error log file (errors only)
            error_log = log_dir / f"{name}_error.log"
            error_handler = RotatingFileHandler(
                error_log,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            self.logger.addHandler(error_handler)
    
    def log_model_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        level: int = logging.INFO
    ) -> None:
        """
        Log model-related events with structured data.
        
        Args:
            event_type: Type of model event
            details: Event details
            level: Logging level for this event
        """
        event_data = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        self.logger.log(level, f"Model Event: {json.dumps(event_data)}")
    
    def log_training_step(
        self,
        step: int,
        metrics: Dict[str, float],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log training step information.
        
        Args:
            step: Training step number
            metrics: Training metrics
            additional_info: Additional information to log
        """
        training_data = {
            'step': step,
            'metrics': metrics,
            'additional_info': additional_info or {}
        }
        self.logger.info(f"Training Step: {json.dumps(training_data)}")
    
    def log_prediction(
        self,
        prediction_type: str,
        prediction_details: Dict[str, Any],
        confidence: Optional[float] = None
    ) -> None:
        """
        Log model predictions with details.
        
        Args:
            prediction_type: Type of prediction
            prediction_details: Prediction details
            confidence: Prediction confidence score
        """
        prediction_data = {
            'type': prediction_type,
            'details': prediction_details,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(f"Prediction: {json.dumps(prediction_data)}")
    
    def log_validation_results(
        self,
        validation_metrics: Dict[str, float],
        validation_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log model validation results.
        
        Args:
            validation_metrics: Validation metrics
            validation_info: Additional validation information
        """
        validation_data = {
            'metrics': validation_metrics,
            'additional_info': validation_info or {},
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(f"Validation Results: {json.dumps(validation_data)}")
    
    def log_battery_state(
        self,
        cycle: int,
        state_data: Dict[str, Any],
        anomalies: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log battery state information.
        
        Args:
            cycle: Battery cycle number
            state_data: Battery state data
            anomalies: Detected anomalies
        """
        state_info = {
            'cycle': cycle,
            'state': state_data,
            'anomalies': anomalies,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(f"Battery State: {json.dumps(state_info)}")
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log error with context information.
        
        Args:
            error: Exception object
            context: Error context information
        """
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        self.logger.error(f"Error: {json.dumps(error_data)}")
    
    def log_warning(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log warning with additional details.
        
        Args:
            message: Warning message
            details: Warning details
        """
        warning_data = {
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        self.logger.warning(f"Warning: {json.dumps(warning_data)}")

def setup_logger(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    **kwargs
) -> BatteryLogger:
    """
    Setup and return a configured BatteryLogger instance.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured BatteryLogger instance
    """
    return BatteryLogger(name, log_dir, level, **kwargs)