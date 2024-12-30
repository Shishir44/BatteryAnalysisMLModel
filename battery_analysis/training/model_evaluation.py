# battery_ml/training/model_evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    confusion_matrix, 
    precision_recall_curve,
    roc_curve,
    auc
)
from sklearn.model_selection import (
    cross_val_score, 
    learning_curve,
    validation_curve
)
from typing import Dict, List, Tuple, Optional, Union
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelEvaluator:
    """
    A comprehensive model evaluation class for battery ML models.
    Handles both regression and anomaly detection models.
    """
    
    def __init__(self, output_dir: str = 'evaluation_results'):
        """
        Initialize ModelEvaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Set up matplotlib and seaborn plotting styles."""
        plt.style.use('seaborn')
        sns.set_palette('husl')
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['font.size'] = 12
    
    def evaluate_regression_model(
        self, 
        model: object,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
    ) -> Dict:
        """
        Comprehensive evaluation of a regression model.
        
        Args:
            model: Trained regression model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        evaluation_results = {}
        
        # Basic predictions and metrics
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate training metrics
        evaluation_results['training_metrics'] = self._calculate_regression_metrics(
            y_train, train_pred, prefix='train'
        )
        
        # Calculate test metrics
        evaluation_results['test_metrics'] = self._calculate_regression_metrics(
            y_test, test_pred, prefix='test'
        )
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        evaluation_results['cross_validation'] = {
            'mean_cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        # Feature importance analysis
        if hasattr(model, 'feature_importances_'):
            evaluation_results['feature_importance'] = self._analyze_feature_importance(
                model, X_train.columns
            )
        
        # Error analysis
        evaluation_results['error_analysis'] = self._analyze_prediction_errors(
            y_test, test_pred, X_test
        )
        
        # Generate plots
        self._generate_regression_plots(
            y_test, test_pred, model, X_train, y_train, model_name
        )
        
        return evaluation_results
    
    def evaluate_anomaly_detector(
        self,
        model: object,
        X: pd.DataFrame,
        known_anomalies: Optional[pd.Series] = None
    ) -> Dict:
        """
        Evaluate anomaly detection model.
        
        Args:
            model: Trained anomaly detection model
            X: Input features
            known_anomalies: Optional ground truth anomaly labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions = model.predict(X)
        anomaly_scores = model.score_samples(X) if hasattr(model, 'score_samples') else None
        
        evaluation_results = {
            'anomaly_stats': {
                'num_anomalies': sum(predictions == -1),
                'anomaly_ratio': sum(predictions == -1) / len(predictions)
            }
        }
        
        if known_anomalies is not None:
            evaluation_results['validation'] = self._validate_anomaly_detection(
                known_anomalies, predictions, anomaly_scores
            )
        
        # Generate anomaly detection plots
        self._generate_anomaly_plots(X, predictions, anomaly_scores)
        
        return evaluation_results
    
    def _calculate_regression_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        prefix: str = ''
    ) -> Dict:
        """Calculate regression metrics."""
        return {
            f'{prefix}_mse': mean_squared_error(y_true, y_pred),
            f'{prefix}_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            f'{prefix}_mae': mean_absolute_error(y_true, y_pred),
            f'{prefix}_r2': r2_score(y_true, y_pred),
            f'{prefix}_mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def _analyze_feature_importance(
        self,
        model: object,
        feature_names: List[str]
    ) -> Dict:
        """Analyze feature importance."""
        importance = model.feature_importances_
        feature_importance = pd.Series(importance, index=feature_names)
        return {
            'feature_importance': feature_importance.to_dict(),
            'top_features': feature_importance.nlargest(5).to_dict()
        }
    
    def _analyze_prediction_errors(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        X: pd.DataFrame
    ) -> Dict:
        """Detailed analysis of prediction errors."""
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        
        analysis = {
            'error_stats': {
                'mean_error': errors.mean(),
                'error_std': errors.std(),
                'max_error': abs_errors.max(),
                'median_error': np.median(abs_errors)
            },
            'error_distribution': {
                'within_5%': np.mean(abs_errors <= 0.05 * np.abs(y_true)),
                'within_10%': np.mean(abs_errors <= 0.10 * np.abs(y_true)),
                'within_20%': np.mean(abs_errors <= 0.20 * np.abs(y_true))
            }
        }
        
        # Add error correlations with features
        error_correlations = {}
        for column in X.columns:
            correlation = np.corrcoef(X[column], errors)[0, 1]
            error_correlations[column] = correlation
        analysis['error_correlations'] = error_correlations
        
        return analysis
    
    def _generate_regression_plots(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        model: object,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str
    ):
        """Generate all regression-related plots."""
        # Actual vs Predicted
        plt.figure(figsize=(10, 10))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name}: Actual vs Predicted')
        plt.savefig(f'{self.output_dir}/{model_name}_actual_vs_predicted.png')
        plt.close()
        
        # Residual Plot
        plt.figure(figsize=(10, 6))
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{model_name}: Residual Plot')
        plt.savefig(f'{self.output_dir}/{model_name}_residuals.png')
        plt.close()
        
        # Feature Importance Plot (if available)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 6))
            importance = pd.Series(model.feature_importances_, index=X_train.columns)
            importance.sort_values().plot(kind='barh')
            plt.title(f'{model_name}: Feature Importance')
            plt.savefig(f'{self.output_dir}/{model_name}_feature_importance.png')
            plt.close()
    
    def _generate_anomaly_plots(
        self,
        X: pd.DataFrame,
        predictions: np.ndarray,
        scores: Optional[np.ndarray]
    ):
        """Generate anomaly detection plots."""
        # 2D visualization of anomalies
        if X.shape[1] >= 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=predictions, cmap='viridis')
            plt.title('Anomaly Detection Results')
            plt.savefig(f'{self.output_dir}/anomaly_detection_2d.png')
            plt.close()
        
        # Score distribution if available
        if scores is not None:
            plt.figure(figsize=(10, 6))
            plt.hist(scores, bins=50)
            plt.title('Anomaly Score Distribution')
            plt.savefig(f'{self.output_dir}/anomaly_scores_distribution.png')
            plt.close()
    
    def _validate_anomaly_detection(
        self,
        true_labels: pd.Series,
        predictions: np.ndarray,
        scores: Optional[np.ndarray]
    ) -> Dict:
        """Validate anomaly detection results against known labels."""
        # Convert predictions from [-1, 1] to [0, 1]
        pred_labels = (predictions == -1).astype(int)
        
        # Calculate basic metrics
        cm = confusion_matrix(true_labels, pred_labels)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
        }
        
        # Calculate ROC curve if scores are available
        if scores is not None:
            fpr, tpr, _ = roc_curve(true_labels, -scores)
            metrics['roc_auc'] = auc(fpr, tpr)
        
        return metrics

    def generate_report(self, evaluation_results: Dict, model_name: str) -> str:
        """Generate a formatted evaluation report."""
        report = [f"\n=== {model_name} Evaluation Report ===\n"]
        
        for section, metrics in evaluation_results.items():
            report.append(f"\n{section.upper()}:")
            if isinstance(metrics, dict):
                for name, value in metrics.items():
                    if isinstance(value, float):
                        report.append(f"  {name}: {value:.4f}")
                    else:
                        report.append(f"  {name}: {value}")
            else:
                report.append(f"  {metrics}")
        
        return "\n".join(report)

# Example usage
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from ..models import SOHPredictor, AnomalyDetector
    
    # Load data
    df = pd.read_csv('data/raw/merged_data.csv')
    
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir='evaluation_results')
    
    # Evaluate SOH predictor
    soh_model = SOHPredictor()
    X = df.drop(['SOH'], axis=1)
    y = df['SOH']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    soh_results = evaluator.evaluate_regression_model(
        soh_model, X_train, y_train, X_test, y_test, "SOH Predictor"
    )
    print(evaluator.generate_report(soh_results, "SOH Predictor"))
    
    # Evaluate anomaly detector
    anomaly_model = AnomalyDetector()
    anomaly_results = evaluator.evaluate_anomaly_detector(anomaly_model, X)
    print(evaluator.generate_report(anomaly_results, "Anomaly Detector"))