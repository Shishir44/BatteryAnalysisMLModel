import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    confusion_matrix
)
from sklearn.model_selection import learning_curve, validation_curve
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation for battery health prediction models.
    Specialized for evaluating models trained on the comprehensive battery dataset.
    """
    
    def __init__(self, output_dir: str = 'evaluation_results'):
        """
        Initialize ModelEvaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_plotting_style()
        
        # Define feature groups for analysis
        self.feature_groups = {
            'voltage_features': [
                'voltage_mean', 'voltage_std', 'voltage_stability',
                'voltage_efficiency'
            ],
            'current_features': [
                'current_mean', 'current_std', 'current_integral'
            ],
            'temperature_features': [
                'temp_mean', 'temp_max', 'temp_stress'
            ],
            'power_features': [
                'power_efficiency', 'avg_power', 'max_power'
            ],
            'degradation_features': [
                'capacity_degradation', 'SOH_degradation',
                'capacity_degradation_rate'
            ]
        }
    
    def _setup_plotting_style(self):
        """Set up matplotlib and seaborn plotting styles."""
        plt.style.use('seaborn')
        sns.set_palette('deep')
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 12
    
    def evaluate_regression_model(
        self, 
        model: object,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        target_type: str = 'SOH'  # 'SOH' or 'capacity'
    ) -> Dict:
        """
        Comprehensive evaluation of regression models for battery health prediction.
        
        Args:
            model: Trained model object
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            target_type: Type of prediction target
            
        Returns:
            Dictionary containing evaluation metrics and analysis
        """
        logger.info(f"Evaluating {model_name} for {target_type} prediction...")
        
        evaluation_results = {
            'model_name': model_name,
            'target_type': target_type,
            'metrics': {},
            'feature_analysis': {},
            'error_analysis': {},
            'degradation_analysis': {}
        }
        
        # Calculate basic metrics
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        evaluation_results['metrics'] = self._calculate_metrics(
            y_train, train_pred, y_test, test_pred
        )
        
        # Feature importance analysis
        if hasattr(model, 'feature_importances_'):
            evaluation_results['feature_analysis'] = self._analyze_feature_importance(
                model, X_train.columns, self.feature_groups
            )
        
        # Error analysis
        evaluation_results['error_analysis'] = self._analyze_prediction_errors(
            y_test, test_pred, X_test
        )
        
        # Degradation analysis (specific to battery health)
        evaluation_results['degradation_analysis'] = self._analyze_degradation_patterns(
            X_test, y_test, test_pred, target_type
        )
        
        # Generate plots
        self._generate_evaluation_plots(
            y_test, test_pred, model, X_train, y_train,
            model_name, target_type
        )
        
        logger.info("Model evaluation completed successfully")
        return evaluation_results

    def _calculate_metrics(
        self,
        y_train: pd.Series,
        train_pred: np.ndarray,
        y_test: pd.Series,
        test_pred: np.ndarray
    ) -> Dict:
        """Calculate comprehensive regression metrics."""
        metrics = {
            'train': {
                'rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
                'mae': float(mean_absolute_error(y_train, train_pred)),
                'r2': float(r2_score(y_train, train_pred)),
                'mape': float(np.mean(np.abs((y_train - train_pred) / y_train)) * 100)
            },
            'test': {
                'rmse': float(np.sqrt(mean_squared_error(y_test, test_pred))),
                'mae': float(mean_absolute_error(y_test, test_pred)),
                'r2': float(r2_score(y_test, test_pred)),
                'mape': float(np.mean(np.abs((y_test - test_pred) / y_test)) * 100)
            }
        }
        
        # Calculate relative error statistics
        rel_errors = np.abs((y_test - test_pred) / y_test) * 100
        metrics['error_distribution'] = {
            'within_1percent': float(np.mean(rel_errors <= 1)),
            'within_5percent': float(np.mean(rel_errors <= 5)),
            'within_10percent': float(np.mean(rel_errors <= 10))
        }
        
        return metrics

    def _analyze_feature_importance(
        self,
        model: object,
        feature_names: List[str],
        feature_groups: Dict[str, List[str]]
    ) -> Dict:
        """Analyze feature importance by group and individually."""
        importance = model.feature_importances_
        
        # Individual feature importance
        feature_importance = pd.Series(
            importance,
            index=feature_names
        ).sort_values(ascending=False)
        
        # Group importance
        group_importance = {}
        for group_name, group_features in feature_groups.items():
            available_features = [f for f in group_features if f in feature_names]
            if available_features:
                group_indices = [
                    list(feature_names).index(f) for f in available_features
                ]
                group_importance[group_name] = float(
                    np.mean(importance[group_indices])
                )
        
        return {
            'feature_importance': feature_importance.to_dict(),
            'group_importance': group_importance,
            'top_features': feature_importance.head(10).to_dict()
        }

    def _analyze_prediction_errors(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        X: pd.DataFrame
    ) -> Dict:
        """Analyze prediction errors and their patterns."""
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        rel_errors = abs_errors / y_true * 100
        
        error_analysis = {
            'error_stats': {
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'max_error': float(np.max(abs_errors)),
                'median_error': float(np.median(abs_errors))
            },
            'relative_error_stats': {
                'mean_rel_error': float(np.mean(rel_errors)),
                'std_rel_error': float(np.std(rel_errors)),
                'max_rel_error': float(np.max(rel_errors))
            }
        }
        
        # Error correlations with features
        error_correlations = {}
        for column in X.columns:
            correlation = np.corrcoef(X[column], errors)[0, 1]
            if not np.isnan(correlation):
                error_correlations[column] = float(correlation)
        
        error_analysis['error_correlations'] = error_correlations
        
        return error_analysis

    def _analyze_degradation_patterns(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        target_type: str
    ) -> Dict:
        """Analyze degradation patterns in predictions."""
        degradation_analysis = {}
        
        if 'cycle' in X.columns:
            # Group by cycle
            cycle_data = pd.DataFrame({
                'cycle': X['cycle'],
                'true': y_true,
                'predicted': y_pred
            })
            
            cycle_analysis = cycle_data.groupby('cycle').agg({
                'true': ['mean', 'std'],
                'predicted': ['mean', 'std']
            })
            
            degradation_analysis['cycle_wise'] = {
                'mean_true_rate': float(np.diff(cycle_analysis['true']['mean']).mean()),
                'mean_pred_rate': float(np.diff(cycle_analysis['predicted']['mean']).mean())
            }
        
        # Analyze degradation rate accuracy
        if target_type == 'SOH' and 'SOH_degradation_rate' in X.columns:
            degradation_analysis['soh_analysis'] = {
                'degradation_correlation': float(np.corrcoef(
                    X['SOH_degradation_rate'],
                    y_true - y_pred
                )[0, 1])
            }
        
        return degradation_analysis

    def _generate_evaluation_plots(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model: object,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        target_type: str
    ):
        """Generate comprehensive evaluation plots."""
        # Actual vs Predicted
        plt.figure(figsize=(10, 10))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel(f'Actual {target_type}')
        plt.ylabel(f'Predicted {target_type}')
        plt.title(f'{model_name}: Actual vs Predicted {target_type}')
        plt.savefig(self.output_dir / f'{model_name}_actual_vs_predicted.png')
        plt.close()
        
        # Error Distribution
        plt.figure(figsize=(10, 6))
        errors = y_true - y_pred
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'{model_name}: Error Distribution')
        plt.savefig(self.output_dir / f'{model_name}_error_distribution.png')
        plt.close()
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 6))
            importance = pd.Series(
                model.feature_importances_,
                index=X_train.columns
            ).sort_values(ascending=True)
            importance.plot(kind='barh')
            plt.title(f'{model_name}: Feature Importance')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{model_name}_feature_importance.png')
            plt.close()
        
        # Learning Curves
        plt.figure(figsize=(10, 6))
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5, scoring='neg_root_mean_squared_error'
        )
        
        plt.plot(train_sizes, -train_scores.mean(axis=1), label='Training Error')
        plt.plot(train_sizes, -val_scores.mean(axis=1), label='Validation Error')
        plt.xlabel('Training Size')
        plt.ylabel('RMSE')
        plt.title(f'{model_name}: Learning Curves')
        plt.legend()
        plt.savefig(self.output_dir / f'{model_name}_learning_curves.png')
        plt.close()

    def generate_report(self, evaluation_results: Dict) -> str:
        """Generate a formatted evaluation report."""
        report = []
        
        # Model information
        report.append(f"=== {evaluation_results['model_name']} Evaluation Report ===")
        report.append(f"Target: {evaluation_results['target_type']}\n")
        
        # Metrics
        report.append("Performance Metrics:")
        for phase in ['train', 'test']:
            report.append(f"\n{phase.capitalize()} Metrics:")
            metrics = evaluation_results['metrics'][phase]
            for metric, value in metrics.items():
                report.append(f"  {metric}: {value:.4f}")
        
        # Error Distribution
        report.append("\nError Distribution:")
        for threshold, value in evaluation_results['metrics']['error_distribution'].items():
            report.append(f"  {threshold}: {value*100:.2f}%")
        
        # Feature Importance
        if 'feature_analysis' in evaluation_results:
            report.append("\nTop 5 Important Features:")
            top_features = dict(sorted(
                evaluation_results['feature_analysis']['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
            for feature, importance in top_features.items():
                report.append(f"  {feature}: {importance:.4f}")
        
        # Error Analysis
        report.append("\nError Analysis:")
        error_stats = evaluation_results['error_analysis']['error_stats']
        for metric, value in error_stats.items():
            report.append(f"  {metric}: {value:.4f}")
        
        return "\n".join(report)
    
    def evaluate_anomaly_detector(
        self,
        model: object,
        X: pd.DataFrame,
        known_anomalies: Optional[pd.Series] = None,
        threshold: float = 0.1
    ) -> Dict:
        """
        Evaluate anomaly detection model performance.
        
        Args:
            model: Trained anomaly detection model
            X: Input features
            known_anomalies: Optional ground truth anomaly labels
            threshold: Anomaly threshold
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating anomaly detection model...")
        
        # Get predictions and scores
        predictions = model.predict(X)
        scores = model.score_samples(X) if hasattr(model, 'score_samples') else None
        
        evaluation_results = {
            'anomaly_stats': self._calculate_anomaly_stats(predictions, scores),
            'feature_analysis': self._analyze_anomaly_features(X, predictions, scores)
        }
        
        # If ground truth is available
        if known_anomalies is not None:
            evaluation_results['validation'] = self._validate_anomalies(
                known_anomalies, predictions
            )
        
        # Generate anomaly detection plots
        self._generate_anomaly_plots(X, predictions, scores)
        
        return evaluation_results

    def _calculate_anomaly_stats(
        self,
        predictions: np.ndarray,
        scores: Optional[np.ndarray]
    ) -> Dict:
        """Calculate basic anomaly detection statistics."""
        stats = {
            'total_samples': len(predictions),
            'anomalies_detected': int(sum(predictions == -1)),
            'anomaly_ratio': float(np.mean(predictions == -1))
        }
        
        if scores is not None:
            stats.update({
                'score_mean': float(np.mean(scores)),
                'score_std': float(np.std(scores)),
                'score_min': float(np.min(scores)),
                'score_max': float(np.max(scores))
            })
        
        return stats

    def _analyze_anomaly_features(
        self,
        X: pd.DataFrame,
        predictions: np.ndarray,
        scores: Optional[np.ndarray]
    ) -> Dict:
        """Analyze feature patterns in anomalies."""
        analysis = {'feature_stats': {}}
        
        # Calculate statistics for each feature group
        for group_name, features in self.feature_groups.items():
            group_features = [f for f in features if f in X.columns]
            if not group_features:
                continue
                
            # Compare normal vs anomalous samples
            normal_samples = X[predictions == 1][group_features]
            anomaly_samples = X[predictions == -1][group_features]
            
            group_stats = {
                'normal_mean': normal_samples.mean().to_dict(),
                'normal_std': normal_samples.std().to_dict(),
                'anomaly_mean': anomaly_samples.mean().to_dict(),
                'anomaly_std': anomaly_samples.std().to_dict()
            }
            
            # Calculate feature importance scores based on separation
            feature_scores = {}
            for feature in group_features:
                separation_score = abs(
                    np.mean(normal_samples[feature]) - 
                    np.mean(anomaly_samples[feature])
                ) / (np.std(normal_samples[feature]) + 1e-10)
                feature_scores[feature] = float(separation_score)
            
            group_stats['feature_scores'] = feature_scores
            analysis['feature_stats'][group_name] = group_stats
        
        return analysis

    def _validate_anomalies(
        self,
        true_labels: pd.Series,
        predictions: np.ndarray
    ) -> Dict:
        """Validate anomaly detection against known labels."""
        # Convert predictions from [-1, 1] to [0, 1]
        pred_labels = (predictions == -1).astype(int)
        true_labels = true_labels.astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
        
        # Calculate metrics
        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        }
        
        # Calculate F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = (
                2 * metrics['precision'] * metrics['recall'] /
                (metrics['precision'] + metrics['recall'])
            )
        else:
            metrics['f1_score'] = 0.0
        
        return metrics

    def _generate_anomaly_plots(
        self,
        X: pd.DataFrame,
        predictions: np.ndarray,
        scores: Optional[np.ndarray]
    ):
        """Generate plots for anomaly detection analysis."""
        # Score distribution plot (if scores available)
        if scores is not None:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=scores, hue=predictions == -1, bins=50)
            plt.title('Anomaly Score Distribution')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Count')
            plt.savefig(self.output_dir / 'anomaly_score_distribution.png')
            plt.close()
        
        # Feature distribution plots
        for group_name, features in self.feature_groups.items():
            group_features = [f for f in features if f in X.columns]
            if not group_features:
                continue
            
            n_features = len(group_features)
            if n_features > 0:
                plt.figure(figsize=(15, 5 * ((n_features + 1) // 2)))
                for i, feature in enumerate(group_features):
                    plt.subplot((n_features + 1) // 2, 2, i + 1)
                    sns.boxplot(x=predictions == -1, y=X[feature])
                    plt.title(f'{feature} Distribution')
                    plt.xlabel('Is Anomaly')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f'anomaly_{group_name}_distributions.png')
                plt.close()

    def save_evaluation_results(self, results: Dict, filename: str):
        """Save evaluation results to file."""
        output_path = self.output_dir / filename
        
        # Convert numpy types to Python native types for JSON serialization
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
        
        # Save as JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Evaluation results saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    from sklearn.model_selection import train_test_split
    from ..models import SOHPredictor, CapacityPredictor, AnomalyDetector
    
    # Load data
    df = pd.read_csv('data/raw/comprehensive_battery_data.csv')
    
    # Initialize evaluator
    evaluator = ModelEvaluator(output_dir='evaluation_results')
    
    # Evaluate SOH predictor
    soh_model = SOHPredictor()
    X = df.drop(['SOH'], axis=1)
    y = df['SOH']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    soh_results = evaluator.evaluate_regression_model(
        soh_model, X_train, y_train, X_test, y_test,
        "SOH Predictor", "SOH"
    )
    
    # Generate and save report
    report = evaluator.generate_report(soh_results)
    print(report)