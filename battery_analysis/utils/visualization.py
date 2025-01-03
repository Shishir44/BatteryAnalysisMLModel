import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class BatteryVisualizer:
    """Enhanced visualization class for battery analysis system."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self._setup_plotting_style()
        
    def _setup_plotting_style(self) -> None:
        """Configure plotting style settings."""
        plt.style.use(self.config.VISUALIZATION['style'])
        plt.rcParams['figure.figsize'] = self.config.VISUALIZATION['figure_size']
        plt.rcParams['figure.dpi'] = self.config.VISUALIZATION['dpi']
        sns.set_palette(self.config.VISUALIZATION['color_palette'])
    
    def save_plot(self, fig: plt.Figure, filename: str) -> None:
        """
        Save plot to visualization directory.
        
        Args:
            fig: Figure to save
            filename: Output filename
        """
        save_path = Path(self.config.VISUALIZATION_DIR) / filename
        fig.savefig(save_path, bbox_inches='tight', dpi=self.config.VISUALIZATION['dpi'])
        plt.close(fig)
        logger.info(f"Plot saved to {save_path}")

    def plot_soh_degradation(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
        add_confidence: bool = True
    ) -> plt.Figure:
        """
        Plot SOH degradation over cycles with confidence intervals.
        
        Args:
            df: DataFrame containing cycle and SOH data
            save_path: Optional path to save the plot
            add_confidence: Whether to add confidence intervals
            
        Returns:
            Created figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot individual points
        sns.scatterplot(data=df, x='cycle', y='SOH', alpha=0.5, ax=ax)
        
        # Calculate and plot trend line with confidence interval
        if add_confidence:
            sns.regplot(
                data=df,
                x='cycle',
                y='SOH',
                scatter=False,
                color='red',
                ax=ax
            )
        
        # Add threshold lines
        warning_threshold = self.config.BATTERY['health_thresholds']['soh_warning']
        critical_threshold = self.config.BATTERY['health_thresholds']['soh_critical']
        
        ax.axhline(y=warning_threshold, color='yellow', linestyle='--', alpha=0.5)
        ax.axhline(y=critical_threshold, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('State of Health (%)')
        ax.set_title('Battery State of Health Degradation')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            self.save_plot(fig, save_path)
        
        return fig

    def plot_voltage_curves(
        self,
        df: pd.DataFrame,
        cycles: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot voltage curves for specified cycles.
        
        Args:
            df: DataFrame containing voltage data
            cycles: List of cycles to plot
            save_path: Optional path to save the plot
            
        Returns:
            Created figure
        """
        if cycles is None:
            # Select evenly spaced cycles
            all_cycles = sorted(df['cycle'].unique())
            cycles = all_cycles[::len(all_cycles)//5]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for cycle in cycles:
            cycle_data = df[df['cycle'] == cycle]
            ax.plot(
                cycle_data['charge_current'].cumsum(),
                cycle_data['terminal_voltage'],
                label=f'Cycle {cycle}'
            )
        
        ax.set_xlabel('Charge Throughput')
        ax.set_ylabel('Terminal Voltage (V)')
        ax.set_title('Voltage Curves at Different Cycles')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            self.save_plot(fig, save_path)
        
        return fig

    def plot_temperature_analysis(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comprehensive temperature analysis.
        
        Args:
            df: DataFrame containing temperature data
            save_path: Optional path to save the plot
            
        Returns:
            Created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Temperature evolution
        sns.scatterplot(
            data=df,
            x='cycle',
            y='temperature',
            hue='SOH',
            alpha=0.5,
            ax=ax1
        )
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Temperature Evolution')
        
        # Temperature distribution by SOH ranges
        df['SOH_range'] = pd.cut(df['SOH'], bins=5)
        sns.boxplot(
            data=df,
            x='SOH_range',
            y='temperature',
            ax=ax2
        )
        ax2.set_xlabel('SOH Range')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Temperature Distribution by SOH')
        
        if save_path:
            self.save_plot(fig, save_path)
        
        return fig

    def plot_capacity_analysis(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comprehensive capacity analysis.
        
        Args:
            df: DataFrame containing capacity data
            save_path: Optional path to save the plot
            
        Returns:
            Created figure
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Create grid for subplots
        gs = plt.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        
        # Capacity vs Cycle
        sns.scatterplot(
            data=df,
            x='cycle',
            y='capacity',
            alpha=0.5,
            ax=ax1
        )
        ax1.set_title('Capacity Fade')
        
        # Capacity Distribution
        sns.histplot(
            data=df,
            x='capacity',
            kde=True,
            ax=ax2
        )
        ax2.set_title('Capacity Distribution')
        
        # Capacity vs Temperature with SOH coloring
        scatter = ax3.scatter(
            df['temperature'],
            df['capacity'],
            c=df['SOH'],
            cmap='viridis',
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax3, label='SOH')
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Capacity')
        ax3.set_title('Capacity vs Temperature')
        
        plt.tight_layout()
        
        if save_path:
            self.save_plot(fig, save_path)
        
        return fig

    def plot_model_performance(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        model_name: str,
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot model performance analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            metrics: Dictionary of performance metrics
            save_path: Optional path to save the plot
            
        Returns:
            Created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Actual vs Predicted
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Actual vs Predicted')
        
        # Residuals Plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals Plot')
        
        # Add metrics text
        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        fig.text(0.98, 0.98, f'Metrics:\n{metrics_text}',
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'{model_name} Performance Analysis')
        plt.tight_layout()
        
        if save_path:
            self.save_plot(fig, save_path)
        
        return fig

    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        top_n: int = 10,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance analysis.
        
        Args:
            feature_importance: Dictionary of feature importance scores
            top_n: Number of top features to show
            save_path: Optional path to save the plot
            
        Returns:
            Created figure
        """
        # Sort features by importance
        sorted_features = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create barplot
        bars = ax.barh(list(sorted_features.keys()),
                      list(sorted_features.values()))
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}',
                   ha='left', va='center', fontsize=10)
        
        ax.set_xlabel('Importance Score')
        ax.set_title('Top Feature Importance')
        
        plt.tight_layout()
        
        if save_path:
            self.save_plot(fig, save_path)
        
        return fig

    def plot_anomaly_detection(
        self,
        df: pd.DataFrame,
        anomalies: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot anomaly detection results.
        
        Args:
            df: DataFrame containing battery data
            anomalies: Array of anomaly indicators
            save_path: Optional path to save the plot
            
        Returns:
            Created figure
        """
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, figure=fig)
        
        # Voltage vs Current
        ax1 = fig.add_subplot(gs[0, 0])
        scatter = ax1.scatter(
            df['terminal_current'],
            df['terminal_voltage'],
            c=anomalies,
            cmap='coolwarm',
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax1, label='Anomaly')
        ax1.set_xlabel('Current (A)')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title('V-I Characteristics')
        
        # Temperature Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        for label, group in [(0, 'Normal'), (1, 'Anomaly')]:
            mask = anomalies == label
            sns.kdeplot(
                data=df[mask]['temperature'],
                label=group,
                ax=ax2
            )
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_title('Temperature Distribution')
        ax2.legend()
        
        # Capacity vs Cycle with anomalies
        ax3 = fig.add_subplot(gs[1, :])
        scatter = ax3.scatter(
            df['cycle'],
            df['capacity'],
            c=anomalies,
            cmap='coolwarm',
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax3, label='Anomaly')
        ax3.set_xlabel('Cycle')
        ax3.set_ylabel('Capacity')
        ax3.set_title('Capacity Evolution with Anomalies')
        
        plt.tight_layout()
        
        if save_path:
            self.save_plot(fig, save_path)
        
        return fig