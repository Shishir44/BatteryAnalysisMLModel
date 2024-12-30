# battery_ml/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from ..utils.logger import setup_logger
from ..utils.config import Config

logger = setup_logger(__name__)

class BatteryVisualizer:
    def __init__(self):
        self.config = Config
        # Set style for better visualizations
        plt.style.use('default')
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['font.size'] = 12
    
    def plot_soh_degradation(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot SOH degradation over cycles."""
        plt.figure(figsize=(14, 7))
        sns.scatterplot(data=df, x='cycle', y='SOH', alpha=0.5)
        plt.plot(df.groupby('cycle')['SOH'].mean(), 'r-', linewidth=2)
        plt.xlabel('Cycle Number')
        plt.ylabel('State of Health (%)')
        plt.title('Battery State of Health Degradation')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_capacity_fade(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot capacity fade analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
        
        # Capacity vs Cycle
        sns.scatterplot(data=df, x='cycle', y='capacity', ax=ax1, alpha=0.5)
        ax1.plot(df.groupby('cycle')['capacity'].mean(), 'r-', linewidth=2)
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Capacity')
        ax1.set_title('Capacity Fade over Cycles')
        ax1.grid(True)
        
        # Capacity Distribution
        sns.histplot(data=df, x='capacity', kde=True, ax=ax2)
        ax2.set_title('Capacity Distribution')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_temperature_analysis(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot temperature-related analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
        
        # Temperature Evolution
        sns.scatterplot(data=df, x='cycle', y='temperature', alpha=0.5, ax=ax1)
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Temperature Evolution over Cycles')
        
        # Temperature vs SOH
        sns.scatterplot(data=df, x='temperature', y='SOH', alpha=0.5, ax=ax2)
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('State of Health (%)')
        ax2.set_title('Temperature Impact on SOH')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_voltage_current(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot voltage-current characteristics."""
        plt.figure(figsize=(14, 7))
        scatter = plt.scatter(df['terminal_current'], df['terminal_voltage'], 
                            c=df['cycle'], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Cycle Number')
        plt.xlabel('Current (A)')
        plt.ylabel('Voltage (V)')
        plt.title('Voltage-Current Characteristics')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_model_predictions(self, y_true: pd.Series, y_pred: pd.Series, 
                             title: str, save_path: Optional[str] = None):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(10, 10))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_feature_importance(self, model, feature_names: List[str], 
                              title: str, save_path: Optional[str] = None):
        """Plot feature importance."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 6))
            plt.title(f"Feature Importances ({title})")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
    
    def plot_anomaly_detection(self, df: pd.DataFrame, anomalies: pd.Series, 
                             save_path: Optional[str] = None):
        """Plot anomaly detection results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
        
        # Voltage vs Current with anomalies highlighted
        scatter = ax1.scatter(df['terminal_current'], df['terminal_voltage'], 
                            c=anomalies, cmap='coolwarm', alpha=0.5)
        ax1.set_xlabel('Current (A)')
        ax1.set_ylabel('Voltage (V)')
        ax1.set_title('Anomalies in V-I Characteristics')
        plt.colorbar(scatter, ax=ax1)
        
        # Temperature distribution for normal vs anomaly points
        sns.kdeplot(data=df[anomalies == 0], x='temperature', ax=ax2, label='Normal')
        sns.kdeplot(data=df[anomalies == 1], x='temperature', ax=ax2, label='Anomaly')
        ax2.set_title('Temperature Distribution: Normal vs Anomaly')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/raw/merged_data.csv')
    
    # Initialize visualizer
    visualizer = BatteryVisualizer()
    
    # Create visualizations
    visualizer.plot_soh_degradation(df, save_path='plots/soh_degradation.png')
    visualizer.plot_capacity_fade(df, save_path='plots/capacity_fade.png')
    visualizer.plot_temperature_analysis(df, save_path='plots/temperature_analysis.png')
    visualizer.plot_voltage_current(df, save_path='plots/voltage_current.png')