# main.py
import pandas as pd
import os
import sys
from battery_analysis.preprocessing import DataPreprocessor
from battery_analysis.models import SOHPredictor, CapacityPredictor, AnomalyDetector
from battery_analysis.training import ModelEvaluator
from battery_analysis.utils.visualization import BatteryVisualizer
from battery_analysis.utils.logger import setup_logger
from battery_analysis.utils.config import Config
from battery_analysis.preprocessing.feature_engineering import FeatureEngineer

def load_config():
    """Load and initialize configuration."""
    config = Config()
    config.ensure_directories()
    return config

def save_pipeline_results(config: Config, df_featured, metrics, anomalies):
    """Save the results of the analysis pipeline."""
    # Save processed data
    df_featured.to_csv(config.PROCESSED_DATA_DIR / 'processed_data.csv', index=False)
    
    # Save metrics
    pd.DataFrame(metrics).to_csv(config.RESULTS_DIR / 'model_metrics.csv')
    
    # Save anomalies
    pd.Series(anomalies).to_csv(config.RESULTS_DIR / 'anomalies.csv')

def prepare_features_and_target(df: pd.DataFrame, target_col: str = 'SOH'):
    """Prepare features and target variables."""
    # Remove any non-feature columns (adjust as needed)
    non_feature_cols = ['time']  # Add any other non-feature columns
    feature_cols = [col for col in df.columns if col not in non_feature_cols + [target_col]]
    
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y

def main():
    try:
        # Setup logger
        logger = setup_logger("main")
        logger.info("Starting battery analysis pipeline")

        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()

        # 1. Load data
        logger.info("Loading data...")
        data_path = config.RAW_DATA_DIR / 'merged_data.csv'
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows of data")

        # 2. Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor()
        df_processed, preprocessing_stats = preprocessor.preprocess_data(
            df, config=config, target='SOH'
        )
        logger.info(f"Preprocessing completed with stats:")
        for key, value in preprocessing_stats.items():
            logger.info(f"{key}: {value}")

        # 3. Engineer features
        logger.info("Engineering features...")
        feature_engineer = FeatureEngineer(config)
        df_featured = feature_engineer.create_features(df_processed)
        logger.info(f"Created features successfully")

        # 4. Train and evaluate models
        logger.info("Training and evaluating models...")
        
        metrics = {}
        
        # Prepare features and targets for each model
        X_soh, y_soh = prepare_features_and_target(df_featured, target_col='SOH')
        X_capacity, y_capacity = prepare_features_and_target(df_featured, target_col='capacity')
        
        # SOH Prediction
        try:
            soh_predictor = SOHPredictor(
                model_type='xgboost',
                config=config
            )
            metrics['soh'] = soh_predictor.train(X_soh, y_soh)
            logger.info(f"SOH Prediction Metrics: {metrics['soh']}")
        except Exception as e:
            logger.error(f"Error in SOH prediction: {str(e)}")
            metrics['soh'] = {'error': str(e)}

        # Capacity Prediction
        try:
            capacity_predictor = CapacityPredictor(**config.CAPACITY_MODEL_PARAMS)
            metrics['capacity'] = capacity_predictor.train(X_capacity, y_capacity)
            logger.info(f"Capacity Prediction Metrics: {metrics['capacity']}")
        except Exception as e:
            logger.error(f"Error in capacity prediction: {str(e)}")
            metrics['capacity'] = {'error': str(e)}

        # Anomaly Detection
        try:
            anomaly_detector = AnomalyDetector(**config.ANOMALY_DETECTOR_PARAMS)
            # For anomaly detection, we use all features
            anomalies = anomaly_detector.fit_predict(X_soh)  # Using SOH features for anomaly detection
            anomaly_stats = anomaly_detector.analyze_anomalies(df_featured, anomalies)
            logger.info(f"Anomaly Detection Stats: {anomaly_stats}")
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            anomalies = pd.Series([False] * len(df_featured))
            anomaly_stats = {'error': str(e)}

        # 5. Create visualizations
        logger.info("Creating visualizations...")
        visualizer = BatteryVisualizer()
        
        # Generate and save all plots
        plot_functions = [
            ('soh_degradation', visualizer.plot_soh_degradation),
            ('capacity_fade', visualizer.plot_capacity_fade),
            ('temperature_analysis', visualizer.plot_temperature_analysis),
            ('voltage_current', visualizer.plot_voltage_current)
        ]
        
        for plot_name, plot_func in plot_functions:
            try:
                figure = plot_func(df_featured)
                figure.savefig(config.VISUALIZATION_DIR / f'{plot_name}.png')
                figure.close()
            except Exception as e:
                logger.error(f"Error creating {plot_name} plot: {str(e)}")

        # 6. Save results
        logger.info("Saving results...")
        save_pipeline_results(config, df_featured, metrics, anomalies)

        logger.info("Battery analysis pipeline completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error in battery analysis pipeline: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())