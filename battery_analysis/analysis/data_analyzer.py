import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pathlib import Path

def analyze_data_quality(file_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze the quality of comprehensive battery data and provide a detailed report.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing comprehensive battery data
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict]
        The loaded data and analysis report
    """
    # Load the data
    print("Loading comprehensive battery data...")
    df = pd.read_csv(file_path)
    
    analysis_report = {}
    
    # Basic information
    print("\n=== Basic Information ===")
    analysis_report['basic_info'] = {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'columns': df.columns.tolist()
    }
    print(f"Number of rows: {analysis_report['basic_info']['num_rows']}")
    print(f"Number of columns: {analysis_report['basic_info']['num_columns']}")
    print("\nColumns:", analysis_report['basic_info']['columns'])
    
    # Feature groups analysis
    feature_groups = {
        'voltage_features': [col for col in df.columns if 'voltage' in col.lower()],
        'current_features': [col for col in df.columns if 'current' in col.lower()],
        'temperature_features': [col for col in df.columns if 'temp' in col.lower()],
        'capacity_features': [col for col in df.columns if 'capacity' in col.lower()],
        'soh_features': [col for col in df.columns if 'soh' in col.lower()],
        'efficiency_features': [col for col in df.columns if 'efficiency' in col.lower()],
        'power_features': [col for col in df.columns if 'power' in col.lower()],
        'stress_features': [col for col in df.columns if 'stress' in col.lower()]
    }
    
    print("\n=== Feature Groups ===")
    analysis_report['feature_groups'] = {
        group: features for group, features in feature_groups.items() if features
    }
    for group, features in analysis_report['feature_groups'].items():
        print(f"\n{group}:")
        print(features)
    
    # Data types
    print("\n=== Data Types ===")
    analysis_report['data_types'] = df.dtypes.to_dict()
    print(df.dtypes)
    
    # Missing values analysis
    print("\n=== Missing Values Analysis ===")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    analysis_report['missing_values'] = missing_info[missing_info['Missing Values'] > 0].to_dict()
    print(missing_info[missing_info['Missing Values'] > 0])
    
    # Statistical analysis by feature group
    print("\n=== Statistical Analysis by Feature Group ===")
    analysis_report['statistics'] = {}
    
    for group, features in feature_groups.items():
        if features:
            group_stats = df[features].describe()
            analysis_report['statistics'][group] = group_stats.to_dict()
            print(f"\n{group}:")
            print(group_stats)
    
    # Correlation analysis
    print("\n=== Key Correlations ===")
    target_variables = ['SOH', 'capacity']
    correlations = {}
    for target in target_variables:
        if target in df.columns:
            corr = df.corr()[target].sort_values(ascending=False)
            correlations[target] = corr.to_dict()
            print(f"\nTop correlations with {target}:")
            print(corr.head())
    analysis_report['correlations'] = correlations
    
    # Check for invalid values
    print("\n=== Invalid Values Check ===")
    invalid_checks = {
        'negative_values': {},
        'zero_values': {},
        'out_of_range': {}
    }
    
    for column in df.select_dtypes(include=[np.number]).columns:
        # Check negative values
        neg_count = (df[column] < 0).sum()
        if neg_count > 0:
            invalid_checks['negative_values'][column] = int(neg_count)
        
        # Check zero values where they might be invalid
        if any(keyword in column.lower() for keyword in ['efficiency', 'temperature', 'voltage']):
            zero_count = (df[column] == 0).sum()
            if zero_count > 0:
                invalid_checks['zero_values'][column] = int(zero_count)
    
    analysis_report['invalid_checks'] = invalid_checks
    print("\nNegative values found:", invalid_checks['negative_values'])
    print("Zero values in critical features:", invalid_checks['zero_values'])
    
    # Cycle analysis
    if 'cycle' in df.columns:
        print("\n=== Cycle Analysis ===")
        cycle_stats = {
            'total_cycles': df['cycle'].nunique(),
            'min_cycle': int(df['cycle'].min()),
            'max_cycle': int(df['cycle'].max())
        }
        analysis_report['cycle_stats'] = cycle_stats
        print(f"Total unique cycles: {cycle_stats['total_cycles']}")
        print(f"Cycle range: {cycle_stats['min_cycle']} to {cycle_stats['max_cycle']}")
    
    return df, analysis_report

def save_analysis_report(report: Dict, output_dir: str = 'analysis_results'):
    """Save the analysis report to a file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert the report to a more readable format
    report_df = pd.DataFrame.from_dict(report, orient='index')
    report_df.to_csv(output_path / 'data_analysis_report.csv')
    
    # Save correlation matrices separately
    if 'correlations' in report:
        for target, corr in report['correlations'].items():
            pd.Series(corr).to_csv(output_path / f'{target}_correlations.csv')

if __name__ == "__main__":
    # Example usage
    df, report = analyze_data_quality('data/raw/comprehensive_battery_data.csv')
    save_analysis_report(report)
    print("\nAnalysis complete. Data and report ready for further processing.")