import pandas as pd
import numpy as np

def analyze_data_quality(file_path):
    """
    Analyze the quality of battery data and provide a detailed report.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing battery data
        
    Returns:
    --------
    pd.DataFrame
        The loaded data for further processing
    """
    # Load the data
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Basic information
    print("\n=== Basic Information ===")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns:", df.columns.tolist())
    
    # Data types
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    # Missing values analysis
    print("\n=== Missing Values Analysis ===")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    print(missing_info[missing_info['Missing Values'] > 0])
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(df.describe())
    
    # Check for negative values
    print("\n=== Negative Values Count ===")
    negative_counts = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        negative_count = (df[column] < 0).sum()
        if negative_count > 0:
            negative_counts[column] = negative_count
    print(negative_counts)
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicate_count}")
    
    # Value ranges for key columns
    print("\n=== Value Ranges for Key Columns ===")
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            print(f"\n{column}:")
            print(f"Min: {df[column].min()}")
            print(f"Max: {df[column].max()}")
            print(f"Unique values: {df[column].nunique()}")
    
    return df

if __name__ == "__main__":
    # Example usage
    df = analyze_data_quality('merged_data.csv')
    print("\nAnalysis complete. Data returned as DataFrame for further processing.")


