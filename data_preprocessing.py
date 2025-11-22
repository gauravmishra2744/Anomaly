import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import re
import argparse

def load_and_inspect_data(file_path):
    """
    Load the time series data file and inspect its structure.
    """
    print(f"Loading file: {file_path}")
    print("-" * 60)
    
    # Try to read the file - it appears to be one number per line
    try:
        # Read as single column
        df = pd.read_csv(file_path, header=None, names=['value'], dtype=float)
    except Exception as e:
        print(f"Error reading file: {e}")
        # Try reading with whitespace separator
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['value'], dtype=float)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 10 rows:")
    print(df.head(10))
    print(f"\nLast 10 rows:")
    print(df.tail(10))
    
    # Check for missing values
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    
    # Check data types
    print(f"\nData types:\n{df.dtypes}")
    
    # Check for scientific notation (already confirmed from inspection)
    print(f"\nSample values (checking format):")
    print(df['value'].head(10).tolist())
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    # Check for any extra blanks or whitespace issues
    print(f"\nChecking for empty rows or whitespace issues...")
    empty_rows = df['value'].isna().sum()
    print(f"Empty/NaN rows: {empty_rows}")
    
    return df

def clean_and_normalize(df, smooth_window=3):
    """
    Clean the data, handle missing values, optionally smooth, and normalize.
    Returns: (df, scaler, rows_dropped)
    """
    print("\n" + "=" * 60)
    print("CLEANING & NORMALIZING")
    print("=" * 60)
    
    rows_dropped = 0
    
    # Handle missing values
    print(f"\nMissing values before fillna: {df.isnull().sum().sum()}")
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.mean())
        print(f"Missing values after fillna: {df.isnull().sum().sum()}")
    
    # Optional smoothing (if requested)
    if smooth_window > 1:
        print(f"\nApplying smoothing with window={smooth_window}...")
        df = df.copy()  # Avoid SettingWithCopyWarning
        df['value_smoothed'] = df['value'].rolling(window=smooth_window).mean()
        # Drop rows where smoothing created NaN (first smooth_window-1 rows)
        rows_before = len(df)
        df = df.dropna()
        rows_after = len(df)
        rows_dropped = rows_before - rows_after
        # Use smoothed values
        df.loc[:, 'value'] = df['value_smoothed']
        df = df.drop(columns=['value_smoothed'])
        print(f"Shape after smoothing: {df.shape} (dropped {rows_dropped} rows)")
    
    # Normalize to [0, 1]
    print("\nNormalizing values to [0, 1] range...")
    scaler = MinMaxScaler()
    df.loc[:, 'value'] = scaler.fit_transform(df[['value']])
    
    print(f"Normalized value range: [{df['value'].min():.6f}, {df['value'].max():.6f}]")
    print(f"Final shape: {df.shape}")
    
    return df, scaler, rows_dropped

def create_sliding_windows(df, window_size=60, step_size=1):
    """
    Create sliding windows from the time series data for LSTM input.
    
    Args:
        df: DataFrame with time series data
        window_size: Size of each window (sequence length)
        step_size: Step size for sliding window (1 = every window, 2 = skip 1, etc.)
    
    Returns:
        X: Array of shape (n_windows, window_size, n_features)
    """
    print("\n" + "=" * 60)
    print("CREATING SLIDING WINDOWS")
    print("=" * 60)
    
    # Get all feature columns (in case there are multiple)
    feature_cols = [col for col in df.columns if col != 'anomaly_label']
    print(f"Feature columns: {feature_cols}")
    
    # Extract values
    values = df[feature_cols].values
    
    # Create windows
    X = []
    for i in range(0, len(values) - window_size + 1, step_size):
        X.append(values[i:i + window_size])
    
    X = np.array(X)
    
    print(f"Window size: {window_size}")
    print(f"Step size: {step_size}")
    print(f"Original data length: {len(df)}")
    print(f"Number of windows created: {X.shape[0]}")
    print(f"Window shape: {X.shape}")  # (n_windows, window_size, n_features)
    
    return X

def extract_anomaly_labels(file_path, data_length):
    """
    Extract anomaly labels from filename.
    Filename format: ..._start_end_anomaly_start_anomaly_end.txt
    Example: ..._35000_52000_52620.txt means anomaly is from index 52000 to 52620
    """
    filename = os.path.basename(file_path)
    
    # Try to extract anomaly range from filename
    # Pattern: numbers like _52000_52620_ or similar
    numbers = re.findall(r'_(\d+)_(\d+)_(\d+)', filename)
    
    if numbers:
        # Last two numbers are usually the anomaly range
        try:
            anomaly_start = int(numbers[-1][1])  # Second number of last match
            anomaly_end = int(numbers[-1][2])    # Third number of last match
            
            # Create binary labels (1 = anomaly, 0 = normal)
            labels = np.zeros(data_length, dtype=int)
            if anomaly_start < data_length and anomaly_end <= data_length:
                labels[anomaly_start:anomaly_end] = 1
            elif anomaly_start < data_length:
                labels[anomaly_start:] = 1
            
            print(f"\nAnomaly range extracted from filename: [{anomaly_start}, {anomaly_end})")
            print(f"Anomaly labels - Normal: {np.sum(labels == 0)}, Anomaly: {np.sum(labels == 1)}")
            return labels
        except:
            pass
    
    # If extraction fails, return None (no labels)
    print("\nCould not extract anomaly labels from filename. Proceeding without labels.")
    return None

def create_window_labels(labels, window_size, step_size=1, rows_dropped=0):
    """
    Create labels for windows. A window is anomalous if any point in it is anomalous.
    
    Args:
        labels: Original labels array
        window_size: Size of sliding window
        step_size: Step size for sliding window
        rows_dropped: Number of rows dropped during smoothing (if any)
    """
    if labels is None:
        return None
    
    # Adjust labels if rows were dropped during smoothing
    if rows_dropped > 0:
        labels = labels[rows_dropped:]
    
    window_labels = []
    for i in range(0, len(labels) - window_size + 1, step_size):
        window_label = 1 if np.any(labels[i:i + window_size] == 1) else 0
        window_labels.append(window_label)
    
    return np.array(window_labels)

def split_data(X, y=None, train_ratio=0.8):
    """
    Split windows into train and test sets.
    For training, use only normal windows (if labels available).
    """
    print("\n" + "=" * 60)
    print("SPLITTING DATA")
    print("=" * 60)
    
    split_idx = int(len(X) * train_ratio)
    
    if y is not None:
        # Separate normal and anomalous windows
        normal_mask = y == 0
        anomaly_mask = y == 1
        
        X_normal = X[normal_mask]
        X_anomaly = X[anomaly_mask]
        
        # Split normal windows for training
        normal_split = int(len(X_normal) * train_ratio)
        X_train = X_normal[:normal_split]
        X_test_normal = X_normal[normal_split:]
        
        # Add all anomalies to test set
        X_test = np.vstack([X_test_normal, X_anomaly]) if len(X_anomaly) > 0 else X_test_normal
        
        # Create corresponding labels
        y_train = np.zeros(len(X_train))
        y_test_normal = np.zeros(len(X_test_normal))
        y_test_anomaly = np.ones(len(X_anomaly)) if len(X_anomaly) > 0 else np.array([])
        y_test = np.concatenate([y_test_normal, y_test_anomaly]) if len(y_test_anomaly) > 0 else y_test_normal
        
        print(f"Training set (normal only): {X_train.shape}")
        print(f"  - Normal windows: {len(X_train)}")
        print(f"Test set (normal + anomaly): {X_test.shape}")
        print(f"  - Normal windows: {len(X_test_normal)}")
        print(f"  - Anomaly windows: {len(X_anomaly)}")
        
        return X_train, X_test, y_train, y_test
    else:
        # No labels available, simple split
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, None, None

def save_preprocessing_artifacts(scaler, window_size, output_dir='./preprocessing_artifacts'):
    """
    Save scaler and configuration for later use in detection phase.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\nScaler saved to: {scaler_path}")
    
    # Save configuration
    config = {
        'window_size': window_size,
        'scaler_type': 'MinMaxScaler',
        'feature_range': (0, 1)
    }
    config_path = os.path.join(output_dir, 'config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"Config saved to: {config_path}")

def save_data_files(X_train, X_test, y_test=None):
    """
    Save preprocessed data files for training pipeline.
    """
    print("\n" + "=" * 60)
    print("SAVING DATA FILES")
    print("=" * 60)
    
    # Save X_train
    np.save('X_train.npy', X_train)
    print(f"X_train saved to: X_train.npy (shape: {X_train.shape})")
    
    # Save X_test
    np.save('X_test.npy', X_test)
    print(f"X_test saved to: X_test.npy (shape: {X_test.shape})")
    
    # Save y_test if available
    if y_test is not None:
        np.save('y_test.npy', y_test)
        print(f"y_test saved to: y_test.npy (shape: {y_test.shape})")
    else:
        print("y_test not available, skipping save.")

def main(file_path=None, window_size=60, step_size=1, smooth_window=3, train_ratio=0.8):
    """
    Main preprocessing pipeline.
    
    Args:
        file_path: Path to the time series data file
        window_size: Size of sliding window for LSTM
        step_size: Step size for sliding window
        smooth_window: Window size for smoothing (set to 1 to disable)
        train_ratio: Ratio of data to use for training
    """
    # Default file path if not provided
    if file_path is None:
        file_path = r"C:\Users\Shaivy Kashyap\Downloads\UCR_TimeSeriesAnomalyDatasets2021\AnomalyDatasets_2021\UCR_TimeSeriesAnomalyDatasets2021\FilesAreInHere\UCR_Anomaly_FullData\001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"
    
    print("=" * 60)
    print("TIME SERIES DATA PREPROCESSING FOR LSTM")
    print("=" * 60)
    
    # Step 1: Load and inspect
    df = load_and_inspect_data(file_path)
    
    # Step 2: Extract anomaly labels (if available)
    labels = extract_anomaly_labels(file_path, len(df))
    if labels is not None:
        df['anomaly_label'] = labels
    
    # Step 3: Clean and normalize
    df, scaler, rows_dropped = clean_and_normalize(df, smooth_window=smooth_window)
    
    # Adjust labels if rows were dropped during smoothing
    if labels is not None and rows_dropped > 0:
        labels = labels[rows_dropped:]
        print(f"\nAdjusted labels after smoothing: Normal: {np.sum(labels == 0)}, Anomaly: {np.sum(labels == 1)}")
    
    # Step 4: Create sliding windows
    X = create_sliding_windows(df, window_size=window_size, step_size=step_size)
    
    # Step 5: Create window labels (if available)
    y = create_window_labels(labels, window_size, step_size, rows_dropped=0) if labels is not None else None
    
    # Step 6: Split data
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=train_ratio)
    
    # Step 7: Save preprocessing artifacts
    save_preprocessing_artifacts(scaler, window_size)
    
    # Step 8: Save data files for training
    save_data_files(X_train, X_test, y_test)
    
    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nFinal Data Shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    if y_train is not None:
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess time series data for LSTM anomaly detection')
    parser.add_argument('--file', type=str, default=None, help='Path to the time series data file')
    parser.add_argument('--window_size', type=int, default=60, help='Size of sliding window (default: 60)')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for sliding window (default: 1)')
    parser.add_argument('--smooth_window', type=int, default=3, help='Window size for smoothing, set to 1 to disable (default: 3)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data for training (default: 0.8)')
    
    args = parser.parse_args()
    
    X_train, X_test, y_train, y_test, scaler = main(
        file_path=args.file,
        window_size=args.window_size,
        step_size=args.step_size,
        smooth_window=args.smooth_window,
        train_ratio=args.train_ratio
    )

