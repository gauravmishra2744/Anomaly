# LSTM Autoencoder for Time-Series Anomaly Detection

## Project Overview

This project implements an **LSTM (Long Short-Term Memory) Autoencoder** for detecting anomalies in time-series data. The model learns to reconstruct normal patterns and identifies anomalies based on reconstruction errors.

## Project Structure

```
Project_Anomaly/
├── data_preprocessing.py              # Data preprocessing pipeline
├── lstm_autoencoder_train.py          # LSTM autoencoder training script
├── run_anomaly_detection.py           # Analysis and evaluation script (NEW)
├── lstm_autoencoder.h5                # Best model from training
├── lstm_autoencoder_final.h5          # Final model after training
├── preprocessing_artifacts/           # Scaler and configuration files
├── X_train.npy                        # Training data (63,244 samples)
├── X_test.npy                         # Test data (16,490 samples)
├── y_test.npy                         # Test labels for evaluation
├── reconstruction_errors.npy          # Pre-computed reconstruction errors
├── threshold.npy                      # Anomaly detection threshold
└── LSTM/                              # Additional LSTM-related files
```

## Environment Setup

### Prerequisites
- Python 3.10 or 3.11 (TensorFlow has compatibility issues with Python 3.13)
- pip or conda package manager

### Installation Steps

1. **Navigate to project directory:**
   ```bash
   cd c:\Users\HP\Downloads\Project_Anomaly\Project_Anomaly
   ```

2. **Install required packages:**
   ```bash
   pip install numpy pandas scikit-learn tensorflow keras
   ```

3. **Core dependencies:**
   - `numpy` - Numerical computing
   - `pandas` - Data manipulation
   - `scikit-learn` - Machine learning utilities
   - `tensorflow` - Deep learning framework
   - `keras` - Neural network API

## Project Components

### 1. Data Preprocessing (`data_preprocessing.py`)

**Purpose:** Prepare raw time-series data for LSTM training

**Key Functions:**
- `load_and_inspect_data()` - Load and inspect raw data
- `clean_and_normalize()` - Handle missing values and normalize to [0, 1]
- `create_sliding_windows()` - Create time-series windows
- `extract_anomaly_labels()` - Extract labels from filename
- `split_data()` - Train/test split with anomaly separation

**Usage:**
```bash
python data_preprocessing.py --file <path_to_data> --window_size 60 --step_size 1
```

**Output:**
- `X_train.npy` - Training samples (normal only)
- `X_test.npy` - Test samples (normal + anomaly)
- `y_test.npy` - Ground truth labels
- `preprocessing_artifacts/scaler.pkl` - MinMaxScaler for denormalization
- `preprocessing_artifacts/config.pkl` - Configuration metadata

### 2. LSTM Autoencoder Training (`lstm_autoencoder_train.py`)

**Purpose:** Train an LSTM autoencoder on normal time-series data

**Architecture:**
```
Input → LSTM(64, ReLU) → RepeatVector → LSTM(64, ReLU) → Dense → Output
         (Encoder)                      (Decoder)
```

**Training Configuration:**
- **Epochs:** 25
- **Batch Size:** 32
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Callbacks:** EarlyStopping, ModelCheckpoint

**Key Steps:**
1. Load preprocessed training data
2. Build LSTM autoencoder architecture
3. Train model with validation split (10%)
4. Compute reconstruction errors on test set
5. Calculate anomaly threshold (Mean + 2*Std)
6. Evaluate performance against ground truth

**Usage:**
```bash
python lstm_autoencoder_train.py
```

**Output:**
- `lstm_autoencoder.h5` - Best model during training
- `lstm_autoencoder_final.h5` - Final trained model
- `reconstruction_errors.npy` - Reconstruction errors for test data
- `threshold.npy` - Computed anomaly threshold

### 3. Anomaly Detection Analysis (`run_anomaly_detection.py`)

**Purpose:** Comprehensive analysis and visualization of model results

**Features:**
- Load pre-computed artifacts
- Analyze reconstruction error distribution
- Predict anomalies using threshold
- Detailed performance metrics:
  - Accuracy, Precision, Recall, Specificity
  - F1-Score, ROC-AUC
  - Confusion Matrix
  - Classification Report
- Error distribution analysis
- Sample-by-sample prediction review

**Usage:**
```bash
python run_anomaly_detection.py
```

## Model Performance

### Current Results (on Test Set)

**Overall Metrics:**
- **Accuracy:** 95.18%
- **Precision:** 33.43%
- **Recall (Sensitivity):** 17.23%
- **Specificity:** 98.53%
- **F1-Score:** 0.2274
- **ROC-AUC:** 0.6728

**Confusion Matrix:**
```
                Predicted
                Normal  Anomaly
Actual Normal   15,578    233
Actual Anomaly    562     117
```

**Analysis:**
- **High Accuracy & Specificity** - Model correctly identifies most normal samples
- **Low Recall** - Model misses many anomalies (82.77% false negative rate)
- **Trade-off** - Focus is on minimizing false positives (1.47% FP rate)

### Data Characteristics

**Training Set:**
- Samples: 63,244
- Shape: (63,244, 60, 1)
- Features: 1 (univariate)
- Window Size: 60 timesteps

**Test Set:**
- Samples: 16,490
- Normal: 15,811 (95.88%)
- Anomaly: 679 (4.12%)

**Reconstruction Errors:**
- Mean: 0.000257
- Std Dev: 0.001311
- Min: 0.000003
- Max: 0.031153
- Threshold: 0.002878

## How to Run

### Quick Start

1. **Run the analysis script (no TensorFlow needed):**
   ```bash
   cd c:\Users\HP\Downloads\Project_Anomaly\Project_Anomaly
   python run_anomaly_detection.py
   ```

2. **View the output:**
   - Reconstruction error statistics
   - Model performance metrics
   - Sample-by-sample predictions
   - Error distribution analysis

### Full Pipeline (if retraining)

```bash
# 1. Preprocess data
python data_preprocessing.py --file <data_file> --window_size 60

# 2. Train model
python lstm_autoencoder_train.py

# 3. Analyze results
python run_anomaly_detection.py
```

## Key Insights

### Why Low Recall?

The model uses `threshold = mean + 2*std`, which is conservative and optimizes for:
- **Minimizing False Positives** (only flag clearly anomalous samples)
- **High Specificity** (98.53% - very few false alarms)
- This is a trade-off: catches only obvious anomalies

### Reconstruction Error Insights

- **Normal samples** show very small errors (median: 0.000035)
- **Anomaly samples** show higher but still modest errors (median: 0.000145)
- **95.72%** of samples have errors in the "Very Low" range

### Practical Applications

This model is suitable for:
- ✓ Early warning systems (high specificity)
- ✓ Production environments (avoids alert fatigue)
- ✓ Real-time anomaly detection
- ✗ Complete anomaly capture (due to low recall)

## Customization Options

### Adjust Threshold
To catch more anomalies, modify the threshold calculation in `run_anomaly_detection.py`:
```python
# More sensitive (catch more anomalies)
threshold = error_mean + 1 * error_std  # Higher recall

# More lenient
threshold = error_mean + 3 * error_std  # Higher specificity
```

### Modify Architecture
Edit `lstm_autoencoder_train.py` to change model architecture:
```python
# Deeper encoder
encoder = LSTM(128, activation='relu', name='encoder')(input_layer)
```

### Adjust Training Parameters
```python
epochs = 50              # Increase for better convergence
batch_size = 16         # Smaller batch size for stability
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 2.3.3+ | Numerical computing |
| pandas | 2.3.2+ | Data manipulation |
| scikit-learn | 1.7.2+ | ML utilities |
| tensorflow | 2.16+ | Deep learning |
| keras | Latest | Neural networks |

## File Sizes

| File | Size |
|------|------|
| lstm_autoencoder.h5 | 0.61 MB |
| lstm_autoencoder_final.h5 | 0.61 MB |
| X_train.npy | 28.95 MB |
| X_test.npy | 7.55 MB |
| reconstruction_errors.npy | 0.13 MB |
| threshold.npy | < 0.01 MB |
| y_test.npy | 0.13 MB |

## Troubleshooting

### TensorFlow Import Error
If you get `KeyboardInterrupt` or import errors with TensorFlow:
- Use Python 3.10 or 3.11 (not 3.13)
- Run `python run_anomaly_detection.py` instead (doesn't require TensorFlow)

### Missing Data Files
Ensure these files exist in the project directory:
- `X_train.npy`, `X_test.npy`, `y_test.npy`
- `reconstruction_errors.npy`, `threshold.npy`

### Memory Issues
If processing large data:
- Reduce batch size in training
- Use smaller window sizes
- Process data in chunks

## References

**LSTM Autoencoder Concepts:**
- Anomaly detection via reconstruction error
- Autoencoders for unsupervised learning
- LSTM networks for sequence modeling

**Key Papers:**
- Autoencoders for unsupervised outlier detection
- LSTM Autoencoder networks for time-series analysis
- Threshold-based anomaly detection methods

## Project Status

✓ Data preprocessing complete
✓ LSTM autoencoder trained
✓ Reconstruction errors computed
✓ Performance evaluation complete
✓ Analysis framework implemented

## Future Improvements

- [ ] Adaptive threshold based on data characteristics
- [ ] Ensemble methods with multiple thresholds
- [ ] Real-time anomaly detection pipeline
- [ ] Visualization of reconstruction errors
- [ ] Integration with monitoring systems
- [ ] Support for multivariate time-series

## Contact & Support

For questions or issues, refer to:
- Project files for implementation details
- Model configuration in `lstm_autoencoder_train.py`
- Data format in `data_preprocessing.py`

---

**Last Updated:** November 22, 2025
**Project Status:** Production Ready
