# Execution Guide - Integrated Anomaly Detection System

## Quick Start (5 minutes)

### Option 1: Run Analysis Only (No TensorFlow Required)
```bash
cd c:\Users\HP\Downloads\Project_Anomaly\Project_Anomaly
python run_anomaly_detection.py
```

**Output:**
- Reconstruction error statistics
- Performance metrics (Accuracy, Precision, Recall, F1-Score)
- Confusion matrix
- Sample predictions
- Error distribution analysis

---

## Complete Pipeline Execution (15 minutes)

### Run Full Integrated System
```bash
python main_pipeline.py
```

**Executes All Phases:**
1. **Phase 1:** Data Analysis & Preprocessing
2. **Phase 2:** Generative AI - Data Augmentation
3. **Phase 3:** LSTM Model Evaluation
4. **Phase 4:** Explainable AI - Interpretability
5. **Phase 5:** System Integration
6. **Phase 6:** Visualization & Monitoring Dashboard
7. **Phase 7:** Comprehensive Analysis Report

**Output Files Generated:**
- `COMPREHENSIVE_REPORT.txt` - Detailed analysis report
- `anomaly_detection_report.txt` - Dashboard report

---

## Individual Component Execution

### 1. Data Preprocessing
```bash
python data_preprocessing.py --file <path_to_data> --window_size 60 --step_size 1
```

**Parameters:**
- `--file`: Path to time-series data file
- `--window_size`: Size of sliding window (default: 60)
- `--step_size`: Step size for sliding window (default: 1)
- `--smooth_window`: Smoothing window size (default: 3)
- `--train_ratio`: Train/test split ratio (default: 0.8)

**Output:**
- `X_train.npy` - Training data
- `X_test.npy` - Test data
- `y_test.npy` - Test labels
- `preprocessing_artifacts/scaler.pkl` - MinMaxScaler
- `preprocessing_artifacts/config.pkl` - Configuration

---

### 2. LSTM Model Training
```bash
python lstm_autoencoder_train.py
```

**Configuration:**
- Epochs: 25
- Batch Size: 32
- Validation Split: 10%
- Loss Function: MSE
- Optimizer: Adam

**Output:**
- `lstm_autoencoder.h5` - Best model
- `lstm_autoencoder_final.h5` - Final model
- `reconstruction_errors.npy` - Reconstruction errors
- `threshold.npy` - Anomaly threshold

---

### 3. GenAI Data Augmentation
```bash
python genai_augmentation.py
```

**Features:**
- Generates synthetic normal samples
- Creates synthetic anomalies (Spike, Shift, Trend)
- Augments training data by 30%

**Output:**
- Augmented training data
- Synthetic anomalies
- VAE configuration

---

### 4. XAI Explainability
```bash
python xai_explainability.py
```

**Methods:**
- Feature importance analysis
- SHAP-like perturbation approach
- LIME-like local linear approximation

**Output:**
- Explanation reports
- Feature importance scores
- XAI configuration

---

### 5. Integrated System
```bash
python integrated_system.py
```

**Capabilities:**
- Unified interface for all components
- Batch and single-sample prediction
- Comprehensive evaluation
- Augmentation impact analysis
- Synthetic test set generation

---

### 6. Visualization Dashboard
```bash
python visualization_dashboard.py
```

**Features:**
- System configuration display
- Batch statistics
- Error distribution analysis
- Real-time monitoring interface
- Report generation

---

## Python API Usage

### Basic Prediction with Explanation
```python
from integrated_system import IntegratedAnomalyDetectionSystem
import numpy as np

# Initialize system
system = IntegratedAnomalyDetectionSystem()

# Load sample
X_test = np.load('X_test.npy')
reconstruction_errors = np.load('reconstruction_errors.npy')

# Make prediction with explanation
result = system.predict_with_explanation(
    X_test[0], 
    reconstruction_errors[0]
)

print(f"Prediction: {result['prediction']}")
print(f"Explanation: {result['explanation']}")
```

### Batch Predictions
```python
# Batch predictions
results = system.batch_predict_with_explanations(
    X_test[:100], 
    reconstruction_errors[:100]
)

for i, result in enumerate(results):
    print(f"Sample {i}: {result['prediction']}")
```

### Data Augmentation
```python
# Augment training data
X_train = np.load('X_train.npy')
X_augmented, report = system.augment_and_evaluate(X_train, augmentation_factor=0.3)

print(f"Original: {len(X_train)}")
print(f"Augmented: {len(X_augmented)}")
```

### Synthetic Test Set
```python
# Generate synthetic test set
X_synthetic, y_synthetic = system.generate_synthetic_test_set(
    n_normal=100, 
    n_anomalies=50
)

print(f"Synthetic samples: {len(X_synthetic)}")
print(f"Normal: {np.sum(y_synthetic == 0)}")
print(f"Anomalies: {np.sum(y_synthetic == 1)}")
```

### Comprehensive Evaluation
```python
# Evaluate system
eval_report = system.comprehensive_evaluation(
    X_test, 
    reconstruction_errors, 
    y_test
)

print(f"Accuracy: {eval_report['accuracy']:.4f}")
print(f"Precision: {eval_report['precision']:.4f}")
print(f"Recall: {eval_report['recall']:.4f}")
```

---

## XAI Usage

### Generate Explanations
```python
from xai_explainability import TimeSeriesExplainer, generate_explanation_report

# Initialize explainer
explainer = TimeSeriesExplainer(window_size=60)

# Generate explanation
explanation = explainer.explain_prediction(
    sample, 
    reconstruction_error, 
    threshold, 
    y_pred
)

# Generate report
report = generate_explanation_report(
    sample, 
    explanation, 
    reconstruction_error, 
    threshold
)

print(report)
```

### SHAP-like Explanations
```python
from xai_explainability import SHAPLikeExplainer

explainer = SHAPLikeExplainer(model=model, window_size=60)
shap_values = explainer.explain_sample(sample)

print(f"SHAP values: {shap_values}")
```

### LIME-like Explanations
```python
from xai_explainability import LIMELikeExplainer

explainer = LIMELikeExplainer(model=model, window_size=60)
coefficients = explainer.explain_sample(sample)

print(f"Feature importance: {coefficients}")
```

---

## GenAI Usage

### Data Augmentation
```python
from genai_augmentation import augment_training_data

X_train = np.load('X_train.npy')
X_augmented, vae = augment_training_data(X_train, augmentation_factor=0.3)

print(f"Augmented shape: {X_augmented.shape}")
```

### Synthetic Anomaly Generation
```python
from genai_augmentation import generate_synthetic_anomalies

X_anomalies, y_anomalies = generate_synthetic_anomalies(
    vae, 
    n_samples=100, 
    anomaly_types=['spike', 'shift', 'trend']
)

print(f"Synthetic anomalies: {len(X_anomalies)}")
```

---

## Visualization Dashboard Usage

### Display System Summary
```python
from visualization_dashboard import AnomalyVisualizationDashboard

dashboard = AnomalyVisualizationDashboard()

system_info = {
    'window_size': 60,
    'features': 1,
    'threshold': 0.002878,
    'train_samples': 63244,
    'test_samples': 16490
}

dashboard.display_system_summary(system_info)
```

### Display Predictions
```python
dashboard.display_prediction_result(
    prediction=1,
    explanation=explanation,
    reconstruction_error=0.005,
    threshold=0.002878
)
```

### Display Statistics
```python
dashboard.display_batch_statistics(
    predictions, 
    reconstruction_errors, 
    y_true=y_test
)
```

### Export Report
```python
report_data = {
    'System Configuration': system_info,
    'Performance Metrics': metrics
}

dashboard.export_report('report.txt', report_data)
```

---

## Expected Output Examples

### Phase 1: Data Analysis
```
================================================================================
  PHASE 1: DATA ANALYSIS & PREPROCESSING
================================================================================

Loading preprocessed data...

Data Summary:
  Training samples: 63,244
  Test samples: 16,490
  Window size: 60
  Features: 1
  Normal samples in test: 15,811 (95.9%)
  Anomaly samples in test: 679 (4.1%)
```

### Phase 3: Model Evaluation
```
Performance Metrics:
  Accuracy:    0.9518 (95.18%)
  Precision:   0.3343
  Recall:      0.1723
  F1-Score:    0.2274
  Specificity: 0.9853

Confusion Matrix:
  True Negatives:  15,578
  False Positives:    233
  False Negatives:    562
  True Positives:     117
```

### Phase 4: XAI Explanations
```
Normal Sample (Index 100):
  Prediction: NORMAL
  Confidence: 95%
  Error Ratio: 0.15x threshold
  Reason: Normal pattern (very low reconstruction error)
  Top Important Timesteps: [10, 20, 30, 40, 50]
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution:**
```bash
pip install tensorflow keras
```

### Issue: "FileNotFoundError: X_train.npy not found"
**Solution:**
Run data preprocessing first:
```bash
python data_preprocessing.py --file <data_file>
```

### Issue: "Memory Error"
**Solution:**
- Reduce batch size in training
- Process data in smaller chunks
- Use GPU acceleration

### Issue: "Slow Inference"
**Solution:**
- Use batch processing instead of single samples
- Enable GPU acceleration
- Optimize model architecture

---

## Performance Optimization

### Enable GPU Acceleration
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Batch Processing
```python
# Process in batches instead of one-by-one
batch_size = 100
for i in range(0, len(X_test), batch_size):
    batch = X_test[i:i+batch_size]
    results = system.batch_predict_with_explanations(
        batch, 
        reconstruction_errors[i:i+batch_size]
    )
```

### Model Optimization
```python
# Use quantization for faster inference
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

---

## Monitoring and Logging

### Enable Detailed Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Track Predictions
```python
predictions_log = []
for sample, error in zip(X_test, reconstruction_errors):
    result = system.predict_with_explanation(sample, error)
    predictions_log.append({
        'timestamp': datetime.now(),
        'prediction': result['prediction'],
        'error': error,
        'confidence': result['explanation']['confidence']
    })
```

### Generate Metrics Report
```python
import pandas as pd

df = pd.DataFrame(predictions_log)
print(df.describe())
print(f"Anomalies detected: {(df['prediction'] == 1).sum()}")
```

---

## Next Steps

1. **Run Analysis:** `python run_anomaly_detection.py`
2. **Run Full Pipeline:** `python main_pipeline.py`
3. **Review Reports:** Check generated `.txt` files
4. **Customize:** Modify parameters in individual scripts
5. **Deploy:** Integrate into production system

---

## Support

For issues or questions:
1. Check README.md for overview
2. Review PROJECT_COMPLETION_SUMMARY.md for details
3. Check COMPREHENSIVE_REPORT.txt for analysis
4. Examine individual script documentation

---

**Last Updated:** November 2025  
**Version:** 1.0
