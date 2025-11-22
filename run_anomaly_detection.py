"""
LSTM Autoencoder Anomaly Detection - Analysis & Visualization Script
This script loads pre-computed reconstruction errors and performs anomaly detection analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
import os

print("=" * 80)
print("LSTM AUTOENCODER ANOMALY DETECTION - RESULTS ANALYSIS")
print("=" * 80)

# ============================================================================
# Step 1: Load Pre-computed Artifacts
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: LOADING PRE-COMPUTED ARTIFACTS")
print("=" * 80)

try:
    print("\nLoading reconstruction errors...")
    reconstruction_errors = np.load('reconstruction_errors.npy')
    print(f"  ✓ Reconstruction errors loaded: shape {reconstruction_errors.shape}")
    
    print("Loading anomaly threshold...")
    threshold = np.load('threshold.npy')[0]
    print(f"  ✓ Threshold loaded: {threshold:.6f}")
    
    print("Loading test data...")
    X_test = np.load('X_test.npy')
    print(f"  ✓ X_test loaded: shape {X_test.shape}")
    
    print("Loading test labels...")
    y_test = np.load('y_test.npy')
    print(f"  ✓ y_test loaded: shape {y_test.shape}")
    
    print("Loading training data...")
    X_train = np.load('X_train.npy')
    print(f"  ✓ X_train loaded: shape {X_train.shape}")
    
except Exception as e:
    print(f"\n✗ Error loading files: {e}")
    exit(1)

# ============================================================================
# Step 2: Analyze Reconstruction Errors
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: RECONSTRUCTION ERROR ANALYSIS")
print("=" * 80)

print(f"\nReconstruction Error Statistics:")
print(f"  Mean:        {np.mean(reconstruction_errors):.6f}")
print(f"  Median:      {np.median(reconstruction_errors):.6f}")
print(f"  Std Dev:     {np.std(reconstruction_errors):.6f}")
print(f"  Min:         {np.min(reconstruction_errors):.6f}")
print(f"  Max:         {np.max(reconstruction_errors):.6f}")
print(f"  25th Percentile: {np.percentile(reconstruction_errors, 25):.6f}")
print(f"  75th Percentile: {np.percentile(reconstruction_errors, 75):.6f}")
print(f"  95th Percentile: {np.percentile(reconstruction_errors, 95):.6f}")
print(f"  99th Percentile: {np.percentile(reconstruction_errors, 99):.6f}")

# Threshold calculation explanation
error_mean = np.mean(reconstruction_errors)
error_std = np.std(reconstruction_errors)
calculated_threshold = error_mean + 2 * error_std

print(f"\nThreshold Calculation (Mean + 2*Std):")
print(f"  Mean + 2*Std = {error_mean:.6f} + 2*{error_std:.6f} = {calculated_threshold:.6f}")
print(f"  Loaded threshold: {threshold:.6f}")

# ============================================================================
# Step 3: Predict Anomalies
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: ANOMALY PREDICTIONS")
print("=" * 80)

# Predict anomalies using the loaded threshold
y_pred = (reconstruction_errors > threshold).astype(int)

print(f"\nPrediction Summary:")
print(f"  Total samples: {len(y_pred)}")
print(f"  Predicted Normal:  {np.sum(y_pred == 0):6d} ({np.sum(y_pred == 0)/len(y_pred)*100:.2f}%)")
print(f"  Predicted Anomaly: {np.sum(y_pred == 1):6d} ({np.sum(y_pred == 1)/len(y_pred)*100:.2f}%)")

print(f"\nActual vs Predicted:")
print(f"  Actual Normal:     {np.sum(y_test == 0):6d} ({np.sum(y_test == 0)/len(y_test)*100:.2f}%)")
print(f"  Actual Anomaly:    {np.sum(y_test == 1):6d} ({np.sum(y_test == 1)/len(y_test)*100:.2f}%)")

# ============================================================================
# Step 4: Detailed Evaluation Metrics
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: DETAILED EVALUATION METRICS")
print("=" * 80)

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\n  True Negatives (TN):  {cm[0, 0]:6d}")
print(f"  False Positives (FP): {cm[0, 1]:6d}")
print(f"  False Negatives (FN): {cm[1, 0]:6d}")
print(f"  True Positives (TP):  {cm[1, 1]:6d}")

# Calculate metrics
tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = recall  # Same as recall

print(f"\nPerformance Metrics:")
print(f"  Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision:    {precision:.4f}")
print(f"  Recall (TPR): {recall:.4f}")
print(f"  Specificity:  {specificity:.4f}")
print(f"  F1-Score:     {f1_score:.4f}")
print(f"  Sensitivity:  {sensitivity:.4f}")

# ROC-AUC Score
try:
    roc_auc = roc_auc_score(y_test, reconstruction_errors)
    print(f"  ROC-AUC:      {roc_auc:.4f}")
except:
    print(f"  ROC-AUC:      Unable to calculate")

# Classification Report
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly'], digits=4))

# ============================================================================
# Step 5: Sample-by-Sample Analysis
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: SAMPLE-BY-SAMPLE ANALYSIS")
print("=" * 80)

print(f"\nFirst 30 Samples:")
print(f"{'Index':<6} {'Error':<12} {'Threshold':<12} {'Prediction':<12} {'Actual':<10} {'Correct':<8}")
print("-" * 70)

for i in range(min(30, len(reconstruction_errors))):
    error = reconstruction_errors[i]
    pred = "ANOMALY" if y_pred[i] == 1 else "NORMAL"
    actual = "ANOMALY" if y_test[i] == 1 else "NORMAL"
    correct = "✓" if y_pred[i] == y_test[i] else "✗"
    
    print(f"{i:<6} {error:<12.6f} {threshold:<12.6f} {pred:<12} {actual:<10} {correct:<8}")

# ============================================================================
# Step 6: Error Distribution Analysis
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: ERROR DISTRIBUTION BY ACTUAL CLASS")
print("=" * 80)

normal_errors = reconstruction_errors[y_test == 0]
anomaly_errors = reconstruction_errors[y_test == 1]

print(f"\nNormal Samples (N={len(normal_errors)}):")
print(f"  Mean:  {np.mean(normal_errors):.6f}")
print(f"  Std:   {np.std(normal_errors):.6f}")
print(f"  Min:   {np.min(normal_errors):.6f}")
print(f"  Max:   {np.max(normal_errors):.6f}")
print(f"  Median: {np.median(normal_errors):.6f}")

print(f"\nAnomaly Samples (N={len(anomaly_errors)}):")
print(f"  Mean:  {np.mean(anomaly_errors):.6f}")
print(f"  Std:   {np.std(anomaly_errors):.6f}")
print(f"  Min:   {np.min(anomaly_errors):.6f}")
print(f"  Max:   {np.max(anomaly_errors):.6f}")
print(f"  Median: {np.median(anomaly_errors):.6f}")

print(f"\nThreshold Analysis:")
print(f"  Normal samples below threshold:  {np.sum(normal_errors <= threshold):6d} / {len(normal_errors)}")
print(f"  Normal samples above threshold:  {np.sum(normal_errors > threshold):6d} / {len(normal_errors)}")
print(f"  Anomaly samples below threshold: {np.sum(anomaly_errors <= threshold):6d} / {len(anomaly_errors)}")
print(f"  Anomaly samples above threshold: {np.sum(anomaly_errors > threshold):6d} / {len(anomaly_errors)}")

# ============================================================================
# Step 7: Training Data Summary
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: TRAINING DATA SUMMARY")
print("=" * 80)

print(f"\nTraining Configuration:")
print(f"  Training samples:  {len(X_train):,}")
print(f"  Test samples:      {len(X_test):,}")
print(f"  Window size:       {X_train.shape[1]}")
print(f"  Features:          {X_train.shape[2]}")

print(f"\nData Characteristics:")
print(f"  X_train shape:     {X_train.shape}")
print(f"  X_train min:       {np.min(X_train):.6f}")
print(f"  X_train max:       {np.max(X_train):.6f}")
print(f"  X_train mean:      {np.mean(X_train):.6f}")

print(f"\n  X_test shape:      {X_test.shape}")
print(f"  X_test min:        {np.min(X_test):.6f}")
print(f"  X_test max:        {np.max(X_test):.6f}")
print(f"  X_test mean:       {np.mean(X_test):.6f}")

# ============================================================================
# Step 8: Reconstruction Error Distribution
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: RECONSTRUCTION ERROR DISTRIBUTION")
print("=" * 80)

# Create bins for histogram
bins = [
    (0, threshold * 0.5, "Very Low (0-50% threshold)"),
    (threshold * 0.5, threshold * 0.8, "Low (50-80% threshold)"),
    (threshold * 0.8, threshold, "Medium (80-100% threshold)"),
    (threshold, threshold * 1.5, "High (100-150% threshold)"),
    (threshold * 1.5, float('inf'), "Very High (>150% threshold)")
]

print(f"\nError Distribution Across Ranges (Threshold = {threshold:.6f}):")
print(f"{'Range':<40} {'Count':>10} {'Percentage':>12}")
print("-" * 65)

for low, high, label in bins:
    if high == float('inf'):
        count = np.sum(reconstruction_errors >= low)
    else:
        count = np.sum((reconstruction_errors >= low) & (reconstruction_errors < high))
    percentage = count / len(reconstruction_errors) * 100
    print(f"{label:<40} {count:>10} {percentage:>11.2f}%")

# ============================================================================
# Step 9: Model Artifacts
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: MODEL ARTIFACTS & FILES")
print("=" * 80)

model_files = {
    'lstm_autoencoder.h5': 'Best model during training',
    'lstm_autoencoder_final.h5': 'Final model after training',
    'reconstruction_errors.npy': 'Pre-computed reconstruction errors',
    'threshold.npy': 'Anomaly detection threshold',
    'X_train.npy': 'Training data (preprocessed)',
    'X_test.npy': 'Test data (preprocessed)',
    'y_test.npy': 'Test labels (ground truth)',
}

print(f"\nGenerated/Available Files:")
for filename, description in model_files.items():
    if os.path.exists(filename):
        size = os.path.getsize(filename) / (1024 * 1024)  # Size in MB
        print(f"  ✓ {filename:<30} ({size:>7.2f} MB) - {description}")
    else:
        print(f"  ✗ {filename:<30} (MISSING) - {description}")

# ============================================================================
# Step 10: Summary & Conclusions
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: SUMMARY & CONCLUSIONS")
print("=" * 80)

def interpret_performance():
    """Provide interpretation of model performance"""
    if accuracy > 0.95:
        return "EXCELLENT - Model is highly accurate"
    elif accuracy > 0.90:
        return "VERY GOOD - Model performs very well"
    elif accuracy > 0.85:
        return "GOOD - Model is reasonably accurate"
    elif accuracy > 0.80:
        return "FAIR - Model needs improvement"
    else:
        return "POOR - Model requires significant improvement"

def recall_desc(recall):
    if recall > 0.95:
        return "Excellent - catches almost all anomalies"
    elif recall > 0.80:
        return "Very good - catches most anomalies"
    elif recall > 0.60:
        return "Good - catches many anomalies"
    else:
        return "Fair - misses many anomalies"

def specificity_desc(specificity):
    if specificity > 0.95:
        return "Excellent - very few false positives"
    elif specificity > 0.80:
        return "Very good - few false positives"
    elif specificity > 0.60:
        return "Good - moderate false positives"
    else:
        return "Fair - many false positives"

print(f"\nModel Performance Summary:")
print(f"  • Overall Accuracy:  {accuracy*100:.2f}%")
print(f"  • Sensitivity (Recall): {recall*100:.2f}% - {recall_desc(recall)}")
print(f"  • Specificity: {specificity*100:.2f}% - {specificity_desc(specificity)}")
print(f"  • Precision: {precision*100:.2f}%")
print(f"  • F1-Score: {f1_score:.4f}")

print(f"\n  Interpretation: {interpret_performance()}")

print(f"\nKey Findings:")
print(f"  1. Threshold set at: {threshold:.6f} (Mean + 2*Std)")
print(f"  2. Correctly identified: {tp + tn} / {len(y_test)} samples")
print(f"  3. Misclassified: {fp + fn} samples")
print(f"  4. Most critical metric (Recall): {recall*100:.2f}%")
if fp > 0:
    print(f"  5. False positive rate: {fp/(fp+tn)*100:.2f}%")
if fn > 0:
    print(f"  6. False negative rate: {fn/(fn+tp)*100:.2f}%")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nThis project successfully demonstrates:")
print("  ✓ Time-series data preprocessing with sliding windows")
print("  ✓ LSTM Autoencoder training for anomaly detection")
print("  ✓ Reconstruction error computation")
print("  ✓ Threshold-based anomaly classification")
print("  ✓ Comprehensive performance evaluation")
print("\n" + "=" * 80)
