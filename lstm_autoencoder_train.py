"""
LSTM Autoencoder Training Pipeline for Time-Series Anomaly Detection

This script:
1. Loads preprocessed data (X_train, X_test, y_test, scaler)
2. Builds and trains an LSTM Autoencoder
3. Computes reconstruction errors and anomaly threshold
4. Evaluates performance (if labels available)
5. Saves model and artifacts
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# Step 1: Load Data
# ============================================================================

print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

# File paths
X_train_path = "X_train.npy"
X_test_path = "X_test.npy"
y_test_path = "y_test.npy"
scaler_path = "preprocessing_artifacts/scaler.pkl"
config_path = "preprocessing_artifacts/config.pkl"

# Load training data
print(f"\nLoading {X_train_path}...")
X_train = np.load(X_train_path)
print(f"  X_train shape: {X_train.shape}")

# Load test data
print(f"\nLoading {X_test_path}...")
X_test = np.load(X_test_path)
print(f"  X_test shape: {X_test.shape}")

# Load test labels (if available)
y_test = None
if os.path.exists(y_test_path):
    print(f"\nLoading {y_test_path}...")
    y_test = np.load(y_test_path)
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Anomaly ratio: {np.sum(y_test == 1) / len(y_test):.4f}")
else:
    print(f"\n{y_test_path} not found. Proceeding without labels for evaluation.")

# Load scaler (for reference, not needed for training)
if os.path.exists(scaler_path):
    print(f"\nLoading {scaler_path}...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("  Scaler loaded successfully.")

# Load config to get window_size and n_features
window_size = X_train.shape[1]
n_features = X_train.shape[2]

print(f"\nData Summary:")
print(f"  Window size: {window_size}")
print(f"  Number of features: {n_features}")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

# ============================================================================
# Step 2: Build LSTM Autoencoder Model
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: BUILDING LSTM AUTOENCODER")
print("=" * 70)

# Input layer
input_layer = keras.Input(shape=(window_size, n_features))

# Encoder: LSTM(64, relu)
encoder = LSTM(64, activation='relu', name='encoder')(input_layer)

# Repeat vector to match window_size for decoder
repeat_vector = RepeatVector(window_size, name='repeat_vector')(encoder)

# Decoder: LSTM(64, relu, return_sequences=True)
decoder = LSTM(64, activation='relu', return_sequences=True, name='decoder')(repeat_vector)

# Output layer: TimeDistributed Dense to reconstruct original features
output_layer = TimeDistributed(Dense(n_features), name='output')(decoder)

# Create model
model = Model(inputs=input_layer, outputs=output_layer, name='lstm_autoencoder')

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Display model architecture
print("\nModel Architecture:")
model.summary()

# Count parameters
total_params = model.count_params()
print(f"\nTotal parameters: {total_params:,}")

# ============================================================================
# Step 3: Train Model
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: TRAINING MODEL")
print("=" * 70)

# Training configuration
epochs = 25
batch_size = 32
validation_split = 0.1

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'lstm_autoencoder.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

print(f"\nTraining Configuration:")
print(f"  Epochs: {epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Validation split: {validation_split}")
print(f"  Callbacks: EarlyStopping, ModelCheckpoint")

# Train the model
print("\nStarting training...")
history = model.fit(
    X_train,  # Input
    X_train,  # Target (autoencoder reconstructs input)
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    callbacks=callbacks,
    verbose=1
)

# Save final model
print("\nSaving final model...")
model.save('lstm_autoencoder_final.h5')
print("  Model saved to: lstm_autoencoder_final.h5")

# Training summary
print("\nTraining Summary:")
print(f"  Best validation loss: {min(history.history['val_loss']):.6f}")
print(f"  Final training loss: {history.history['loss'][-1]:.6f}")
print(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")
print(f"  Training epochs: {len(history.history['loss'])}")

# ============================================================================
# Step 4: Reconstruction and Error Computation
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: COMPUTING RECONSTRUCTION ERRORS")
print("=" * 70)

# Reconstruct test data
print("\nReconstructing X_test...")
X_test_reconstructed = model.predict(X_test, verbose=1)

# Compute reconstruction error per window (MSE for each window)
print("\nComputing reconstruction errors...")
# Calculate MSE for each window: mean over time steps and features
reconstruction_errors = np.mean(np.square(X_test - X_test_reconstructed), axis=(1, 2))

print(f"  Reconstruction errors shape: {reconstruction_errors.shape}")
print(f"  Error statistics:")
print(f"    Mean: {np.mean(reconstruction_errors):.6f}")
print(f"    Std:  {np.std(reconstruction_errors):.6f}")
print(f"    Min:  {np.min(reconstruction_errors):.6f}")
print(f"    Max:  {np.max(reconstruction_errors):.6f}")
print(f"    Median: {np.median(reconstruction_errors):.6f}")

# Display sample errors
print(f"\n  Sample errors (first 10):")
for i in range(min(10, len(reconstruction_errors))):
    print(f"    Window {i}: {reconstruction_errors[i]:.6f}")

# Save reconstruction errors
print("\nSaving reconstruction errors...")
np.save('reconstruction_errors.npy', reconstruction_errors)
print("  Saved to: reconstruction_errors.npy")

# ============================================================================
# Step 5: Compute Anomaly Threshold
# ============================================================================

print("\n" + "=" * 70)
print("STEP 5: COMPUTING ANOMALY THRESHOLD")
print("=" * 70)

# Compute threshold: mean + 2 * std
error_mean = np.mean(reconstruction_errors)
error_std = np.std(reconstruction_errors)
threshold = error_mean + 2 * error_std

print(f"\nThreshold Calculation:")
print(f"  Mean error: {error_mean:.6f}")
print(f"  Std error:  {error_std:.6f}")
print(f"  Threshold:  {threshold:.6f} (mean + 2*std)")

# Save threshold
print("\nSaving threshold...")
np.save('threshold.npy', np.array([threshold]))
print("  Saved to: threshold.npy")

# ============================================================================
# Step 6: Predict Anomalies
# ============================================================================

print("\n" + "=" * 70)
print("STEP 6: PREDICTING ANOMALIES")
print("=" * 70)

# Predict anomalies: error > threshold
pred_labels = (reconstruction_errors > threshold).astype(int)

print(f"\nPrediction Summary:")
print(f"  Total predictions: {len(pred_labels)}")
print(f"  Predicted normal:   {np.sum(pred_labels == 0)} ({np.sum(pred_labels == 0)/len(pred_labels)*100:.2f}%)")
print(f"  Predicted anomaly:  {np.sum(pred_labels == 1)} ({np.sum(pred_labels == 1)/len(pred_labels)*100:.2f}%)")

print(f"\n  Sample predictions (first 20):")
for i in range(min(20, len(pred_labels))):
    status = "ANOMALY" if pred_labels[i] == 1 else "NORMAL"
    print(f"    Window {i:4d}: Error={reconstruction_errors[i]:.6f}, Prediction={status}")

# ============================================================================
# Step 7: Evaluation (if y_test available)
# ============================================================================

if y_test is not None:
    print("\n" + "=" * 70)
    print("STEP 7: EVALUATION")
    print("=" * 70)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, pred_labels, 
                                target_names=['Normal', 'Anomaly'],
                                digits=4))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, pred_labels)
    print(cm)
    print(f"\n  True Negatives (TN):  {cm[0, 0]}")
    print(f"  False Positives (FP): {cm[0, 1]}")
    print(f"  False Negatives (FN): {cm[1, 0]}")
    print(f"  True Positives (TP):  {cm[1, 1]}")
    
    # Additional metrics
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1_score:.4f}")
    
    # Compare with actual labels
    print(f"\nActual vs Predicted:")
    print(f"  Actual normal:   {np.sum(y_test == 0)} ({np.sum(y_test == 0)/len(y_test)*100:.2f}%)")
    print(f"  Actual anomaly:  {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.2f}%)")
    print(f"  Predicted normal:   {np.sum(pred_labels == 0)} ({np.sum(pred_labels == 0)/len(pred_labels)*100:.2f}%)")
    print(f"  Predicted anomaly:  {np.sum(pred_labels == 1)} ({np.sum(pred_labels == 1)/len(pred_labels)*100:.2f}%)")
else:
    print("\n" + "=" * 70)
    print("STEP 7: EVALUATION SKIPPED (no y_test available)")
    print("=" * 70)

# ============================================================================
# Step 8: Final Summary
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING PIPELINE COMPLETE")
print("=" * 70)

print("\nSaved Files:")
print("  ✓ lstm_autoencoder.h5 (best model during training)")
print("  ✓ lstm_autoencoder_final.h5 (final model)")
print("  ✓ reconstruction_errors.npy")
print("  ✓ threshold.npy")

print("\nModel Configuration:")
print(f"  Architecture: LSTM(64) -> RepeatVector -> LSTM(64) -> Dense")
print(f"  Input shape: ({window_size}, {n_features})")
print(f"  Total parameters: {total_params:,}")

print("\nAnomaly Detection Summary:")
print(f"  Threshold: {threshold:.6f}")
print(f"  Predicted anomalies: {np.sum(pred_labels == 1)} / {len(pred_labels)}")

if y_test is not None:
    print(f"  Actual anomalies: {np.sum(y_test == 1)} / {len(y_test)}")

print("\n" + "=" * 70)
print("Pipeline completed successfully!")
print("=" * 70)
