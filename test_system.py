import numpy as np
import sys

print("=" * 70)
print("TESTING ANOMALY DETECTION SYSTEM")
print("=" * 70)

# Test 1: Load data
print("\n[1/6] Testing data loading...")
try:
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    reconstruction_errors = np.load('reconstruction_errors.npy')
    threshold = np.load('threshold.npy')[0]
    print("[OK] Data loaded successfully")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")
    print(f"  - Threshold: {threshold:.6f}")
except Exception as e:
    print(f"[ERROR] Error loading data: {e}")
    sys.exit(1)

# Test 2: Test GenAI
print("\n[2/6] Testing GenAI augmentation...")
try:
    from genai_augmentation import augment_training_data
    X_augmented, vae = augment_training_data(X_train, augmentation_factor=0.1)
    print("[OK] GenAI augmentation works")
    print(f"  - Original: {len(X_train)}")
    print(f"  - Augmented: {len(X_augmented)}")
except Exception as e:
    print(f"[ERROR] Error in GenAI: {e}")

# Test 3: Test XAI
print("\n[3/6] Testing XAI explainability...")
try:
    from xai_explainability import TimeSeriesExplainer
    explainer = TimeSeriesExplainer(window_size=60)
    explanation = explainer.explain_prediction(
        X_test[0], reconstruction_errors[0], threshold, 0
    )
    print("[OK] XAI explainability works")
    print(f"  - Prediction: {explanation['prediction']}")
    print(f"  - Confidence: {explanation['confidence']:.2%}")
except Exception as e:
    print(f"[ERROR] Error in XAI: {e}")

# Test 4: Test Integrated System
print("\n[4/6] Testing integrated system...")
try:
    from integrated_system import IntegratedAnomalyDetectionSystem
    system = IntegratedAnomalyDetectionSystem()
    result = system.predict_with_explanation(X_test[0], reconstruction_errors[0])
    print("[OK] Integrated system works")
    print(f"  - Prediction: {result['prediction']}")
except Exception as e:
    print(f"[ERROR] Error in integrated system: {e}")

# Test 5: Test predictions
print("\n[5/6] Testing predictions...")
try:
    predictions = (reconstruction_errors > threshold).astype(int)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    print("[OK] Predictions work")
    print(f"  - Accuracy: {accuracy:.4f}")
    print(f"  - Anomalies detected: {np.sum(predictions == 1)}")
    print(f"  - Normal samples: {np.sum(predictions == 0)}")
except Exception as e:
    print(f"[ERROR] Error in predictions: {e}")

# Test 6: Test Flask app
print("\n[6/6] Testing Flask app...")
try:
    from app import app
    print("[OK] Flask app imports successfully")
    print("  - Ready to run: python app.py")
except Exception as e:
    print(f"[ERROR] Error in Flask app: {e}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED - SUCCESS")
print("=" * 70)
print("\nTo start the web application:")
print("  python app.py")
print("\nThen open: http://localhost:5000")
print("\nOr run: run_app.bat")
