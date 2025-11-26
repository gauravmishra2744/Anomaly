"""
Integrated Anomaly Detection System
Combines LSTM, GenAI, and XAI modules into a cohesive system
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
import pickle

from genai_augmentation import TimeSeriesVAE, augment_training_data, generate_synthetic_anomalies
from xai_explainability import TimeSeriesExplainer, generate_explanation_report


class IntegratedAnomalyDetectionSystem:
    """Complete anomaly detection system with LSTM, GenAI, and XAI"""
    
    def __init__(self, model_path='lstm_autoencoder_final.h5', window_size=60):
        self.window_size = window_size
        self.model_path = model_path
        self.model = None
        self.threshold = None
        self.explainer = None
        self.vae = None
        self.scaler = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load pre-trained model and artifacts"""
        print("=" * 70)
        print("LOADING SYSTEM ARTIFACTS")
        print("=" * 70)
        
        # Load model
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"[OK] Model loaded from: {self.model_path}")
        except Exception as e:
            print(f"[ERROR] Could not load TensorFlow model: {e}")
            print("  System will operate in analysis-only mode")
        
        # Load threshold
        if os.path.exists('threshold.npy'):
            self.threshold = np.load('threshold.npy')[0]
            print(f"[OK] Threshold loaded: {self.threshold:.6f}")
        
        # Load scaler
        if os.path.exists('preprocessing_artifacts/scaler.pkl'):
            with open('preprocessing_artifacts/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print("[OK] Scaler loaded")
        
        # Initialize explainer
        self.explainer = TimeSeriesExplainer(model=self.model, window_size=self.window_size)
        print("[OK] XAI Explainer initialized")
        
        # Initialize VAE for augmentation
        self.vae = TimeSeriesVAE(window_size=self.window_size)
        print("[OK] GenAI VAE initialized")
    
    def predict_with_explanation(self, sample, reconstruction_error):
        """
        Make prediction and generate explanation
        
        Args:
            sample: Single time-series sample (window_size, 1)
            reconstruction_error: Pre-computed reconstruction error
        
        Returns:
            result: Dictionary with prediction and explanation
        """
        # Make prediction
        y_pred = 1 if reconstruction_error > self.threshold else 0
        
        # Generate explanation
        explanation = self.explainer.explain_prediction(
            sample, reconstruction_error, self.threshold, y_pred
        )
        
        result = {
            'prediction': y_pred,
            'explanation': explanation,
            'reconstruction_error': reconstruction_error,
            'threshold': self.threshold
        }
        
        return result
    
    def batch_predict_with_explanations(self, X_samples, reconstruction_errors):
        """Predict and explain batch of samples"""
        results = []
        
        for sample, error in zip(X_samples, reconstruction_errors):
            result = self.predict_with_explanation(sample, error)
            results.append(result)
        
        return results
    
    def augment_and_evaluate(self, X_train, augmentation_factor=0.3):
        """
        Augment training data and evaluate impact
        
        Returns:
            X_train_augmented: Augmented training data
            augmentation_report: Report on augmentation impact
        """
        print("\n" + "=" * 70)
        print("DATA AUGMENTATION & IMPACT ANALYSIS")
        print("=" * 70)
        
        # Fit VAE on original data
        self.vae.fit(X_train)
        
        # Generate synthetic samples
        n_synthetic = int(len(X_train) * augmentation_factor)
        X_synthetic = self.vae.generate_normal_samples(n_samples=n_synthetic)
        
        # Combine data
        X_train_augmented = np.vstack([X_train, X_synthetic])
        
        report = {
            'original_samples': len(X_train),
            'synthetic_samples': len(X_synthetic),
            'augmented_samples': len(X_train_augmented),
            'augmentation_ratio': len(X_synthetic) / len(X_train),
            'original_mean': np.mean(X_train),
            'original_std': np.std(X_train),
            'synthetic_mean': np.mean(X_synthetic),
            'synthetic_std': np.std(X_synthetic),
            'augmented_mean': np.mean(X_train_augmented),
            'augmented_std': np.std(X_train_augmented)
        }
        
        print(f"\nAugmentation Report:")
        print(f"  Original samples: {report['original_samples']}")
        print(f"  Synthetic samples: {report['synthetic_samples']}")
        print(f"  Total augmented: {report['augmented_samples']}")
        print(f"  Augmentation ratio: {report['augmentation_ratio']:.1%}")
        
        print(f"\nData Distribution Comparison:")
        print(f"  Original - Mean: {report['original_mean']:.6f}, Std: {report['original_std']:.6f}")
        print(f"  Synthetic - Mean: {report['synthetic_mean']:.6f}, Std: {report['synthetic_std']:.6f}")
        print(f"  Augmented - Mean: {report['augmented_mean']:.6f}, Std: {report['augmented_std']:.6f}")
        
        return X_train_augmented, report
    
    def generate_synthetic_test_set(self, n_normal=100, n_anomalies=50):
        """Generate synthetic test set with known anomalies"""
        print("\n" + "=" * 70)
        print("GENERATING SYNTHETIC TEST SET")
        print("=" * 70)
        
        # Generate normal samples
        X_normal = self.vae.generate_normal_samples(n_samples=n_normal)
        y_normal = np.zeros(n_normal)
        
        # Generate anomalies
        X_anomalies = self.vae.generate_anomaly_samples(n_samples=n_anomalies, anomaly_type='spike')
        y_anomalies = np.ones(n_anomalies)
        
        # Combine
        X_synthetic = np.vstack([X_normal, X_anomalies])
        y_synthetic = np.concatenate([y_normal, y_anomalies])
        
        print(f"\nSynthetic Test Set:")
        print(f"  Normal samples: {n_normal}")
        print(f"  Anomaly samples: {n_anomalies}")
        print(f"  Total: {len(X_synthetic)}")
        
        return X_synthetic, y_synthetic
    
    def comprehensive_evaluation(self, X_test, reconstruction_errors, y_test):
        """
        Comprehensive evaluation with all metrics
        
        Returns:
            evaluation_report: Detailed evaluation metrics
        """
        print("\n" + "=" * 70)
        print("COMPREHENSIVE SYSTEM EVALUATION")
        print("=" * 70)
        
        # Make predictions
        y_pred = (reconstruction_errors > self.threshold).astype(int)
        
        # Compute metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        try:
            roc_auc = roc_auc_score(y_test, reconstruction_errors)
        except:
            roc_auc = 0
        
        report = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'threshold': self.threshold,
            'error_mean': np.mean(reconstruction_errors),
            'error_std': np.std(reconstruction_errors)
        }
        
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"  F1-Score:    {f1:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  ROC-AUC:     {roc_auc:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TN: {tn}, FP: {fp}")
        print(f"  FN: {fn}, TP: {tp}")
        
        return report


def run_integrated_demo():
    """Run complete integrated system demonstration"""
    
    print("\n" + "=" * 70)
    print("INTEGRATED ANOMALY DETECTION SYSTEM - DEMONSTRATION")
    print("=" * 70)
    
    # Initialize system
    system = IntegratedAnomalyDetectionSystem()
    
    # Load data
    print("\nLoading data...")
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    reconstruction_errors = np.load('reconstruction_errors.npy')
    
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Reconstruction errors shape: {reconstruction_errors.shape}")
    
    # 1. Comprehensive Evaluation
    eval_report = system.comprehensive_evaluation(X_test, reconstruction_errors, y_test)
    
    # 2. Generate Explanations for Sample Predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS WITH EXPLANATIONS")
    print("=" * 70)
    
    # Select diverse samples
    normal_idx = np.where(y_test == 0)[0][0]
    anomaly_idx = np.where(y_test == 1)[0][0]
    
    for label, idx in [("Normal", normal_idx), ("Anomaly", anomaly_idx)]:
        result = system.predict_with_explanation(X_test[idx], reconstruction_errors[idx])
        report = generate_explanation_report(
            X_test[idx], result['explanation'], 
            reconstruction_errors[idx], system.threshold
        )
        print(f"\n{label} Sample (Index {idx}):")
        print(report)
    
    # 3. Data Augmentation Impact
    print("\n" + "=" * 70)
    print("DATA AUGMENTATION IMPACT ANALYSIS")
    print("=" * 70)
    
    X_train = np.load('X_train.npy')
    X_train_augmented, aug_report = system.augment_and_evaluate(X_train, augmentation_factor=0.3)
    
    # 4. Synthetic Test Set Generation
    X_synthetic, y_synthetic = system.generate_synthetic_test_set(n_normal=50, n_anomalies=25)
    
    print("\n" + "=" * 70)
    print("INTEGRATED SYSTEM DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    return system, eval_report, aug_report


if __name__ == "__main__":
    system, eval_report, aug_report = run_integrated_demo()
