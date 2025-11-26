"""
Interactive Visualization Dashboard for Anomaly Detection
Provides real-time visualization and analysis interface
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime


class AnomalyVisualizationDashboard:
    """Dashboard for visualizing anomaly detection results"""
    
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.results_history = []
    
    def print_header(self, title):
        """Print formatted header"""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)
    
    def print_section(self, title):
        """Print formatted section"""
        print(f"\n{title}")
        print("-" * 80)
    
    def display_time_series(self, sample, title="Time Series Sample", max_width=60):
        """Display time series as ASCII plot"""
        self.print_section(title)
        
        # Normalize for display
        sample_flat = sample.flatten()
        min_val = np.min(sample_flat)
        max_val = np.max(sample_flat)
        
        if max_val == min_val:
            normalized = np.zeros_like(sample_flat)
        else:
            normalized = (sample_flat - min_val) / (max_val - min_val)
        
        # Create ASCII plot
        height = 10
        plot = [[' ' for _ in range(len(sample_flat))] for _ in range(height)]
        
        for i, val in enumerate(normalized):
            row = int((1 - val) * (height - 1))
            if 0 <= row < height:
                plot[row][i] = '█'
        
        # Print plot
        for row in plot:
            print(''.join(row))
        
        # Print scale
        print(f"Range: [{min_val:.4f}, {max_val:.4f}]")
    
    def display_prediction_result(self, prediction, explanation, reconstruction_error, threshold):
        """Display prediction result with explanation"""
        self.print_section("PREDICTION RESULT")
        
        # Prediction status
        status = "🔴 ANOMALY DETECTED" if prediction == 1 else "🟢 NORMAL"
        print(f"Status: {status}")
        print(f"Confidence: {explanation['confidence']:.1%}")
        
        # Error analysis
        print(f"\nReconstruction Error Analysis:")
        print(f"  Error Value: {reconstruction_error:.6f}")
        print(f"  Threshold:   {threshold:.6f}")
        print(f"  Ratio:       {explanation['error_ratio']:.2f}x")
        
        # Reason
        print(f"\nReason: {explanation['reason']}")
        
        # Important timesteps
        print(f"\nMost Important Timesteps:")
        for rank, (idx, score) in enumerate(zip(explanation['top_important_timesteps'],
                                                 explanation['top_importance_scores']), 1):
            bar = '█' * int(score * 30)
            print(f"  {rank}. T{idx:2d}: {bar} {score:.4f}")
    
    def display_batch_statistics(self, predictions, reconstruction_errors, y_true=None):
        """Display statistics for batch predictions"""
        self.print_section("BATCH STATISTICS")
        
        n_anomalies = np.sum(predictions == 1)
        n_normal = np.sum(predictions == 0)
        
        print(f"Total Samples: {len(predictions)}")
        print(f"  Normal:  {n_normal:6d} ({n_normal/len(predictions)*100:5.1f}%)")
        print(f"  Anomaly: {n_anomalies:6d} ({n_anomalies/len(predictions)*100:5.1f}%)")
        
        print(f"\nReconstruction Error Statistics:")
        print(f"  Mean:   {np.mean(reconstruction_errors):.6f}")
        print(f"  Median: {np.median(reconstruction_errors):.6f}")
        print(f"  Std:    {np.std(reconstruction_errors):.6f}")
        print(f"  Min:    {np.min(reconstruction_errors):.6f}")
        print(f"  Max:    {np.max(reconstruction_errors):.6f}")
        
        if y_true is not None:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, predictions)
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"\nPerformance Metrics:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
    
    def display_error_distribution(self, reconstruction_errors, threshold, y_true=None):
        """Display error distribution analysis"""
        self.print_section("ERROR DISTRIBUTION ANALYSIS")
        
        # Categorize errors
        very_low = np.sum(reconstruction_errors < threshold * 0.5)
        low = np.sum((reconstruction_errors >= threshold * 0.5) & (reconstruction_errors < threshold * 0.8))
        medium = np.sum((reconstruction_errors >= threshold * 0.8) & (reconstruction_errors < threshold))
        high = np.sum((reconstruction_errors >= threshold) & (reconstruction_errors < threshold * 1.5))
        very_high = np.sum(reconstruction_errors >= threshold * 1.5)
        
        categories = [
            ("Very Low (0-50%)", very_low),
            ("Low (50-80%)", low),
            ("Medium (80-100%)", medium),
            ("High (100-150%)", high),
            ("Very High (>150%)", very_high)
        ]
        
        total = len(reconstruction_errors)
        
        for label, count in categories:
            pct = count / total * 100
            bar = '█' * int(pct / 2)
            print(f"{label:20s}: {count:6d} ({pct:5.1f}%) {bar}")
        
        if y_true is not None:
            print(f"\nError Distribution by Class:")
            normal_errors = reconstruction_errors[y_true == 0]
            anomaly_errors = reconstruction_errors[y_true == 1]
            
            print(f"  Normal samples:")
            print(f"    Mean: {np.mean(normal_errors):.6f}, Std: {np.std(normal_errors):.6f}")
            print(f"  Anomaly samples:")
            print(f"    Mean: {np.mean(anomaly_errors):.6f}, Std: {np.std(anomaly_errors):.6f}")
    
    def display_system_summary(self, system_info):
        """Display system configuration summary"""
        self.print_header("SYSTEM CONFIGURATION SUMMARY")
        
        print(f"\nModel Configuration:")
        print(f"  Architecture: LSTM Autoencoder")
        print(f"  Window Size: {system_info.get('window_size', 'N/A')}")
        print(f"  Encoder: LSTM(64, relu)")
        print(f"  Decoder: LSTM(64, relu)")
        print(f"  Output: TimeDistributed Dense")
        
        print(f"\nThreshold Configuration:")
        print(f"  Method: Mean + 2*Std")
        print(f"  Threshold: {system_info.get('threshold', 'N/A'):.6f}")
        
        print(f"\nData Configuration:")
        print(f"  Training Samples: {system_info.get('train_samples', 'N/A')}")
        print(f"  Test Samples: {system_info.get('test_samples', 'N/A')}")
        print(f"  Features: {system_info.get('features', 'N/A')}")
        
        print(f"\nAugmentation Configuration:")
        print(f"  Method: VAE-based synthetic generation")
        print(f"  Anomaly Types: Spike, Shift, Trend")
        
        print(f"\nExplainability Configuration:")
        print(f"  Methods: Feature Importance, SHAP-like, LIME-like")
        print(f"  Explanation Types: Timestep importance, Error ratio analysis")
    
    def display_real_time_monitoring(self, samples_batch, errors_batch, predictions_batch, 
                                     threshold, window_size=5):
        """Display real-time monitoring interface"""
        self.print_header("REAL-TIME MONITORING INTERFACE")
        
        print(f"\nMonitoring {len(samples_batch)} samples...")
        print(f"Threshold: {threshold:.6f}\n")
        
        print(f"{'Index':<6} {'Error':<12} {'Status':<12} {'Confidence':<12} {'Reason':<30}")
        print("-" * 80)
        
        for i, (sample, error, pred) in enumerate(zip(samples_batch, errors_batch, predictions_batch)):
            status = "🔴 ANOMALY" if pred == 1 else "🟢 NORMAL"
            confidence = abs(error - threshold) / (threshold + 1e-8)
            confidence = min(confidence, 1.0)
            
            if pred == 1:
                reason = f"Error {error/threshold:.1f}x threshold"
            else:
                reason = "Within normal range"
            
            print(f"{i:<6} {error:<12.6f} {status:<12} {confidence:<12.1%} {reason:<30}")
    
    def export_report(self, filename, report_data):
        """Export analysis report to file"""
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ANOMALY DETECTION SYSTEM - ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for section, data in report_data.items():
                f.write(f"\n{section}\n")
                f.write("-" * 80 + "\n")
                
                if isinstance(data, dict):
                    for key, value in data.items():
                        f.write(f"  {key}: {value}\n")
                elif isinstance(data, list):
                    for item in data:
                        f.write(f"  {item}\n")
                else:
                    f.write(f"  {data}\n")
        
        print(f"\nReport exported to: {filename}")


def run_interactive_dashboard():
    """Run interactive dashboard demonstration"""
    
    dashboard = AnomalyVisualizationDashboard()
    
    # Load data
    print("Loading data for dashboard...")
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    reconstruction_errors = np.load('reconstruction_errors.npy')
    threshold = np.load('threshold.npy')[0]
    
    # System info
    system_info = {
        'window_size': X_test.shape[1],
        'features': X_test.shape[2],
        'threshold': threshold,
        'train_samples': len(np.load('X_train.npy')),
        'test_samples': len(X_test)
    }
    
    # Display system summary
    dashboard.display_system_summary(system_info)
    
    # Display batch statistics
    predictions = (reconstruction_errors > threshold).astype(int)
    dashboard.display_batch_statistics(predictions, reconstruction_errors, y_test)
    
    # Display error distribution
    dashboard.display_error_distribution(reconstruction_errors, threshold, y_test)
    
    # Display sample predictions
    normal_idx = np.where(y_test == 0)[0][0]
    anomaly_idx = np.where(y_test == 1)[0][0]
    
    for label, idx in [("Normal", normal_idx), ("Anomaly", anomaly_idx)]:
        dashboard.print_header(f"{label.upper()} SAMPLE ANALYSIS (Index {idx})")
        dashboard.display_time_series(X_test[idx], f"{label} Time Series")
        
        # Create mock explanation
        explanation = {
            'confidence': 0.85 if label == "Anomaly" else 0.95,
            'error_ratio': reconstruction_errors[idx] / threshold,
            'reason': f"Sample shows {'anomalous' if label == 'Anomaly' else 'normal'} pattern",
            'top_important_timesteps': np.array([10, 20, 30, 40, 50]),
            'top_importance_scores': np.array([0.25, 0.20, 0.18, 0.15, 0.12])
        }
        
        dashboard.display_prediction_result(
            1 if label == "Anomaly" else 0,
            explanation,
            reconstruction_errors[idx],
            threshold
        )
    
    # Display real-time monitoring
    sample_indices = np.random.choice(len(X_test), 10, replace=False)
    dashboard.display_real_time_monitoring(
        X_test[sample_indices],
        reconstruction_errors[sample_indices],
        predictions[sample_indices],
        threshold
    )
    
    # Export report
    report_data = {
        'System Configuration': system_info,
        'Performance Metrics': {
            'Total Samples': len(predictions),
            'Anomalies Detected': np.sum(predictions == 1),
            'Normal Samples': np.sum(predictions == 0)
        }
    }
    
    dashboard.export_report('anomaly_detection_report.txt', report_data)
    
    print("\n" + "=" * 80)
    print("DASHBOARD DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_interactive_dashboard()
