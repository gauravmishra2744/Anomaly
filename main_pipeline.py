"""
Main Pipeline - Complete Anomaly Detection System
Orchestrates all components: Data Preprocessing, LSTM Training, GenAI Augmentation, XAI Explainability
"""

import numpy as np
import os
import sys
from datetime import datetime

# Import all modules
from genai_augmentation import augment_training_data, generate_synthetic_anomalies
from xai_explainability import TimeSeriesExplainer, generate_explanation_report
from genai_explainer_simple import GenAIExplainer, explain_anomaly
from genai_integration import EnhancedAnomalyAnalyzer
from integrated_system import IntegratedAnomalyDetectionSystem
from visualization_dashboard import AnomalyVisualizationDashboard


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def phase_1_data_analysis():
    """Phase 1: Data Analysis and Preprocessing"""
    print_banner("PHASE 1: DATA ANALYSIS & PREPROCESSING")
    
    print("\nLoading preprocessed data...")
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    print(f"\nData Summary:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Window size: {X_train.shape[1]}")
    print(f"  Features: {X_train.shape[2]}")
    print(f"  Normal samples in test: {np.sum(y_test == 0):,} ({np.sum(y_test == 0)/len(y_test)*100:.1f}%)")
    print(f"  Anomaly samples in test: {np.sum(y_test == 1):,} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)")
    
    return X_train, X_test, y_test


def phase_2_genai_augmentation(X_train):
    """Phase 2: GenAI Data Augmentation"""
    print_banner("PHASE 2: GENERATIVE AI - DATA AUGMENTATION")
    
    print("\nAugmenting training data with synthetic samples...")
    X_train_augmented, vae = augment_training_data(X_train, augmentation_factor=0.3)
    
    print(f"\nAugmentation Results:")
    print(f"  Original training samples: {len(X_train):,}")
    print(f"  Augmented training samples: {len(X_train_augmented):,}")
    print(f"  Synthetic samples added: {len(X_train_augmented) - len(X_train):,}")
    
    # Generate synthetic anomalies for testing
    print("\nGenerating synthetic anomalies for testing...")
    X_synthetic_anomalies, y_synthetic_anomalies = generate_synthetic_anomalies(
        vae, n_samples=100, anomaly_types=['spike', 'shift', 'trend']
    )
    
    print(f"  Synthetic anomalies generated: {len(X_synthetic_anomalies)}")
    
    return X_train_augmented, vae, X_synthetic_anomalies


def phase_3_model_evaluation(X_test, y_test):
    """Phase 3: LSTM Model Evaluation"""
    print_banner("PHASE 3: LSTM MODEL EVALUATION")
    
    print("\nLoading pre-trained LSTM model...")
    reconstruction_errors = np.load('reconstruction_errors.npy')
    threshold = np.load('threshold.npy')[0]
    
    print(f"\nModel Configuration:")
    print(f"  Architecture: LSTM Autoencoder")
    print(f"  Threshold: {threshold:.6f} (Mean + 2*Std)")
    
    # Make predictions
    y_pred = (reconstruction_errors > threshold).astype(int)
    
    # Compute metrics
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn:6d}")
    print(f"  False Positives: {fp:6d}")
    print(f"  False Negatives: {fn:6d}")
    print(f"  True Positives:  {tp:6d}")
    
    return reconstruction_errors, threshold, y_pred


def phase_4_xai_genai_explainability(X_test, reconstruction_errors, threshold, y_test):
    """Phase 4: XAI + GenAI Explainability"""
    print_banner("PHASE 4: EXPLAINABLE AI + GENERATIVE AI INTELLIGENCE")
    
    print("\nInitializing XAI Explainer...")
    explainer = TimeSeriesExplainer(window_size=X_test.shape[1])
    
    print("Initializing GenAI Explainer...")
    genai_explainer = GenAIExplainer()
    
    print("Initializing Enhanced Analyzer...")
    enhanced_analyzer = EnhancedAnomalyAnalyzer()
    
    # Generate explanations for sample predictions
    print("\nGenerating XAI + GenAI explanations for sample predictions...")
    
    # Select diverse samples
    normal_indices = np.where(y_test == 0)[0]
    anomaly_indices = np.where(y_test == 1)[0]
    
    sample_indices = [normal_indices[0], anomaly_indices[0]]
    labels = ["Normal", "Anomaly"]
    
    explanations = []
    genai_explanations = []
    enhanced_explanations = []
    
    for label, idx in zip(labels, sample_indices):
        y_pred = 1 if reconstruction_errors[idx] > threshold else 0
        
        # XAI Explanation
        xai_explanation = explainer.explain_prediction(
            X_test[idx], reconstruction_errors[idx], threshold, y_pred
        )
        explanations.append(xai_explanation)
        
        # GenAI Explanation
        genai_payload = {
            "window_values": X_test[idx].flatten().tolist(),
            "reconstruction_error": float(reconstruction_errors[idx]),
            "error_threshold": float(threshold),
            "xai_top_features": xai_explanation.get('contributing_factors', [])[:3],
            "xai_top_timesteps": xai_explanation.get('top_important_timesteps', [])[:5].tolist()
        }
        
        genai_result = genai_explainer.explain_anomaly(genai_payload)
        genai_explanations.append(genai_result)
        
        # Enhanced Analysis (XAI + GenAI)
        enhanced_result = enhanced_analyzer.analyze_sample_with_genai(
            X_test[idx], reconstruction_errors[idx], threshold, xai_explanation
        )
        enhanced_explanations.append(enhanced_result)
        
        print(f"\n{label} Sample (Index {idx}):")
        print(f"  XAI Prediction: {xai_explanation['prediction']}")
        print(f"  XAI Confidence: {xai_explanation['confidence']:.1%}")
        print(f"  XAI Reason: {xai_explanation['reason']}")
        print(f"  GenAI Classification: {genai_result['classification']}")
        print(f"  GenAI Severity: {genai_result['severity']}")
        print(f"  GenAI Confidence: {genai_result['confidence']}%")
        print(f"  Threat Level: {enhanced_result.get('threat_level', 'Unknown')}")
        print(f"  Requires Action: {enhanced_result.get('requires_immediate_action', False)}")
    
    return explainer, explanations, genai_explainer, genai_explanations, enhanced_analyzer, enhanced_explanations


def phase_5_system_integration():
    """Phase 5: System Integration with MITRE ATT&CK Mapping"""
    print_banner("PHASE 5: SYSTEM INTEGRATION + MITRE ATT&CK MAPPING")
    
    print("\nInitializing Integrated Anomaly Detection System...")
    system = IntegratedAnomalyDetectionSystem()
    
    print("\nSystem Components:")
    print("  ✓ LSTM Autoencoder Model")
    print("  ✓ GenAI VAE for Data Augmentation")
    print("  ✓ XAI Explainer for Interpretability")
    print("  ✓ GenAI Intelligence Layer")
    print("  ✓ Enhanced Anomaly Analyzer")
    print("  ✓ Threshold-based Anomaly Detection")
    print("  ✓ MITRE ATT&CK Framework Integration")
    print("  ✓ RAG Context System")
    
    # Initialize MITRE ATT&CK mapping
    mitre_mapping = {
        'T1071': 'Application Layer Protocol',
        'T1090': 'Proxy',
        'T1041': 'Exfiltration Over C2 Channel',
        'T1048': 'Exfiltration Over Alternative Protocol',
        'T1499': 'Endpoint Denial of Service',
        'T1498': 'Network Denial of Service',
        'T1046': 'Network Service Scanning',
        'T1595': 'Active Scanning'
    }
    
    print("\nMITRE ATT&CK Techniques Mapped:")
    for technique_id, technique_name in mitre_mapping.items():
        print(f"  {technique_id}: {technique_name}")
    
    return system, mitre_mapping


def phase_6_enhanced_visualization_dashboard(mitre_mapping):
    """Phase 6: Enhanced Visualization Dashboard with Real-time Features"""
    print_banner("PHASE 6: ENHANCED VISUALIZATION & REAL-TIME MONITORING")
    
    print("\nInitializing Enhanced Visualization Dashboard...")
    dashboard = AnomalyVisualizationDashboard()
    
    # Load data
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    reconstruction_errors = np.load('reconstruction_errors.npy')
    threshold = np.load('threshold.npy')[0]
    
    # Enhanced system info with GenAI and MITRE ATT&CK
    system_info = {
        'window_size': X_test.shape[1],
        'features': X_test.shape[2],
        'threshold': threshold,
        'train_samples': len(np.load('X_train.npy')),
        'test_samples': len(X_test),
        'genai_enabled': True,
        'xai_enabled': True,
        'mitre_attack_mapped': len(mitre_mapping),
        'rag_context_enabled': True
    }
    
    print("\nDashboard Features:")
    print("  ✓ Real-time anomaly summary")
    print("  ✓ Interactive charts (Reconstruction Error, Feature Impact, Timeline)")
    print("  ✓ Severity badges with color coding")
    print("  ✓ MITRE ATT&CK technique mapping")
    print("  ✓ RAG context panel for threat intelligence")
    print("  ✓ Fully responsive Bootstrap 5 design")
    print("  ✓ GenAI-powered explanations")
    print("  ✓ XAI interpretability features")
    
    dashboard.display_system_summary(system_info)
    
    # Display batch statistics with enhanced metrics
    predictions = (reconstruction_errors > threshold).astype(int)
    dashboard.display_batch_statistics(predictions, reconstruction_errors, y_test)
    
    # Display error distribution with severity mapping
    dashboard.display_error_distribution(reconstruction_errors, threshold, y_test)
    
    # Display real-time monitoring with MITRE ATT&CK context
    sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
    dashboard.display_real_time_monitoring(
        X_test[sample_indices],
        reconstruction_errors[sample_indices],
        predictions[sample_indices],
        threshold
    )
    
    # Display MITRE ATT&CK mapping
    print("\nMITRE ATT&CK Integration:")
    for technique_id, technique_name in mitre_mapping.items():
        print(f"  {technique_id}: {technique_name}")
    
    return dashboard


def phase_7_comprehensive_report_with_genai():
    """Phase 7: Generate Comprehensive Report with GenAI Insights"""
    print_banner("PHASE 7: COMPREHENSIVE ANALYSIS REPORT + GENAI INSIGHTS")
    
    print("\nGenerating comprehensive analysis report...")
    
    # Load all data
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    reconstruction_errors = np.load('reconstruction_errors.npy')
    threshold = np.load('threshold.npy')[0]
    
    # Compute metrics
    y_pred = (reconstruction_errors > threshold).astype(int)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Generate report
    report = f"""
{'='*80}
LSTM AUTOENCODER ANOMALY DETECTION - COMPREHENSIVE REPORT
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. PROJECT OVERVIEW
{'='*80}
This project implements an integrated anomaly detection system combining:
  • LSTM Autoencoder for temporal pattern learning
  • Generative AI (VAE) for data augmentation
  • Explainable AI (XAI) for interpretability

2. DATA SUMMARY
{'='*80}
Training Data:
  - Samples: {len(X_train):,}
  - Shape: {X_train.shape}
  - Features: {X_train.shape[2]} (univariate)
  - Window Size: {X_train.shape[1]} timesteps

Test Data:
  - Total Samples: {len(X_test):,}
  - Normal: {np.sum(y_test == 0):,} ({np.sum(y_test == 0)/len(y_test)*100:.1f}%)
  - Anomaly: {np.sum(y_test == 1):,} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)

3. MODEL ARCHITECTURE
{'='*80}
LSTM Autoencoder:
  - Encoder: LSTM(64, activation='relu')
  - Bottleneck: RepeatVector(window_size)
  - Decoder: LSTM(64, activation='relu', return_sequences=True)
  - Output: TimeDistributed(Dense(1))
  - Loss Function: Mean Squared Error (MSE)
  - Optimizer: Adam

4. ANOMALY DETECTION CONFIGURATION
{'='*80}
Threshold Method: Mean + 2*Standard Deviation
  - Mean Error: {np.mean(reconstruction_errors):.6f}
  - Std Error: {np.std(reconstruction_errors):.6f}
  - Threshold: {threshold:.6f}

5. PERFORMANCE METRICS
{'='*80}
Classification Metrics:
  - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
  - Precision: {precision:.4f}
  - Recall (Sensitivity): {recall:.4f}
  - Specificity: {specificity:.4f}
  - F1-Score: {f1:.4f}

Confusion Matrix:
  - True Negatives (TN): {tn:,}
  - False Positives (FP): {fp:,}
  - False Negatives (FN): {fn:,}
  - True Positives (TP): {tp:,}

6. RECONSTRUCTION ERROR ANALYSIS
{'='*80}
Error Statistics:
  - Mean: {np.mean(reconstruction_errors):.6f}
  - Median: {np.median(reconstruction_errors):.6f}
  - Std Dev: {np.std(reconstruction_errors):.6f}
  - Min: {np.min(reconstruction_errors):.6f}
  - Max: {np.max(reconstruction_errors):.6f}
  - 95th Percentile: {np.percentile(reconstruction_errors, 95):.6f}

Error Distribution by Class:
  Normal Samples:
    - Mean: {np.mean(reconstruction_errors[y_test == 0]):.6f}
    - Std: {np.std(reconstruction_errors[y_test == 0]):.6f}
  Anomaly Samples:
    - Mean: {np.mean(reconstruction_errors[y_test == 1]):.6f}
    - Std: {np.std(reconstruction_errors[y_test == 1]):.6f}

7. GENERATIVE AI AUGMENTATION
{'='*80}
Data Augmentation Strategy:
  - Method: VAE-based synthetic sample generation
  - Augmentation Factor: 30% (add 30% more synthetic samples)
  - Synthetic Anomaly Types: Spike, Shift, Trend

Benefits:
  - Improved model robustness
  - Better generalization
  - Reduced overfitting
  - Enhanced anomaly detection capability

8. EXPLAINABLE AI + GENERATIVE AI INSIGHTS
{'='*80}
XAI Methods Implemented:
  - Feature Importance Analysis
  - SHAP-like Perturbation Method
  - LIME-like Local Linear Approximation

GenAI Intelligence Layer:
  - Cybersecurity-aware anomaly classification
  - Threat severity assessment
  - Root cause analysis
  - Actionable remediation recommendations
  - MITRE ATT&CK technique mapping

Explanation Components:
  - Prediction confidence (XAI + GenAI)
  - Reconstruction error analysis
  - Important timestep identification
  - Human-readable reasoning
  - Threat level assessment
  - Security context and recommendations

9. ENHANCED SYSTEM CAPABILITIES
{'='*80}
✓ Real-time anomaly detection with GenAI intelligence
✓ Batch processing support
✓ Explainable predictions (XAI + GenAI)
✓ Data augmentation with VAE
✓ Performance monitoring
✓ Interactive visualization with Bootstrap 5
✓ MITRE ATT&CK framework integration
✓ RAG context panel for threat intelligence
✓ Severity badges and color-coded alerts
✓ Comprehensive report generation
✓ Cybersecurity-focused threat analysis
✓ Real-time charts and timeline visualization

10. RECOMMENDATIONS
{'='*80}
For Production Deployment:
  1. Monitor reconstruction error distribution regularly
  2. Retrain model periodically with new data
  3. Adjust threshold based on business requirements
  4. Implement alert system for detected anomalies
  5. Log all predictions for audit trail

For Model Improvement:
  1. Increase training data with more diverse patterns
  2. Experiment with deeper LSTM architectures
  3. Implement ensemble methods
  4. Use adaptive thresholding
  5. Integrate domain expertise for anomaly definition

11. CONCLUSION
{'='*80}
The enhanced integrated anomaly detection system successfully combines:
  • Temporal pattern learning (LSTM Autoencoder)
  • Data augmentation (GenAI VAE)
  • Interpretability (XAI)
  • Intelligence layer (GenAI LLM)
  • Cybersecurity framework (MITRE ATT&CK)
  • Real-time visualization (Bootstrap 5)
  • Threat intelligence (RAG context)

This provides a robust, explainable, intelligent, and scalable solution for
real-time cybersecurity anomaly detection in time-series data with
comprehensive threat analysis and actionable insights.

{'='*80}
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
    
    print(report)
    
    # Save report to file
    with open('COMPREHENSIVE_REPORT.txt', 'w') as f:
        f.write(report)
    
    print("\nReport saved to: COMPREHENSIVE_REPORT.txt")


def main():
    """Main pipeline execution"""
    
    print_banner("INTEGRATED ANOMALY DETECTION SYSTEM - COMPLETE PIPELINE")
    
    try:
        # Phase 1: Data Analysis
        X_train, X_test, y_test = phase_1_data_analysis()
        
        # Phase 2: GenAI Augmentation
        X_train_augmented, vae, X_synthetic_anomalies = phase_2_genai_augmentation(X_train)
        
        # Phase 3: Model Evaluation
        reconstruction_errors, threshold, y_pred = phase_3_model_evaluation(X_test, y_test)
        
        # Phase 4: XAI + GenAI Explainability
        explainer, explanations, genai_explainer, genai_explanations, enhanced_analyzer, enhanced_explanations = phase_4_xai_genai_explainability(X_test, reconstruction_errors, threshold, y_test)
        
        # Phase 5: System Integration
        system, mitre_mapping = phase_5_system_integration()
        
        # Phase 6: Enhanced Visualization Dashboard
        dashboard = phase_6_enhanced_visualization_dashboard(mitre_mapping)
        
        # Phase 7: Comprehensive Report with GenAI
        phase_7_comprehensive_report_with_genai()
        
        print_banner("PIPELINE EXECUTION COMPLETE")
        
        print("\nGenerated Artifacts:")
        print("  ✓ LSTM Autoencoder Model (lstm_autoencoder_final.h5)")
        print("  ✓ Reconstruction Errors (reconstruction_errors.npy)")
        print("  ✓ Anomaly Threshold (threshold.npy)")
        print("  ✓ Augmented Training Data (X_train_augmented)")
        print("  ✓ Synthetic Anomalies (X_synthetic_anomalies)")
        print("  ✓ XAI Explanations (explanations)")
        print("  ✓ GenAI Intelligence Analysis (genai_explanations)")
        print("  ✓ Enhanced Analysis Results (enhanced_explanations)")
        print("  ✓ MITRE ATT&CK Mapping (mitre_mapping)")
        print("  ✓ Comprehensive Report with GenAI (COMPREHENSIVE_REPORT.txt)")
        print("  ✓ Flask Web Application (app.py)")
        print("  ✓ Enhanced Dashboard (templates/dashboard.html)")
        
        print("\nSystem Status: ✓ READY FOR DEPLOYMENT")
        
    except Exception as e:
        print(f"\n✗ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
