"""
Flask Web Application for Anomaly Detection System
Frontend + Backend Integration
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import json
from datetime import datetime

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Load data and models
try:
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    reconstruction_errors = np.load('reconstruction_errors.npy')
    threshold = np.load('threshold.npy')[0]
    X_train = np.load('X_train.npy')
    print("[OK] Data loaded successfully")
except Exception as e:
    print(f"[ERROR] Error loading data: {e}")
    X_test = X_train = y_test = reconstruction_errors = None
    threshold = 0

# Import system modules
try:
    from integrated_system import IntegratedAnomalyDetectionSystem
    from xai_explainability import TimeSeriesExplainer
    from genai_integration import EnhancedAnomalyAnalyzer, enhance_prediction_with_genai
    from genai_explainer import explain_anomaly
    system = IntegratedAnomalyDetectionSystem()
    genai_analyzer = EnhancedAnomalyAnalyzer()
    print("[OK] System initialized with GenAI")
except Exception as e:
    print(f"[ERROR] Error initializing system: {e}")
    system = None
    genai_analyzer = None


@app.route('/')
def index():
    """Home page"""
    return render_template('dashboard.html')


@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')


@app.route('/api/dashboard')
def dashboard():
    """Get dashboard data"""
    if reconstruction_errors is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    predictions = (reconstruction_errors > threshold).astype(int)
    
    data = {
        'total_samples': len(reconstruction_errors),
        'anomalies_detected': int(np.sum(predictions == 1)),
        'normal_samples': int(np.sum(predictions == 0)),
        'threshold': float(threshold),
        'error_mean': float(np.mean(reconstruction_errors)),
        'error_std': float(np.std(reconstruction_errors)),
        'accuracy': float(np.sum(predictions == y_test) / len(y_test)) if y_test is not None else 0,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(data)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on sample with XAI and GenAI explanation"""
    try:
        data = request.json
        sample_idx = data.get('sample_idx', 0)
        use_genai = data.get('use_genai', True)
        
        if sample_idx >= len(X_test):
            return jsonify({'error': 'Invalid sample index'}), 400
        
        sample = X_test[sample_idx]
        error = reconstruction_errors[sample_idx]
        actual = int(y_test[sample_idx]) if y_test is not None else None
        
        # Make prediction
        pred = 1 if error > threshold else 0
        
        # Generate XAI explanation
        explainer = TimeSeriesExplainer(window_size=60)
        explanation = explainer.explain_prediction(sample, error, threshold, pred)
        
        # Base result with XAI
        result = {
            'sample_idx': sample_idx,
            'prediction': pred,
            'prediction_label': explanation['prediction'],
            'actual': actual,
            'error': float(error),
            'threshold': float(threshold),
            'confidence': float(explanation['confidence']),
            'severity': explanation['severity'],
            'reason': explanation['reason'],
            'top_timesteps': explanation['top_important_timesteps'][:5].tolist(),
            'importance_scores': explanation['top_importance_scores'][:5].tolist(),
            'pattern_analysis': explanation['pattern_analysis'],
            'contributing_factors': explanation['contributing_factors'],
            'xai_enabled': True
        }
        
        # Enhance with GenAI if available and requested
        if use_genai and genai_analyzer:
            try:
                enhanced = genai_analyzer.analyze_sample_with_genai(
                    sample, error, threshold, explanation
                )
                # Add GenAI fields to result
                result.update({
                    'genai_enabled': enhanced.get('genai_enabled', False),
                    'genai_severity': enhanced.get('genai_severity', 'Medium'),
                    'genai_classification': enhanced.get('genai_classification', 'unknown'),
                    'genai_confidence': enhanced.get('genai_confidence', 75),
                    'genai_root_cause': enhanced.get('genai_root_cause', 'Analysis unavailable'),
                    'genai_recommendation': enhanced.get('genai_recommendation', 'Continue monitoring'),
                    'genai_summary': enhanced.get('genai_summary', 'Anomaly detected'),
                    'threat_level': enhanced.get('threat_level', 'Unknown'),
                    'is_malicious': enhanced.get('is_malicious', False),
                    'requires_immediate_action': enhanced.get('requires_immediate_action', False)
                })
                # Override severity with GenAI assessment
                result['severity'] = enhanced.get('genai_severity', result['severity'])
            except Exception as genai_error:
                result['genai_enabled'] = False
                result['genai_error'] = str(genai_error)
        else:
            result['genai_enabled'] = False
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction"""
    try:
        data = request.json
        start_idx = data.get('start_idx', 0)
        batch_size = data.get('batch_size', 10)
        
        end_idx = min(start_idx + batch_size, len(X_test))
        
        predictions = []
        for idx in range(start_idx, end_idx):
            error = reconstruction_errors[idx]
            pred = 1 if error > threshold else 0
            actual = int(y_test[idx]) if y_test is not None else None
            
            predictions.append({
                'idx': idx,
                'prediction': pred,
                'actual': actual,
                'error': float(error),
                'status': 'ANOMALY' if pred == 1 else 'NORMAL'
            })
        
        return jsonify({
            'predictions': predictions,
            'total': len(X_test),
            'processed': len(predictions)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics')
def statistics():
    """Get system statistics"""
    if reconstruction_errors is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    predictions = (reconstruction_errors > threshold).astype(int)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    stats = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'error_distribution': {
            'mean': float(np.mean(reconstruction_errors)),
            'median': float(np.median(reconstruction_errors)),
            'std': float(np.std(reconstruction_errors)),
            'min': float(np.min(reconstruction_errors)),
            'max': float(np.max(reconstruction_errors))
        }
    }
    return jsonify(stats)


@app.route('/api/augmentation', methods=['POST'])
def augmentation():
    """Data augmentation"""
    try:
        from genai_augmentation import augment_training_data
        
        factor = request.json.get('factor', 0.3)
        X_augmented, report = augment_training_data(X_train, augmentation_factor=factor)
        
        result = {
            'original_samples': int(report['original_samples']),
            'synthetic_samples': int(report['synthetic_samples']),
            'augmented_samples': int(report['augmented_samples']),
            'augmentation_ratio': float(report['augmentation_ratio'])
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/synthetic-anomalies', methods=['POST'])
def synthetic_anomalies():
    """Generate synthetic anomalies"""
    try:
        from genai_augmentation import TimeSeriesVAE, generate_synthetic_anomalies
        
        vae = TimeSeriesVAE(window_size=60)
        vae.fit(X_train)
        
        n_samples = request.json.get('n_samples', 50)
        X_anomalies, y_anomalies = generate_synthetic_anomalies(vae, n_samples=n_samples)
        
        return jsonify({
            'synthetic_anomalies': int(len(X_anomalies)),
            'message': f'Generated {len(X_anomalies)} synthetic anomalies'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/xai-explain', methods=['POST'])
def xai_explain():
    """Get detailed XAI + GenAI explanation for a sample"""
    try:
        from xai_explainability import generate_explanation_report
        
        data = request.json
        sample_idx = data.get('sample_idx', 0)
        use_genai = data.get('use_genai', True)
        
        if sample_idx >= len(X_test):
            return jsonify({'error': 'Invalid sample index'}), 400
        
        sample = X_test[sample_idx]
        error = reconstruction_errors[sample_idx]
        pred = 1 if error > threshold else 0
        
        # Generate comprehensive XAI explanation
        explainer = TimeSeriesExplainer(window_size=60)
        explanation = explainer.explain_prediction(sample, error, threshold, pred)
        
        # Generate text report
        text_report = generate_explanation_report(sample, explanation, error, threshold)
        
        # Base result with XAI
        result = {
            'sample_idx': sample_idx,
            'explanation': {
                'prediction': explanation['prediction'],
                'confidence': float(explanation['confidence']),
                'reconstruction_error': float(explanation['reconstruction_error']),
                'threshold': float(explanation['threshold']),
                'error_ratio': float(explanation['error_ratio']),
                'severity': explanation['severity'],
                'reason': explanation['reason'],
                'top_important_timesteps': explanation['top_important_timesteps'][:10].tolist(),
                'top_importance_scores': explanation['top_importance_scores'][:10].tolist(),
                'pattern_analysis': explanation['pattern_analysis'],
                'contributing_factors': explanation['contributing_factors']
            },
            'text_report': text_report,
            'feature_importance': explanation['feature_importance'].tolist(),
            'sample_data': sample.flatten().tolist()
        }
        
        # Enhance with GenAI analysis
        if use_genai and genai_analyzer:
            try:
                enhanced = genai_analyzer.analyze_sample_with_genai(
                    sample, error, threshold, explanation
                )
                
                # Add GenAI insights to result
                result['genai_analysis'] = {
                    'enabled': enhanced.get('genai_enabled', False),
                    'root_cause': enhanced.get('genai_root_cause', ''),
                    'severity': enhanced.get('genai_severity', 'Medium'),
                    'classification': enhanced.get('genai_classification', 'unknown'),
                    'confidence': enhanced.get('genai_confidence', 75),
                    'recommendation': enhanced.get('genai_recommendation', ''),
                    'summary': enhanced.get('genai_summary', ''),
                    'threat_level': enhanced.get('threat_level', 'Unknown'),
                    'is_malicious': enhanced.get('is_malicious', False),
                    'requires_immediate_action': enhanced.get('requires_immediate_action', False),
                    'threat_indicators': enhanced.get('genai_threat_indicators', [])
                }
                
                # Update explanation severity with GenAI assessment
                result['explanation']['severity'] = enhanced.get('genai_severity', result['explanation']['severity'])
                
            except Exception as genai_error:
                result['genai_analysis'] = {
                    'enabled': False,
                    'error': str(genai_error)
                }
        else:
            result['genai_analysis'] = {'enabled': False}
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/genai-analyze', methods=['POST'])
def genai_analyze():
    """Direct GenAI analysis of anomaly data"""
    try:
        data = request.json
        sample_idx = data.get('sample_idx', 0)
        
        if sample_idx >= len(X_test):
            return jsonify({'error': 'Invalid sample index'}), 400
        
        sample = X_test[sample_idx]
        error = reconstruction_errors[sample_idx]
        
        # Create GenAI payload
        genai_payload = {
            "window_values": sample.flatten().tolist(),
            "reconstruction_error": float(error),
            "error_threshold": float(threshold),
            "xai_top_features": ["time_series_value", "temporal_pattern"],
            "xai_top_timesteps": list(range(min(10, len(sample.flatten()))))
        }
        
        # Get GenAI explanation
        genai_result = explain_anomaly(genai_payload)
        
        # Add sample context
        result = {
            'sample_idx': sample_idx,
            'genai_analysis': genai_result,
            'sample_stats': {
                'mean': float(np.mean(sample)),
                'std': float(np.std(sample)),
                'min': float(np.min(sample)),
                'max': float(np.max(sample))
            },
            'error_context': {
                'reconstruction_error': float(error),
                'threshold': float(threshold),
                'error_ratio': float(error / threshold) if threshold > 0 else 1.0
            }
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/threat-report', methods=['POST'])
def threat_report():
    """Generate comprehensive threat report with GenAI insights"""
    try:
        data = request.json
        start_idx = data.get('start_idx', 0)
        batch_size = data.get('batch_size', 50)
        
        end_idx = min(start_idx + batch_size, len(X_test))
        
        if genai_analyzer:
            # Analyze batch with GenAI
            samples = [X_test[i] for i in range(start_idx, end_idx)]
            errors = [reconstruction_errors[i] for i in range(start_idx, end_idx)]
            
            # Create mock XAI explanations for batch
            xai_explanations = []
            for i in range(len(samples)):
                xai_explanations.append({
                    'prediction': 'ANOMALY' if errors[i] > threshold else 'NORMAL',
                    'confidence': min(0.95, 0.5 + (errors[i] / threshold) * 0.3),
                    'contributing_factors': ['temporal_pattern', 'value_deviation']
                })
            
            # Get enhanced analyses
            analyses = genai_analyzer.batch_analyze_with_genai(
                samples, errors, threshold, xai_explanations
            )
            
            # Generate threat report
            threat_report = genai_analyzer.generate_threat_report(analyses)
            
            return jsonify({
                'threat_report': threat_report,
                'analyzed_range': {'start': start_idx, 'end': end_idx},
                'sample_analyses': analyses[:10]  # Return first 10 detailed analyses
            })
        else:
            return jsonify({'error': 'GenAI analyzer not available'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export-report', methods=['GET'])
def export_report():
    """Export comprehensive analysis report with XAI insights"""
    try:
        predictions = (reconstruction_errors > threshold).astype(int)
        
        from sklearn.metrics import confusion_matrix
        from xai_explainability import TimeSeriesExplainer
        
        cm = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Get XAI + GenAI insights for sample anomalies
        anomaly_indices = np.where(predictions == 1)[0][:5]  # Top 5 anomalies
        explainer = TimeSeriesExplainer(window_size=60)
        
        xai_insights = "\n\nXAI + GENAI INSIGHTS - TOP DETECTED ANOMALIES:\n" + "="*80 + "\n"
        for i, idx in enumerate(anomaly_indices, 1):
            sample = X_test[idx]
            error = reconstruction_errors[idx]
            explanation = explainer.explain_prediction(sample, error, threshold, 1)
            
            # Add GenAI analysis if available
            genai_summary = "GenAI analysis unavailable"
            if genai_analyzer:
                try:
                    enhanced = genai_analyzer.analyze_sample_with_genai(
                        sample, error, threshold, explanation
                    )
                    genai_summary = enhanced.get('genai_summary', 'Analysis failed')
                    threat_level = enhanced.get('threat_level', 'Unknown')
                    is_malicious = enhanced.get('is_malicious', False)
                except:
                    genai_summary = "GenAI analysis failed"
                    threat_level = "Unknown"
                    is_malicious = False
            
            xai_insights += f"\nAnomaly #{i} (Sample {idx}):\n"
            xai_insights += f"  XAI Severity: {explanation['severity']}\n"
            xai_insights += f"  XAI Reason: {explanation['reason']}\n"
            xai_insights += f"  Contributing Factors: {', '.join(explanation['contributing_factors'][:2])}\n"
            if genai_analyzer:
                xai_insights += f"  GenAI Threat Level: {threat_level}\n"
                xai_insights += f"  Malicious: {'Yes' if is_malicious else 'No'}\n"
                xai_insights += f"  GenAI Summary: {genai_summary}\n"
        
        report = f"""
{'='*80}
ANOMALY DETECTION SYSTEM - COMPREHENSIVE ANALYSIS REPORT (XAI + GenAI)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

PERFORMANCE METRICS:
  Accuracy:    {accuracy:.4f}
  Precision:   {precision:.4f}
  Recall:      {recall:.4f}
  F1-Score:    {f1:.4f}

CONFUSION MATRIX:
  True Negatives:  {tn}
  False Positives: {fp}
  False Negatives: {fn}
  True Positives:  {tp}

ERROR STATISTICS:
  Mean:   {np.mean(reconstruction_errors):.6f}
  Std:    {np.std(reconstruction_errors):.6f}
  Min:    {np.min(reconstruction_errors):.6f}
  Max:    {np.max(reconstruction_errors):.6f}
  Threshold: {threshold:.6f}

DATA SUMMARY:
  Total Samples: {len(reconstruction_errors)}
  Anomalies Detected: {np.sum(predictions == 1)}
  Normal Samples: {np.sum(predictions == 0)}
{xai_insights}
{'='*80}
"""
        
        with open('report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        return jsonify({'message': 'Report exported to report.txt', 'report': report})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
