"""
Explainable AI (XAI) for Anomaly Detection
Provides interpretability using SHAP-like and LIME-like techniques
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


class TimeSeriesExplainer:
    """Explains anomaly detection decisions using gradient-based and perturbation methods"""
    
    def __init__(self, model=None, window_size=60):
        self.model = model
        self.window_size = window_size
        
    def compute_feature_importance(self, sample, reconstruction_error, threshold):
        """
        Compute importance of each timestep in the sample
        Uses multiple methods: deviation, gradient, and variance
        """
        sample_flat = sample.flatten()
        
        # Method 1: Deviation from mean
        mean_val = np.mean(sample_flat)
        deviation = np.abs(sample_flat - mean_val)
        
        # Method 2: Gradient (rate of change)
        gradient = np.abs(np.gradient(sample_flat))
        
        # Method 3: Local variance
        window = 5
        local_var = np.zeros_like(sample_flat)
        for i in range(len(sample_flat)):
            start = max(0, i - window // 2)
            end = min(len(sample_flat), i + window // 2 + 1)
            local_var[i] = np.var(sample_flat[start:end])
        
        # Combine methods (weighted average)
        importance = 0.4 * deviation + 0.3 * gradient + 0.3 * local_var
        
        # Normalize importance scores
        importance = importance / (np.sum(importance) + 1e-8)
        
        return importance
    
    def explain_prediction(self, sample, reconstruction_error, threshold, y_pred):
        """
        Generate comprehensive explanation for a single prediction
        
        Returns:
            explanation: Dictionary with detailed explanation
        """
        is_anomaly = y_pred == 1
        sample_flat = sample.flatten()
        
        # Compute feature importance
        importance = self.compute_feature_importance(sample, reconstruction_error, threshold)
        
        # Find most important timesteps
        top_k = min(10, len(importance))
        top_indices = np.argsort(importance)[-top_k:][::-1]
        
        # Analyze pattern characteristics
        pattern_analysis = self._analyze_pattern(sample_flat, importance)
        
        # Generate detailed reason
        reason, severity = _generate_detailed_reason(is_anomaly, reconstruction_error, threshold, 
                                                      importance, pattern_analysis)
        
        explanation = {
            'prediction': 'ANOMALY' if is_anomaly else 'NORMAL',
            'confidence': min(abs(reconstruction_error - threshold) / (threshold + 1e-8), 1.0),
            'reconstruction_error': reconstruction_error,
            'threshold': threshold,
            'error_ratio': reconstruction_error / (threshold + 1e-8),
            'severity': severity,
            'feature_importance': importance,
            'top_important_timesteps': top_indices,
            'top_importance_scores': importance[top_indices],
            'reason': reason,
            'pattern_analysis': pattern_analysis,
            'contributing_factors': self._identify_factors(sample_flat, importance, is_anomaly)
        }
        
        return explanation
    
    def _analyze_pattern(self, sample, importance):
        """Analyze pattern characteristics"""
        return {
            'mean': float(np.mean(sample)),
            'std': float(np.std(sample)),
            'max': float(np.max(sample)),
            'min': float(np.min(sample)),
            'range': float(np.max(sample) - np.min(sample)),
            'trend': 'increasing' if sample[-1] > sample[0] else 'decreasing',
            'volatility': float(np.std(np.diff(sample))),
            'peak_count': int(len(self._find_peaks(sample))),
            'anomaly_concentration': float(np.sum(importance > np.mean(importance)) / len(importance))
        }
    
    def _find_peaks(self, sample, threshold=0.5):
        """Find peaks in time series"""
        peaks = []
        for i in range(1, len(sample) - 1):
            if sample[i] > sample[i-1] and sample[i] > sample[i+1]:
                if sample[i] > threshold:
                    peaks.append(i)
        return peaks
    
    def _identify_factors(self, sample, importance, is_anomaly):
        """Identify contributing factors to the prediction"""
        factors = []
        
        # Check for spikes
        if np.max(sample) > np.mean(sample) + 2 * np.std(sample):
            factors.append('Spike detected in time series')
        
        # Check for sudden drops
        if np.min(sample) < np.mean(sample) - 2 * np.std(sample):
            factors.append('Sudden drop detected')
        
        # Check for high variance
        if np.std(sample) > 1.5:
            factors.append('High variance in pattern')
        
        # Check for concentrated importance
        high_importance_ratio = np.sum(importance > 2 * np.mean(importance)) / len(importance)
        if high_importance_ratio > 0.2:
            factors.append(f'Anomaly concentrated in {high_importance_ratio:.1%} of timesteps')
        
        # Check for trend changes
        mid = len(sample) // 2
        first_half_mean = np.mean(sample[:mid])
        second_half_mean = np.mean(sample[mid:])
        if abs(second_half_mean - first_half_mean) > 0.5:
            factors.append('Significant trend change detected')
        
        if not factors:
            factors.append('Normal pattern within expected range')
        
        return factors
    
    def explain_batch(self, X_samples, reconstruction_errors, threshold, y_preds):
        """Explain predictions for a batch of samples"""
        explanations = []
        
        for i, (sample, error, pred) in enumerate(zip(X_samples, reconstruction_errors, y_preds)):
            exp = self.explain_prediction(sample, error, threshold, pred)
            explanations.append(exp)
        
        return explanations


def _generate_detailed_reason(is_anomaly, error, threshold, importance, pattern_analysis):
    """Generate detailed human-readable reason for prediction"""
    error_ratio = error / (threshold + 1e-8)
    
    if is_anomaly:
        if error_ratio > 5:
            severity = 'CRITICAL'
            reason = f"CRITICAL THREAT: Reconstruction error {error_ratio:.1f}x threshold. "
        elif error_ratio > 3:
            severity = 'HIGH'
            reason = f"HIGH RISK: Reconstruction error {error_ratio:.1f}x threshold. "
        elif error_ratio > 2:
            severity = 'MEDIUM'
            reason = f"MEDIUM RISK: Reconstruction error {error_ratio:.1f}x threshold. "
        else:
            severity = 'LOW'
            reason = f"LOW RISK: Reconstruction error {error_ratio:.1f}x threshold. "
        
        # Add pattern details
        if pattern_analysis['volatility'] > 1.0:
            reason += "High volatility detected. "
        if pattern_analysis['peak_count'] > 3:
            reason += f"{pattern_analysis['peak_count']} peaks found. "
        if pattern_analysis['anomaly_concentration'] > 0.3:
            reason += "Anomaly concentrated in specific timesteps."
    else:
        severity = 'NORMAL'
        if error_ratio < 0.3:
            reason = "SECURE: Very low reconstruction error. Pattern matches normal behavior closely."
        elif error_ratio < 0.7:
            reason = "SECURE: Low reconstruction error. Pattern within expected range."
        else:
            reason = "SECURE: Pattern within normal threshold. No threat detected."
    
    return reason, severity


class SHAPLikeExplainer:
    """SHAP-inspired explainer using perturbation-based approach"""
    
    def __init__(self, model, window_size=60, n_samples=100):
        self.model = model
        self.window_size = window_size
        self.n_samples = n_samples
        
    def explain_sample(self, sample, baseline=None):
        """
        Explain prediction using SHAP-like approach
        Computes contribution of each feature to the prediction
        """
        if baseline is None:
            baseline = np.zeros_like(sample)
        
        # Compute baseline prediction
        baseline_error = self._compute_error(baseline)
        
        # Compute sample prediction
        sample_error = self._compute_error(sample)
        
        # Compute SHAP values (simplified)
        shap_values = np.zeros_like(sample)
        
        for i in range(sample.shape[0]):
            # Create perturbed sample with feature i from baseline
            perturbed = sample.copy()
            perturbed[i] = baseline[i]
            
            perturbed_error = self._compute_error(perturbed)
            
            # SHAP value = contribution to prediction
            shap_values[i] = sample_error - perturbed_error
        
        return shap_values
    
    def _compute_error(self, sample):
        """Compute reconstruction error for a sample"""
        if self.model is not None:
            reconstructed = self.model.predict(sample.reshape(1, -1, 1), verbose=0)
            error = np.mean(np.square(sample - reconstructed.flatten()))
        else:
            error = np.mean(np.square(sample))
        
        return error


class LIMELikeExplainer:
    """LIME-inspired explainer using local linear approximation"""
    
    def __init__(self, model, window_size=60, n_samples=100):
        self.model = model
        self.window_size = window_size
        self.n_samples = n_samples
        
    def explain_sample(self, sample):
        """
        Explain prediction using LIME-like approach
        Fits local linear model around the sample
        """
        # Generate perturbed samples
        perturbed_samples = []
        predictions = []
        
        for _ in range(self.n_samples):
            # Create perturbed sample (random masking)
            mask = np.random.binomial(1, 0.5, size=sample.shape)
            perturbed = sample * mask
            
            perturbed_samples.append(perturbed.flatten())
            
            # Get prediction for perturbed sample
            if self.model is not None:
                pred = self.model.predict(perturbed.reshape(1, -1, 1), verbose=0)
                error = np.mean(np.square(perturbed - pred.flatten()))
            else:
                error = np.mean(np.square(perturbed))
            
            predictions.append(error)
        
        # Fit linear model: predictions ~ perturbed_samples
        X_perturbed = np.array(perturbed_samples)
        y_predictions = np.array(predictions)
        
        # Simple linear regression: compute coefficients
        # coefficients represent feature importance
        coefficients = np.zeros(sample.shape[0])
        
        for i in range(sample.shape[0]):
            # Correlation between feature i and predictions
            if np.std(X_perturbed[:, i]) > 0:
                coefficients[i] = np.corrcoef(X_perturbed[:, i], y_predictions)[0, 1]
        
        # Normalize coefficients
        coefficients = np.abs(coefficients) / (np.sum(np.abs(coefficients)) + 1e-8)
        
        return coefficients


def generate_explanation_report(sample, explanation, reconstruction_error, threshold):
    """Generate comprehensive human-readable explanation report"""
    
    report = f"""
{'='*80}
XAI EXPLAINABILITY REPORT - ANOMALY DETECTION SYSTEM
{'='*80}

PREDICTION: {explanation['prediction']} (Severity: {explanation['severity']})
Confidence: {explanation['confidence']:.2%}

{'='*80}
RECONSTRUCTION ERROR ANALYSIS:
{'='*80}
  Reconstruction Error: {reconstruction_error:.6f}
  Threshold: {threshold:.6f}
  Error Ratio: {explanation['error_ratio']:.2f}x threshold

{'='*80}
EXPLANATION:
{'='*80}
{explanation['reason']}

{'='*80}
PATTERN ANALYSIS:
{'='*80}
  Mean Value: {explanation['pattern_analysis']['mean']:.4f}
  Std Deviation: {explanation['pattern_analysis']['std']:.4f}
  Range: {explanation['pattern_analysis']['range']:.4f}
  Trend: {explanation['pattern_analysis']['trend']}
  Volatility: {explanation['pattern_analysis']['volatility']:.4f}
  Peak Count: {explanation['pattern_analysis']['peak_count']}
  Anomaly Concentration: {explanation['pattern_analysis']['anomaly_concentration']:.2%}

{'='*80}
CONTRIBUTING FACTORS:
{'='*80}
"""
    
    for i, factor in enumerate(explanation['contributing_factors'], 1):
        report += f"  {i}. {factor}\n"
    
    report += f"\n{'='*80}\nTOP IMPORTANT TIMESTEPS:\n{'='*80}\n"
    
    for rank, (idx, score) in enumerate(zip(explanation['top_important_timesteps'][:5], 
                                             explanation['top_importance_scores'][:5]), 1):
        marker = '[HIGH]' if score > 0.05 else '[LOW]'
        report += f"  {rank}. Timestep {idx}: Importance {score:.4f} {marker}\n"
    
    report += f"\n{'='*80}\n"
    
    return report


def save_explainer_config(output_dir='./preprocessing_artifacts'):
    """Save explainer configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        'explainer_type': 'TimeSeriesExplainer',
        'methods': ['feature_importance', 'shap_like', 'lime_like'],
        'description': 'XAI methods for time-series anomaly detection'
    }
    
    config_path = os.path.join(output_dir, 'xai_config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    
    print(f"XAI configuration saved to: {config_path}")


if __name__ == "__main__":
    # Example usage
    print("Initializing XAI Explainer...")
    explainer = TimeSeriesExplainer(window_size=60)
    
    # Create sample data
    sample = np.random.randn(60, 1)
    reconstruction_error = 0.005
    threshold = 0.003
    y_pred = 1  # Anomaly
    
    # Generate explanation
    explanation = explainer.explain_prediction(sample, reconstruction_error, threshold, y_pred)
    
    # Print report
    report = generate_explanation_report(sample, explanation, reconstruction_error, threshold)
    print(report)
    
    # Save configuration
    save_explainer_config()
    
    print("XAI module initialized successfully!")
