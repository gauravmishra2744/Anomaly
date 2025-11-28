"""
GenAI Integration Module for LSTM Anomaly Detection System
==========================================================

This module integrates the GenAI explainer with the existing LSTM autoencoder
anomaly detection system, providing enhanced explanations for detected anomalies.

Features:
- Seamless integration with existing XAI explanations
- Enhanced anomaly classification and severity assessment
- Cybersecurity-focused threat analysis
- Production-ready API endpoints

Author: Team Disruptors
Date: 2025-01-22
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional
from genai_explainer_simple import GenAIExplainer, explain_anomaly

logger = logging.getLogger(__name__)


class EnhancedAnomalyAnalyzer:
    """
    Enhanced anomaly analyzer that combines LSTM, XAI, and GenAI explanations
    """
    
    def __init__(self):
        """Initialize the enhanced analyzer"""
        self.genai_explainer = GenAIExplainer()
        logger.info("Enhanced Anomaly Analyzer initialized")
    
    def analyze_sample_with_genai(self, 
                                 sample_data: np.ndarray,
                                 reconstruction_error: float,
                                 threshold: float,
                                 xai_explanation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single sample with GenAI enhancement
        
        Args:
            sample_data: Time series window data
            reconstruction_error: LSTM reconstruction error
            threshold: Anomaly detection threshold
            xai_explanation: XAI explanation from existing system
            
        Returns:
            Enhanced explanation with GenAI insights
        """
        try:
            # Prepare payload for GenAI
            window_values = sample_data.flatten().tolist() if len(sample_data.shape) > 1 else sample_data.tolist()
            
            # Extract XAI features and timesteps
            xai_features = []
            xai_timesteps = []
            
            if 'top_important_timesteps' in xai_explanation:
                xai_timesteps = xai_explanation['top_important_timesteps'][:5].tolist()
            
            if 'contributing_factors' in xai_explanation:
                xai_features = xai_explanation['contributing_factors'][:3]
            
            # Create GenAI payload
            genai_payload = {
                "window_values": window_values,
                "reconstruction_error": float(reconstruction_error),
                "error_threshold": float(threshold),
                "xai_top_features": xai_features,
                "xai_top_timesteps": xai_timesteps
            }
            
            # Get GenAI explanation
            genai_result = self.genai_explainer.explain_anomaly(genai_payload)
            
            # Combine with existing XAI explanation
            enhanced_result = {
                # Original XAI fields
                'prediction': xai_explanation.get('prediction', 'UNKNOWN'),
                'confidence': xai_explanation.get('confidence', 0.5),
                'reconstruction_error': float(reconstruction_error),
                'threshold': float(threshold),
                'error_ratio': float(reconstruction_error / threshold) if threshold > 0 else 1.0,
                
                # Enhanced GenAI fields
                'genai_root_cause': genai_result['root_cause'],
                'genai_severity': genai_result['severity'],
                'genai_classification': genai_result['classification'],
                'genai_confidence': genai_result['confidence'],
                'genai_recommendation': genai_result['recommendation'],
                'genai_summary': genai_result['summary'],
                'genai_technical_details': genai_result.get('technical_details', ''),
                'genai_threat_indicators': genai_result.get('threat_indicators', []),
                
                # Combined analysis
                'severity': genai_result['severity'],
                'threat_level': self._calculate_threat_level(genai_result),
                'is_malicious': genai_result['classification'] == 'malicious',
                'requires_immediate_action': genai_result['severity'] in ['High', 'Critical'],
                
                # Original XAI context
                'xai_explanation': xai_explanation,
                'genai_enabled': True
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in GenAI analysis: {e}")
            # Return original explanation with error flag
            return {
                **xai_explanation,
                'genai_enabled': False,
                'genai_error': str(e),
                'severity': 'Medium',  # Default severity
                'threat_level': 'Unknown',
                'is_malicious': False,
                'requires_immediate_action': False
            }
    
    def _calculate_threat_level(self, genai_result: Dict[str, Any]) -> str:
        """
        Calculate overall threat level based on GenAI analysis
        
        Args:
            genai_result: GenAI analysis result
            
        Returns:
            Threat level string
        """
        severity = genai_result.get('severity', 'Medium')
        classification = genai_result.get('classification', 'unknown')
        confidence = genai_result.get('confidence', 50)
        
        if classification == 'malicious':
            if severity == 'Critical' and confidence > 80:
                return 'CRITICAL THREAT'
            elif severity in ['High', 'Critical']:
                return 'HIGH THREAT'
            else:
                return 'MODERATE THREAT'
        elif classification == 'benign':
            return 'LOW RISK'
        else:
            if severity in ['High', 'Critical']:
                return 'INVESTIGATE'
            else:
                return 'MONITOR'
    
    def batch_analyze_with_genai(self,
                                samples: List[np.ndarray],
                                errors: List[float],
                                threshold: float,
                                xai_explanations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple samples with GenAI enhancement
        
        Args:
            samples: List of time series samples
            errors: List of reconstruction errors
            threshold: Anomaly detection threshold
            xai_explanations: List of XAI explanations
            
        Returns:
            List of enhanced explanations
        """
        results = []
        
        for i, (sample, error, xai_exp) in enumerate(zip(samples, errors, xai_explanations)):
            try:
                enhanced = self.analyze_sample_with_genai(sample, error, threshold, xai_exp)
                enhanced['batch_index'] = i
                results.append(enhanced)
            except Exception as e:
                logger.error(f"Error analyzing sample {i}: {e}")
                # Add error result
                results.append({
                    'batch_index': i,
                    'genai_enabled': False,
                    'genai_error': str(e),
                    'severity': 'Medium',
                    'threat_level': 'Unknown'
                })
        
        return results
    
    def generate_threat_report(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive threat report from multiple analyses
        
        Args:
            analyses: List of enhanced analysis results
            
        Returns:
            Comprehensive threat report
        """
        total_samples = len(analyses)
        malicious_count = sum(1 for a in analyses if a.get('is_malicious', False))
        critical_count = sum(1 for a in analyses if a.get('severity') == 'Critical')
        high_count = sum(1 for a in analyses if a.get('severity') == 'High')
        
        # Collect threat indicators
        all_indicators = []
        for analysis in analyses:
            indicators = analysis.get('genai_threat_indicators', [])
            all_indicators.extend(indicators)
        
        # Count unique indicators
        from collections import Counter
        indicator_counts = Counter(all_indicators)
        top_indicators = indicator_counts.most_common(5)
        
        # Calculate risk score
        risk_score = min(100, (malicious_count * 20 + critical_count * 15 + high_count * 10) / max(1, total_samples) * 100)
        
        report = {
            'timestamp': np.datetime64('now').astype(str),
            'total_samples_analyzed': total_samples,
            'threat_summary': {
                'malicious_samples': malicious_count,
                'critical_severity': critical_count,
                'high_severity': high_count,
                'risk_score': round(risk_score, 2)
            },
            'top_threat_indicators': [{'indicator': ind, 'count': count} for ind, count in top_indicators],
            'recommendations': self._generate_report_recommendations(analyses),
            'genai_enabled_samples': sum(1 for a in analyses if a.get('genai_enabled', False)),
            'analysis_quality': 'High' if sum(1 for a in analyses if a.get('genai_enabled', False)) > total_samples * 0.8 else 'Medium'
        }
        
        return report
    
    def _generate_report_recommendations(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        malicious_count = sum(1 for a in analyses if a.get('is_malicious', False))
        critical_count = sum(1 for a in analyses if a.get('severity') == 'Critical')
        
        if critical_count > 0:
            recommendations.append("IMMEDIATE ACTION: Critical threats detected - initiate incident response procedures")
        
        if malicious_count > len(analyses) * 0.1:  # More than 10% malicious
            recommendations.append("HIGH PRIORITY: Elevated threat activity - enhance monitoring and security controls")
        
        if malicious_count > 0:
            recommendations.append("INVESTIGATE: Review security logs and network traffic for confirmed threats")
        
        recommendations.append("MONITOR: Continue automated anomaly detection and periodic threat assessment")
        
        return recommendations


# Integration functions for Flask app
def enhance_prediction_with_genai(sample_data: np.ndarray,
                                 reconstruction_error: float,
                                 threshold: float,
                                 xai_explanation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to enhance a single prediction with GenAI
    
    Args:
        sample_data: Time series sample
        reconstruction_error: LSTM reconstruction error
        threshold: Anomaly threshold
        xai_explanation: Existing XAI explanation
        
    Returns:
        Enhanced explanation with GenAI insights
    """
    analyzer = EnhancedAnomalyAnalyzer()
    return analyzer.analyze_sample_with_genai(sample_data, reconstruction_error, threshold, xai_explanation)


def create_genai_payload_from_sample(sample_idx: int,
                                    X_test: np.ndarray,
                                    reconstruction_errors: np.ndarray,
                                    threshold: float) -> Dict[str, Any]:
    """
    Create a GenAI payload from sample data for direct analysis
    
    Args:
        sample_idx: Index of the sample
        X_test: Test data array
        reconstruction_errors: Array of reconstruction errors
        threshold: Anomaly detection threshold
        
    Returns:
        GenAI payload dictionary
    """
    sample = X_test[sample_idx]
    error = reconstruction_errors[sample_idx]
    
    # Create basic payload
    payload = {
        "window_values": sample.flatten().tolist(),
        "reconstruction_error": float(error),
        "error_threshold": float(threshold),
        "xai_top_features": ["time_series_value"],  # Generic feature name
        "xai_top_timesteps": list(range(min(5, len(sample.flatten()))))  # First 5 timesteps
    }
    
    return payload


# Demo usage
if __name__ == "__main__":
    """
    Demo of GenAI integration with existing system
    """
    print("=" * 80)
    print("GenAI Integration Demo")
    print("=" * 80)
    
    # Simulate existing system data
    sample_data = np.array([0.1, 0.12, 0.11, 0.89, 0.95, 0.92, 0.14, 0.13, 0.12, 0.11])
    reconstruction_error = 0.0156
    threshold = 0.002
    
    # Simulate existing XAI explanation
    xai_explanation = {
        'prediction': 'ANOMALY',
        'confidence': 0.85,
        'top_important_timesteps': np.array([3, 4, 5]),
        'contributing_factors': ['traffic_spike', 'connection_burst'],
        'pattern_analysis': {'mean': 0.35, 'std': 0.32}
    }
    
    # Test enhanced analysis
    analyzer = EnhancedAnomalyAnalyzer()
    enhanced_result = analyzer.analyze_sample_with_genai(
        sample_data, reconstruction_error, threshold, xai_explanation
    )
    
    print("\nEnhanced Analysis Result:")
    print(f"Severity: {enhanced_result['severity']}")
    print(f"Threat Level: {enhanced_result['threat_level']}")
    print(f"Is Malicious: {enhanced_result['is_malicious']}")
    print(f"Requires Action: {enhanced_result['requires_immediate_action']}")
    print(f"GenAI Summary: {enhanced_result['genai_summary']}")
    print(f"Recommendation: {enhanced_result['genai_recommendation']}")
    
    print("\n" + "=" * 80)
    print("Integration demo completed successfully!")
    print("=" * 80)