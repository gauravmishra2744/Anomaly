"""
GenAI LLM Intelligence Layer for Cybersecurity Anomaly Detection
================================================================

This module provides intelligent, human-readable explanations for anomalies
detected by LSTM autoencoders using Large Language Models (LLMs).

Features:
- OpenAI GPT integration with fallback mechanisms
- Cybersecurity-aware anomaly classification
- Severity assessment and confidence scoring
- Actionable remediation recommendations
- Robust error handling and data validation

Author: Team Disruptors
Date: 2025-01-22
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")


@dataclass
class AnomalyExplanation:
    """Data class for structured anomaly explanations"""
    root_cause: str
    severity: str  # Low, Medium, High, Critical
    classification: str  # malicious, benign, unknown
    confidence: float  # 0-100
    recommendation: str
    summary: str
    technical_details: Optional[str] = None
    threat_indicators: Optional[List[str]] = None


class CybersecurityPatternMatcher:
    """
    Pattern matching for common cybersecurity anomaly signatures
    Provides fallback explanations when LLM is unavailable
    """
    
    def __init__(self):
        self.patterns = {
            'ddos_spike': {
                'indicators': ['sudden_increase', 'traffic_burst', 'high_volume'],
                'severity': 'High',
                'classification': 'malicious',
                'description': 'Potential DDoS attack detected'
            },
            'memory_leak': {
                'indicators': ['gradual_increase', 'memory_growth', 'resource_exhaustion'],
                'severity': 'Medium',
                'classification': 'benign',
                'description': 'Memory leak or resource exhaustion pattern'
            },
            'cpu_spike': {
                'indicators': ['cpu_burst', 'processing_anomaly', 'compute_spike'],
                'severity': 'Medium',
                'classification': 'unknown',
                'description': 'Unusual CPU utilization pattern'
            },
            'network_scan': {
                'indicators': ['port_scan', 'connection_burst', 'probe_pattern'],
                'severity': 'High',
                'classification': 'malicious',
                'description': 'Network scanning or reconnaissance activity'
            },
            'data_exfiltration': {
                'indicators': ['data_transfer', 'bandwidth_spike', 'upload_anomaly'],
                'severity': 'Critical',
                'classification': 'malicious',
                'description': 'Potential data exfiltration attempt'
            }
        }
    
    def analyze_pattern(self, window_values: List[float], error: float, threshold: float) -> Dict[str, Any]:
        """Analyze anomaly pattern using rule-based approach"""
        
        # Calculate basic statistics
        mean_val = np.mean(window_values)
        std_val = np.std(window_values)
        max_val = np.max(window_values)
        min_val = np.min(window_values)
        
        # Detect pattern characteristics
        is_spike = max_val > (mean_val + 3 * std_val)
        is_gradual = std_val < 0.1 * mean_val
        error_ratio = error / threshold if threshold > 0 else 1
        
        # Pattern matching logic
        if is_spike and error_ratio > 5:
            if mean_val > 0.8:  # High baseline activity
                pattern = 'ddos_spike'
            else:
                pattern = 'cpu_spike'
        elif is_gradual and error_ratio > 2:
            pattern = 'memory_leak'
        elif error_ratio > 10:
            pattern = 'data_exfiltration'
        elif error_ratio > 3:
            pattern = 'network_scan'
        else:
            pattern = 'cpu_spike'  # Default
        
        return self.patterns.get(pattern, self.patterns['cpu_spike'])


class GenAIExplainer:
    """
    Main GenAI explainer class for cybersecurity anomaly analysis
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", max_retries: int = 3):
        """
        Initialize the GenAI explainer
        
        Args:
            model_name: OpenAI model to use
            max_retries: Maximum retry attempts for API calls
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.pattern_matcher = CybersecurityPatternMatcher()
        
        # Initialize OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {e}")
            else:
                logger.warning("OPENAI_API_KEY environment variable not set")
        
        # Cybersecurity domain knowledge
        self.domain_context = """
        You are a cybersecurity expert analyzing time-series anomalies in network traffic, 
        system metrics, and security logs. Focus on identifying potential threats, 
        attack patterns, and security incidents.
        
        Common cybersecurity anomaly types:
        - DDoS attacks (traffic spikes, connection floods)
        - Data exfiltration (unusual outbound traffic)
        - Malware activity (CPU/memory spikes, network beaconing)
        - Brute force attacks (authentication spikes)
        - Network scanning (port scan patterns)
        - System compromise (privilege escalation, lateral movement)
        - Resource exhaustion attacks
        - Insider threats (unusual access patterns)
        """
    
    def _create_analysis_prompt(self, payload: Dict[str, Any]) -> str:
        """
        Create a structured prompt for LLM analysis
        
        Args:
            payload: Input data containing anomaly information
            
        Returns:
            Formatted prompt string
        """
        
        window_values = payload.get('window_values', [])
        reconstruction_error = payload.get('reconstruction_error', 0)
        error_threshold = payload.get('error_threshold', 0)
        xai_features = payload.get('xai_top_features', [])
        xai_timesteps = payload.get('xai_top_timesteps', [])
        
        # Calculate basic statistics
        if window_values:
            stats = {
                'mean': np.mean(window_values),
                'std': np.std(window_values),
                'min': np.min(window_values),
                'max': np.max(window_values),
                'range': np.max(window_values) - np.min(window_values)
            }
        else:
            stats = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'range': 0}
        
        error_ratio = reconstruction_error / error_threshold if error_threshold > 0 else 1
        
        prompt = f"""
{self.domain_context}

ANOMALY ANALYSIS REQUEST:
========================

Time Series Data:
- Window Length: {len(window_values)}
- Values: {window_values[:10]}{'...' if len(window_values) > 10 else ''}
- Statistics: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, Min={stats['min']:.4f}, Max={stats['max']:.4f}

Anomaly Detection Results:
- Reconstruction Error: {reconstruction_error:.6f}
- Error Threshold: {error_threshold:.6f}
- Error Ratio: {error_ratio:.2f}x above threshold

XAI Explanations:
- Top Important Features: {xai_features}
- Critical Timesteps: {xai_timesteps}

ANALYSIS REQUIREMENTS:
======================
Provide a JSON response with the following structure:
{{
    "root_cause": "Detailed technical explanation of what caused this anomaly",
    "severity": "Low|Medium|High|Critical",
    "classification": "malicious|benign|unknown",
    "confidence": 85,
    "recommendation": "Specific actionable steps to address this anomaly",
    "summary": "Brief 1-2 sentence summary for dashboard display",
    "technical_details": "Additional technical context",
    "threat_indicators": ["indicator1", "indicator2"]
}}

ANALYSIS GUIDELINES:
===================
1. Consider the error ratio - higher ratios indicate more severe anomalies
2. Look for cybersecurity patterns (spikes, gradual increases, periodic behavior)
3. Assess whether this could be an attack, system issue, or normal variation
4. Provide confidence based on data quality and pattern clarity
5. Give specific, actionable recommendations
6. Keep summary concise but informative

Respond ONLY with valid JSON, no additional text.
"""
        return prompt
    
    def _call_openai_api(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Make API call to OpenAI with retry logic
        
        Args:
            prompt: Formatted prompt for analysis
            
        Returns:
            Parsed JSON response or None if failed
        """
        if not self.openai_client:
            return None
        
        for attempt in range(self.max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a cybersecurity expert specializing in anomaly analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                
                content = response.choices[0].message.content.strip()
                
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    return json.loads(json_str)
                else:
                    logger.warning(f"No JSON found in response: {content}")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"OpenAI API error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def _generate_fallback_explanation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanation using rule-based fallback when LLM is unavailable
        
        Args:
            payload: Input anomaly data
            
        Returns:
            Structured explanation dictionary
        """
        window_values = payload.get('window_values', [])
        reconstruction_error = payload.get('reconstruction_error', 0)
        error_threshold = payload.get('error_threshold', 0.001)
        
        # Use pattern matcher
        pattern_info = self.pattern_matcher.analyze_pattern(
            window_values, reconstruction_error, error_threshold
        )
        
        error_ratio = reconstruction_error / error_threshold if error_threshold > 0 else 1
        
        # Calculate confidence based on error ratio and data quality
        confidence = min(95, max(60, 70 + (error_ratio - 1) * 10))
        
        # Generate recommendations based on classification
        if pattern_info['classification'] == 'malicious':
            recommendation = "Immediate investigation required. Check network logs, isolate affected systems, and review security controls."
        elif pattern_info['classification'] == 'benign':
            recommendation = "Monitor system resources and consider optimization. Schedule maintenance if pattern persists."
        else:
            recommendation = "Continue monitoring. Gather additional data to determine if this represents a genuine threat."
        
        return {
            "root_cause": f"{pattern_info['description']}. Reconstruction error is {error_ratio:.1f}x above normal threshold, indicating significant deviation from learned patterns.",
            "severity": pattern_info['severity'],
            "classification": pattern_info['classification'],
            "confidence": confidence,
            "recommendation": recommendation,
            "summary": f"{pattern_info['severity']} severity anomaly detected with {error_ratio:.1f}x error threshold exceedance",
            "technical_details": f"Error: {reconstruction_error:.6f}, Threshold: {error_threshold:.6f}, Window size: {len(window_values)}",
            "threat_indicators": ["reconstruction_error_spike", "pattern_deviation"]
        }
    
    def explain_anomaly(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main function to explain an anomaly using GenAI
        
        Args:
            payload: Dictionary containing:
                - window_values: List of time series values
                - reconstruction_error: Float error value
                - error_threshold: Float threshold value
                - xai_top_features: List of important features
                - xai_top_timesteps: List of critical timesteps
                
        Returns:
            Dictionary with explanation fields
        """
        logger.info("Starting anomaly explanation analysis")
        
        try:
            # Validate input payload
            if not isinstance(payload, dict):
                raise ValueError("Payload must be a dictionary")
            
            required_keys = ['window_values', 'reconstruction_error', 'error_threshold']
            for key in required_keys:
                if key not in payload:
                    raise ValueError(f"Missing required key: {key}")
            
            # Try LLM analysis first
            if self.openai_client:
                logger.info("Attempting LLM analysis")
                prompt = self._create_analysis_prompt(payload)
                llm_response = self._call_openai_api(prompt)
                
                if llm_response:
                    logger.info("LLM analysis successful")
                    return llm_response
                else:
                    logger.warning("LLM analysis failed, using fallback")
            else:
                logger.info("LLM not available, using fallback analysis")
            
            # Use fallback analysis
            return self._generate_fallback_explanation(payload)
            
        except Exception as e:
            logger.error(f"Error in anomaly explanation: {e}")
            # Return minimal safe response
            return {
                "root_cause": "Unable to analyze anomaly due to system error",
                "severity": "Medium",
                "classification": "unknown",
                "confidence": 50,
                "recommendation": "Manual investigation required",
                "summary": "Anomaly detected but analysis failed",
                "technical_details": f"Error: {str(e)}",
                "threat_indicators": ["analysis_error"]
            }


# Convenience function for easy integration
def explain_anomaly(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to explain an anomaly
    
    Args:
        payload: Anomaly data dictionary
        
    Returns:
        Explanation dictionary
    """
    explainer = GenAIExplainer()
    return explainer.explain_anomaly(payload)