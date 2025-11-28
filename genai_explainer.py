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

# =====================================================================
# GenAI Explainer v3.0 Enterprise Edition
# - Adds RAG (Retrieval-Augmented Generation)
# - Adds MITRE ATT&CK mapping
# - Uses GPT-4.1-mini + Function Calling
# =====================================================================

import os
import json
import logging
import numpy as np
from dataclasses import dataclass

from openai import OpenAI
import chromadb

# Initialize
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GenAI-RAG-Explainer")


# =====================================================================
# MITRE ATT&CK Mapping
# =====================================================================
MITRE_MAP = {
    "ddos_spike": {
        "tactics": ["Impact"],
        "techniques": ["T1499 - Endpoint DoS", "T1498 - Network DoS"],
    },
    "network_scan": {
        "tactics": ["Discovery"],
        "techniques": ["T1046 - Network Service Scanning"],
    },
    "data_exfiltration": {
        "tactics": ["Exfiltration"],
        "techniques": ["T1041 - Exfiltration Over Command and Control Channel",
                       "T1048 - Exfiltration Over Alternative Protocol"],
    },
    "memory_leak": {
        "tactics": [],
        "techniques": [],
    },
    "cpu_spike": {
        "tactics": ["Execution"],
        "techniques": ["T1059 - Command & Scripting Interpreter",
                       "T1204 - User Execution"],
    }
}


# =====================================================================
# Dataclass
# =====================================================================
@dataclass
class AnomalyExplanation:
    root_cause: str
    severity: str
    classification: str
    confidence: float
    recommendation: str
    summary: str
    technical_details: str
    threat_indicators: list
    mitre_tactics: list
    mitre_techniques: list
    retrieved_context: str


# =====================================================================
# Fallback Rule-Based Interpreter
# =====================================================================
class PatternMatcher:
    def analyze(self, values, error, threshold):
        mean = np.mean(values)
        std = np.std(values)
        max_val = np.max(values)

        ratio = error / threshold if threshold > 0 else 10

        if max_val > (mean + 3 * std) and ratio > 5:
            return "ddos_spike", "High", "malicious"

        if std < 0.1 * mean and ratio > 2:
            return "memory_leak", "Medium", "benign"

        if ratio > 10:
            return "data_exfiltration", "Critical", "malicious"

        if ratio > 3:
            return "network_scan", "High", "malicious"

        return "cpu_spike", "Medium", "unknown"


# =====================================================================
# GenAI Explainer v3.0 (with RAG + MITRE)
# =====================================================================
class GenAIExplainer:

    def __init__(self, model="gpt-4.1-mini"):
        self.model = model
        self.patterns = PatternMatcher()

        # OpenAI Setup
        api = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api) if api else None

        # Vector Store (ChromaDB)
        self.chroma = chromadb.Client()
        self.collection = self.chroma.get_or_create_collection(
            name="cybersecurity_docs",
            metadata={"description": "Cybersecurity RAG Knowledge Base"},
        )

        logger.info("GenAI v3.0 Initialized (RAG + MITRE)")


    # RAG Retrieval
    def _rag_search(self, query):
        if self.collection.count() == 0:
            return "No RAG documents found."

        results = self.collection.query(
            query_texts=[query],
            n_results=3
        )

        return "\n\n".join(results["documents"][0])


    # LLM Function Schema
    def _schema(self):
        return {
            "name": "cyber_anomaly_report",
            "parameters": {
                "type": "object",
                "properties": {
                    "root_cause": {"type": "string"},
                    "severity": {"type": "string"},
                    "classification": {"type": "string"},
                    "confidence": {"type": "number"},
                    "recommendation": {"type": "string"},
                    "summary": {"type": "string"},
                    "technical_details": {"type": "string"},
                    "threat_indicators": {
                        "type": "array", "items": {"type": "string"}
                    },
                },
                "required": ["root_cause", "severity", "classification",
                             "confidence", "recommendation", "summary",
                             "technical_details", "threat_indicators"]
            }
        }


    # Build the full LLM input including RAG + MITRE
    def _build_prompt(self, payload, pattern):
        rag_context = self._rag_search(pattern)
        mitre = MITRE_MAP.get(pattern, {"tactics": [], "techniques": []})

        return f"""
You are a senior cybersecurity analyst. Use the RAG context and MITRE ATT&CK data.

ANOMALY:
Window: {payload['window_values']}
Error: {payload['reconstruction_error']}
Threshold: {payload['error_threshold']}
Pattern: {pattern}

RAG CONTEXT:
{rag_context}

MITRE:
Tactics: {mitre['tactics']}
Techniques: {mitre['techniques']}

Generate a JSON response using the function schema.
"""


    # GPT Call
    def _call_gpt(self, prompt):
        try:
            out = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                functions=[self._schema()],
                function_call={"name": "cyber_anomaly_report"}
            )
            return json.loads(out.choices[0].message.function_call.arguments)
        except:
            return None


    # Public Main Function
    def explain(self, payload):

        # Determine Pattern
        pattern, severity, classification = self.patterns.analyze(
            payload["window_values"],
            payload["reconstruction_error"],
            payload["error_threshold"]
        )

        prompt = self._build_prompt(payload, pattern)

        # Try LLM First
        if self.client:
            llm = self._call_gpt(prompt)
            if llm:
                mitre = MITRE_MAP.get(pattern)
                llm["mitre_tactics"] = mitre["tactics"]
                llm["mitre_techniques"] = mitre["techniques"]
                llm["retrieved_context"] = self._rag_search(pattern)
                return llm

        # Fallback
        fb = {
            "root_cause": f"{pattern} behavior detected",
            "severity": severity,
            "classification": classification,
            "confidence": 70,
            "recommendation": "Investigate the anomaly using system logs.",
            "summary": f"{severity} severity anomaly ({pattern})",
            "technical_details": f"Error={payload['reconstruction_error']}",
            "threat_indicators": ["error_spike"],
            "mitre_tactics": MITRE_MAP.get(pattern)["tactics"],
            "mitre_techniques": MITRE_MAP.get(pattern)["techniques"],
            "retrieved_context": self._rag_search(pattern),
        }
        return fb



# Convenience
def explain_anomaly(payload):
    return GenAIExplainer().explain(payload)
