import re
import requests
from typing import Dict, List, Tuple, Optional
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification
)
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

try:
    from plagiarism_detection import ai_plagiarism_detection
    DESKLIB_AVAILABLE = True
except ImportError:
    DESKLIB_AVAILABLE = False
    print("Warning: plagiarism_detection module not found. Using fallback AI detection.")




class AITextDetector:
    def __init__(self, device: str = None, threshold: float = 0.78):
        self.threshold = threshold
        
        if not DESKLIB_AVAILABLE:
            print("Warning: plagiarism_detection module not found. AI detection will not be available.")
            print("Ensure plagiarism_detection.py is in the same directory.")
            self.available = False
        else:
            print(f"Using Desklib AI text detector (threshold: {self.threshold})")
            self.available = True
    
    def detect_ai_text(self, text: str) -> Dict:

        if not self.available:
            # Return neutral result if Desklib not available
            return {
                'ai_generated': False,
                'confidence': 0.5,
                'indicators': [],
                'interpretation': "AI detection not available. Install plagiarism_detection module.",
                'model_used': 'N/A (module not found)'
            }
        
        # Use Desklib AI detector
        try:
            probability, ai_detected = ai_plagiarism_detection(
                text, 
                threshold=self.threshold, 
                show_results=False
            )
            
            return {
                'ai_generated': ai_detected,
                'confidence': float(probability),
                'indicators': self._identify_ai_indicators(probability),
                'interpretation': self._interpret_ai_detection(probability),
                'model_used': 'Desklib AI Detector v1.01'
            }
        except Exception as e:
            print(f"Error in AI detection: {e}")
            return {
                'ai_generated': False,
                'confidence': 0.5,
                'indicators': [],
                'interpretation': f"AI detection error: {str(e)}",
                'model_used': 'Error'
            }
    
    
    def _identify_ai_indicators(self, probability: float) -> List[str]:
        indicators = []
        
        if probability > 0.9:
            indicators.append("Very high AI probability (>90%)")
        elif probability > 0.7:
            indicators.append("High AI probability (70-90%)")
        elif probability > self.threshold:
            indicators.append(f"AI detected above threshold ({self.threshold*100:.0f}%)")
        
        return indicators
    
    def _interpret_ai_detection(self, score: float) -> str:
        interpretation = f"**AI-Generated Text Detection:**\n\n"
        interpretation += f"- AI Probability Score: {score*100:.1f}%\n"
        interpretation += f"- Detection Threshold: {self.threshold*100:.0f}%\n"
        
        return interpretation


class TextAuthenticityAnalyzer:

    def __init__(self, device: str = None, ai_threshold: float = 0.78):

        self.ai_detector = AITextDetector(device=device, threshold=ai_threshold)
        
    def analyze(self, text: str) -> Dict:
        # Run AI detection
        ai_results = self.ai_detector.detect_ai_text(text)
        
        # Calculate overall authenticity score based on AI detection
        ai_penalty = ai_results['confidence']
        authenticity_score = 1.0 - ai_penalty
        
        # Determine overall assessment
        if authenticity_score < 0.3:
            overall_assessment = "HIGH RISK: Strong AI-generated text indicators"
            risk_level = "high"
        elif authenticity_score < 0.5:
            overall_assessment = "MODERATE RISK: Likely AI-generated"
            risk_level = "moderate"
        elif authenticity_score < 0.7:
            overall_assessment = "LOW RISK: Some AI characteristics"
            risk_level = "low"
        else:
            overall_assessment = "AUTHENTIC: Text appears human-written"
            risk_level = "minimal"
        
        return {
            'authenticity_score': float(authenticity_score),
            'risk_level': risk_level,
            'overall_assessment': overall_assessment,
            'ai_detection': ai_results,
        }
    

if __name__ == "__main__":
    # Example usage
    analyzer = TextAuthenticityAnalyzer()
    print("Text authenticity analyzer initialized.")
    print("Components: Plagiarism Detector + AI Text Detector")

