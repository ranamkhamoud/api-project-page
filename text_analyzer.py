import warnings
warnings.filterwarnings("ignore")

# try to import the desklib ai detector
try:
    from plagiarism_detection import ai_plagiarism_detection
    DESKLIB_AVAILABLE = True
except ImportError:
    DESKLIB_AVAILABLE = False
    print("warning: cant find plagiarism_detection module")


# class for detecting ai-generated text
class AITextDetector:
    def __init__(self, device=None, threshold=0.78):
        self.threshold = threshold
        
        if not DESKLIB_AVAILABLE:
            print("warning: plagiarism module missing, ai detection wont work")
            print("make sure plagiarism_detection.py is in same folder")
            self.available = False
        else:
            print(f"using desklib detector (threshold={self.threshold})")
            self.available = True
    
    # main detection function
    def detect_ai_text(self, text):

        # return neutral result if detector not available
        if not self.available:
            return {
                'ai_generated': False,
                'confidence': 0.5,
                'indicators': [],
                'interpretation': "AI detection not available. Install plagiarism_detection module.",
                'model_used': 'N/A (module not found)'
            }
        
        # run detection using desklib model
        try:
            probability, ai_detected = ai_plagiarism_detection(
                text, 
                threshold=self.threshold, 
                show_results=False
            )
            
            return {
                'ai_generated': ai_detected,
                'confidence': float(probability),
                'indicators': self.identify_ai_indicators(probability),
                'interpretation': self.interpret_ai_detection(probability),
                'model_used': 'Desklib AI Detector v1.01'
            }
        except Exception as e:
            print(f"ai detection failed: {e}")
            return {
                'ai_generated': False,
                'confidence': 0.5,
                'indicators': [],
                'interpretation': "Could not run AI detection",
                'model_used': 'Error'
            }
    
    # identify specific indicators based on probability
    def identify_ai_indicators(self, probability):
        indicators = []
        
        if probability > 0.9:
            indicators.append("Very high AI probability (>90%)")
        elif probability > 0.7:
            indicators.append("High AI probability (70-90%)")
        elif probability > self.threshold:
            indicators.append(f"AI detected above threshold ({self.threshold*100:.0f}%)")
        
        return indicators
    
    def interpret_ai_detection(self, score):
        interpretation = f"**AI-Generated Text Detection:**\n\n"
        interpretation += f"- AI Probability Score: {score*100:.1f}%\n"
        interpretation += f"- Detection Threshold: {self.threshold*100:.0f}%\n"
        
        return interpretation


class TextAuthenticityAnalyzer:

    def __init__(self, device=None, ai_threshold=0.78):
        # initialize ai detector
        self.ai_detector = AITextDetector(device=device, threshold=ai_threshold)
        
    # analyze text for authenticity
    def analyze(self, text):
        # run ai detection
        ai_results = self.ai_detector.detect_ai_text(text)
        
        # calculate authenticity score
        ai_penalty = ai_results['confidence']
        authenticity_score = 1.0 - ai_penalty
        
        # determine risk level based on authenticity
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
    analyzer = TextAuthenticityAnalyzer()
    print("text analyzer ready")
    print("using plagiarism detector + ai detector")
