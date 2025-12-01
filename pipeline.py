from typing import Dict, Optional
import time
from audio_classifier import AudioClassifier
from speech_recognizer import SpeechRecognizer
from text_analyzer import TextAuthenticityAnalyzer


class AuthenticityDetectionPipeline:
    def __init__(
        self,
        audio_model_path: Optional[str] = None,
        whisper_model_size: str = "base",
        device: Optional[str] = None,
        ai_detection_threshold: float = 0.78
    ):
        print("\n" + "="*60)
        print("Initializing Multimodal Authenticity Detection Pipeline")
        print("="*60 + "\n")
        
        # Initialize components
        print("📊 Loading Audio Classifier (CNN)...")
        self.audio_classifier = AudioClassifier(
            model_path=audio_model_path,
            device=device
        )
        
        print("\n🎤 Loading Speech Recognizer (Whisper)...")
        self.speech_recognizer = SpeechRecognizer(
            model_size=whisper_model_size,
            device=device
        )
        
        print("\n📝 Loading Text Authenticity Analyzer...")
        self.text_analyzer = TextAuthenticityAnalyzer(device=device, ai_threshold=ai_detection_threshold)
        
        print("\n✅ Pipeline initialization complete!")
        print("="*60 + "\n")
    
    def analyze_audio(self, audio_path: str, language: Optional[str] = None) -> Dict:
        print("\n" + "="*60)
        print("MULTIMODAL AUTHENTICITY ANALYSIS")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        # Stage 1: Audio Classification (CNN-based read vs spontaneous detection)
        print("Stage 1: CNN Audio Classification...")
        print("-" * 40)
        audio_results = self.audio_classifier.classify(audio_path)
        print(f"✓ CNN classification complete")
        print(f"  ## Classification: {audio_results['classification'].upper()}")
        print(f"  Confidence: {audio_results['confidence']*100:.1f}%")
        
        # Stage 2: Speech Analysis (Whisper for linguistic analysis)
        print("\nStage 2: Speech Analysis (Whisper)...")
        print("-" * 40)
        asr_results = self.speech_recognizer.transcribe(audio_path, language=language)
        print(f"✓ Speech analysis complete")
        print(f"  Language: {asr_results['language']}")
        print(f"  Word count: {asr_results['word_count']}")
        print(f"  Kopparapu classification: {asr_results['kopparapu_classification'].upper()}")
        
        # Stage 3: Text Authenticity Analysis
        print("\nStage 3: Analyzing text authenticity...")
        print("-" * 40)
        text_results = self.text_analyzer.analyze(asr_results['transcription'])
        print(f"✓ Text analysis complete")
        print(f"  Authenticity score: {text_results['authenticity_score']*100:.1f}%")
        print(f"  Risk level: {text_results['risk_level'].upper()}")
        
        # Stage 4: Combined Assessment
        print("\nStage 4: Generating final assessment...")
        print("-" * 40)
        final_assessment = self._generate_final_assessment(
            audio_results,
            asr_results,
            text_results
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"✓ Analysis complete in {elapsed_time:.2f} seconds")
        print("\n" + "="*60 + "\n")
        
        return {
            'audio_classification': audio_results,
            'speech_recognition': asr_results,
            'text_authenticity': text_results,
            'final_assessment': final_assessment,
            'processing_time': elapsed_time
        }
    
    def _generate_final_assessment(
        self,
        audio_results: Dict,
        asr_results: Dict,
        text_results: Dict
    ) -> Dict:

        # CNN score: spontaneous = authentic (high), read = inauthentic (low)
        if audio_results['classification'] == 'spontaneous':
            audio_score = audio_results['confidence']
        else:  # read
            audio_score = 1.0 - audio_results['confidence']
        
        # Kopparapu score: 0=spontaneous, 1=read
        # Invert so spontaneous (low kopparapu) = high authenticity
        speech_pattern_score = 1.0 - asr_results['kopparapu_score']
        
        # Filler words: higher ratio = more spontaneous = more authentic
        filler_ratio = asr_results['filler_words']['ratio']
        filler_score = min(1.0, filler_ratio / 0.05)  # Normalize: 5%+ = max score
        
        # Pause variability: higher = more spontaneous = more authentic
        pause_var = asr_results['pause_patterns']['pause_variability']
        pause_score = min(1.0, pause_var / 0.5)  # Normalize: 0.5+ = max score
        
        text_auth_score = text_results['authenticity_score']

        composite_score = (
            audio_score * 0.15 +            # CNN - weakest component
            speech_pattern_score * 0.20 +   # Kopparapu linguistic
            filler_score * 0.10 +           # Filler word ratio
            pause_score * 0.05 +            # Pause variability
            text_auth_score * 0.50          # Text authenticity - strongest signal
        )
        
        if composite_score >= 0.7:
            verdict = "AUTHENTIC"
            risk = "low"
            recommendation = "Response appears genuine with strong authenticity indicators."
        elif composite_score >= 0.5:
            verdict = "LIKELY AUTHENTIC"
            risk = "moderate"
            recommendation = "Response shows mostly authentic characteristics but has some concerns."
        elif composite_score >= 0.3:
            verdict = "QUESTIONABLE"
            risk = "high"
            recommendation = "Response has multiple authenticity concerns. Further investigation recommended."
        else:
            verdict = "LIKELY INAUTHENTIC"
            risk = "critical"
            recommendation = "Response shows strong indicators of inauthenticity. Manual review required."
        
        concerns = []
        strengths = []
        
        if audio_results['classification'] == 'read':
            concerns.append(f"CNN detected read speech pattern ({audio_results['confidence']*100:.0f}% confidence)")
        else:
            strengths.append(f"CNN detected spontaneous speech ({audio_results['confidence']*100:.0f}% confidence)")
        
        if asr_results['kopparapu_classification'] == 'read':
            concerns.append(f"Linguistic analysis suggests read speech (score: {asr_results['kopparapu_score']:.2f})")
        else:
            strengths.append(f"Linguistic analysis suggests spontaneous speech (score: {asr_results['kopparapu_score']:.2f})")
        
        filler_ratio = asr_results['filler_words']['ratio']
        if filler_ratio < 0.02:
            concerns.append(f"Low filler word usage ({filler_ratio*100:.1f}%) suggests scripted speech")
        else:
            strengths.append(f"Natural filler word usage ({filler_ratio*100:.1f}%) indicates spontaneity")
        
        if asr_results['pause_patterns']['pause_variability'] < 0.3:
            concerns.append("Regular pause patterns suggest reading at punctuation")
        else:
            strengths.append("Irregular pause patterns indicate spontaneous thinking")
        
        if text_results['ai_detection']['ai_generated']:
            concerns.append(f"AI-generated text detected ({text_results['ai_detection']['confidence']*100:.0f}% probability)")
        
        if text_results['authenticity_score'] > 0.7:
            strengths.append("Text shows strong originality indicators")

        
        return {
            'verdict': verdict,
            'risk_level': risk,
            'composite_authenticity_score': float(composite_score),
            'concerns': concerns,
            'strengths': strengths,
            'recommendation': recommendation,
        }
    
if __name__ == "__main__":
    # Example usage
    print("Initializing Authenticity Detection Pipeline...")
    model_path = "spectrogram_cnn_3s_window.pth"
    pipeline = AuthenticityDetectionPipeline(
        audio_model_path=model_path,
        whisper_model_size="base"
    )
    print("\nPipeline ready for audio analysis.")
