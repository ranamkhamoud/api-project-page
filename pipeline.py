import time
from audio_classifier import AudioClassifier
from speech_recognizer import SpeechRecognizer
from text_analyzer import TextAuthenticityAnalyzer


# main pipeline class that orchestrates all analysis components
class AuthenticityDetectionPipeline:
    def __init__(
        self,
        audio_model_path=None,
        whisper_model_size="base",
        device=None,
        ai_detection_threshold=0.78
    ):
        print("\nstarting up the pipeline...")
        
        # load the cnn-based audio classifier
        print("loading audio classifier...")
        self.audio_classifier = AudioClassifier(
            model_path=audio_model_path,
            device=device
        )
        
        # load whisper model for speech-to-text
        print("loading whisper...")
        self.speech_recognizer = SpeechRecognizer(
            model_size=whisper_model_size,
            device=device
        )
        
        # load text analyzer for ai detection
        print("loading text analyzer...")
        self.text_analyzer = TextAuthenticityAnalyzer(device=device, ai_threshold=ai_detection_threshold)
        
        print("pipeline ready\n")
    
    # main analysis function
    def analyze_audio(self, audio_path, language=None):
        print("\nrunning analysis...")
        
        start_time = time.time()
        
        # stage 1: classify audio using cnn
        print("step 1: cnn classification")
        audio_results = self.audio_classifier.classify(audio_path)
        print(f"done - got {audio_results['classification']} ({audio_results['confidence']*100:.1f}%)")
        
        # stage 2: transcribe and analyze speech patterns
        print("\nstep 2: speech analysis")
        asr_results = self.speech_recognizer.transcribe(audio_path, language=language)
        print(f"done - {asr_results['word_count']} words, {asr_results['kopparapu_classification']}")
        
        # stage 3: analyze transcribed text for ai patterns
        print("\nstep 3: text analysis")
        text_results = self.text_analyzer.analyze(asr_results['transcription'])
        print(f"done - authenticity {text_results['authenticity_score']*100:.1f}%")
        
        # stage 4: combine all results into final assessment
        print("\nstep 4: final assessment")
        final_assessment = self.generate_final_assessment(
            audio_results,
            asr_results,
            text_results
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"finished in {elapsed_time:.2f}s\n")
        
        return {
            'audio_classification': audio_results,
            'speech_recognition': asr_results,
            'text_analysis': text_results,
            'text_authenticity': text_results,
            'final_assessment': final_assessment,
            'processing_time': elapsed_time
        }
    
    # combine scores from all components into final verdict
    def generate_final_assessment(
        self,
        audio_results,
        asr_results,
        text_results
    ):

        if audio_results['classification'] == 'spontaneous':
            audio_score = audio_results['confidence']
        else:
            audio_score = 1.0 - audio_results['confidence']
        
        speech_pattern_score = 1.0 - asr_results['kopparapu_score']
        
        # filler words indicate spontaneous speech
        filler_ratio = asr_results['filler_words']['ratio']
        filler_score = min(1.0, filler_ratio / 0.05)
        
        # pause variability, higher = more spontaneous
        pause_var = asr_results['pause_patterns']['pause_variability']
        pause_score = min(1.0, pause_var / 0.5)
        
        # text authenticity from ai detector
        text_auth_score = text_results['authenticity_score']
        
        # get additional linguistic features
        kf = asr_results['kopparapu_features']
        
        # speech rate variability
        rate_var = kf.get('speech_rate_variability', 0.0)
        rate_var_score = min(1.0, rate_var / 0.15)
        
        # pause regularity, lower = more spontaneous
        pause_reg = kf.get('pause_regularity', 0.5)
        pause_reg_score = 1.0 - pause_reg
        
        # self-corrections indicate spontaneous speech
        corrections = kf.get('self_correction_count', 0)
        correction_score = min(1.0, corrections / 2.0)

        # calculate weighted composite score
        # weights: cnn+prosody=15%, linguistic=35%, ai detection=50%
        composite_score = (
            audio_score * 0.15 +
            speech_pattern_score * 0.25 +
            filler_score * 0.05 +
            pause_score * 0.03 +
            rate_var_score * 0.02 +
            text_auth_score * 0.50
        )
        
        # determine verdict based on composite score
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
        
        # check cnn classification
        if audio_results['classification'] == 'read':
            concerns.append(f"CNN detected read speech pattern ({audio_results['confidence']*100:.0f}% confidence)")
        else:
            strengths.append(f"CNN detected spontaneous speech ({audio_results['confidence']*100:.0f}% confidence)")
        
        # check linguistic analysis
        if asr_results['kopparapu_classification'] == 'read':
            concerns.append(f"Linguistic analysis suggests read speech (score: {asr_results['kopparapu_score']:.2f})")
        else:
            strengths.append(f"Linguistic analysis suggests spontaneous speech (score: {asr_results['kopparapu_score']:.2f})")
        
        # check filler words
        filler_ratio = asr_results['filler_words']['ratio']
        if filler_ratio < 0.02:
            concerns.append(f"Low filler word usage ({filler_ratio*100:.1f}%) suggests scripted speech")
        else:
            strengths.append(f"Natural filler word usage ({filler_ratio*100:.1f}%) indicates spontaneity")
        
        # check pause patterns
        if asr_results['pause_patterns']['pause_variability'] < 0.3:
            concerns.append("Regular pause patterns suggest reading at punctuation")
        else:
            strengths.append("Irregular pause patterns indicate spontaneous thinking")
        
        # check ai detection
        if text_results['ai_detection']['ai_generated']:
            concerns.append(f"AI-generated text detected ({text_results['ai_detection']['confidence']*100:.0f}% probability)")
        
        # check text originality
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
    print("setting up pipeline...")
    model_path = "spectrogram_cnn_3s_window.pth"
    pipeline = AuthenticityDetectionPipeline(
        audio_model_path=model_path,
        whisper_model_size="base"
    )
    print("ready to go")
