import whisper
import torch
import numpy as np
import re
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings("ignore")


class SpeechRecognizer:
    def __init__(self, model_size: str = "base", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Whisper {model_size} model on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print(f"Whisper model loaded successfully.")
        
        self.model_size = model_size
        
    def transcribe(
        self, 
        audio_path: str, 
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict[str, any]:
        # Transcribe with Whisper (with word-level timestamps for better pause detection)
        result = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            verbose=False,
            word_timestamps=True
        )
        
        transcription = result['text'].strip()
        detected_language = result.get('language', 'unknown')
        segments = result.get('segments', [])
        
        analysis = self._analyze_transcription(transcription, segments)
        
        duration = analysis['duration'] if analysis['duration'] > 0 else 1.0
        kopparapu_features = self._extract_kopparapu_features(transcription, duration)
        kopparapu_score = self._calculate_kopparapu_score(kopparapu_features)
        
        return {
            'transcription': transcription,
            'language': detected_language,
            'segments': segments,
            'word_count': analysis['word_count'],
            'duration': analysis['duration'],
            'speech_rate': analysis['speech_rate'],
            'pause_patterns': analysis['pause_patterns'],
            'filler_words': analysis['filler_words'],
            'kopparapu_features': kopparapu_features,
            'kopparapu_score': kopparapu_score,
            'kopparapu_classification': 'read' if kopparapu_score >= 0.5 else 'spontaneous',
            'interpretation': self._interpret_speech_patterns(analysis, kopparapu_features, kopparapu_score)
        }
    
    def _analyze_transcription(self, text: str, segments: List[Dict]) -> Dict:
        words = text.split()
        word_count = len(words)
        
        duration = 0
        if segments:
            duration = segments[-1]['end'] - segments[0]['start']
        
        speech_rate = (word_count / duration * 60) if duration > 0 else 0
        

        filler_words_list = [
            ('um', r'\bum\b'), ('uh', r'\buh\b'), ('er', r'\ber\b'), 
            ('ah', r'\bah\b'), ('like', r'\blike\b'), ('you know', r'\byou know\b'),
            ('i mean', r'\bi mean\b'), ('actually', r'\bactually\b'),
            ('basically', r'\bbasically\b'), ('literally', r'\bliterally\b'),
            ('so', r'\bso\b'), ('well', r'\bwell\b'), ('okay', r'\bokay\b'),
            ('hmm', r'\bhmm+\b'), ('mm', r'\bmm+\b')
        ]
        
        text_lower = text.lower()
        filler_count = {}
        total_fillers = 0
        
        for filler_name, filler_pattern in filler_words_list:
            matches = re.findall(filler_pattern, text_lower, re.IGNORECASE)
            count = len(matches)
            if count > 0:
                filler_count[filler_name] = count
                total_fillers += count
        
        filler_ratio = total_fillers / word_count if word_count > 0 else 0
        
        pause_patterns = self._analyze_pauses(segments)
        
        return {
            'word_count': word_count,
            'duration': duration,
            'speech_rate': speech_rate,
            'filler_words': {
                'count': total_fillers,
                'ratio': filler_ratio,
                'details': filler_count
            },
            'pause_patterns': pause_patterns
        }
    
    def _analyze_pauses(self, segments: List[Dict]) -> Dict:
        pauses = []
        
        if len(segments) >= 2:
            for i in range(len(segments) - 1):
                pause = segments[i + 1]['start'] - segments[i]['end']
                if pause > 0.05:  # Consider pauses > 50ms (lowered threshold)
                    pauses.append(pause)
        
        for segment in segments:
            if 'words' in segment and len(segment['words']) > 1:
                words = segment['words']
                for i in range(len(words) - 1):
                    if 'start' in words[i] and 'end' in words[i] and 'start' in words[i+1]:
                        pause = words[i + 1]['start'] - words[i]['end']
                        if pause > 0.15:  # Word-level pauses (>150ms significant)
                            pauses.append(pause)
        
        if not pauses:
            return {
                'avg_pause': 0.0,
                'max_pause': 0.0,
                'num_pauses': 0,
                'pause_variability': 0.0
            }
        
        return {
            'avg_pause': float(np.mean(pauses)),
            'max_pause': float(np.max(pauses)),
            'num_pauses': len(pauses),
            'pause_variability': float(np.std(pauses)) if len(pauses) > 1 else 0.0
        }
    
    def _extract_kopparapu_features(self, text: str, duration_sec: float) -> Dict:
        text = text.strip()
        if len(text) == 0:
            return {
                'alpha_ratio': 0.0,
                'chars_per_word': 0.0,
                'words_per_sec': 0.0,
                'nonalpha_per_sec': 0.0,
                'repetition_count': 0,
                'filler_rate': 0.0
            }
        
        total_chars = len(text)
        alpha_chars = sum(c.isalpha() for c in text)
        nonalpha_chars = total_chars - alpha_chars
        
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        
        words = text.split()
        num_words = max(len(words), 1)
        chars_per_word = alpha_chars / num_words
        
        duration_sec = max(duration_sec, 1e-3)
        words_per_sec = num_words / duration_sec
        nonalpha_per_sec = nonalpha_chars / duration_sec
        
        char_reps = len(re.findall(r'(.)\1{2,}', text))
        
        words_list = text.lower().split()
        word_reps = 0
        for i in range(len(words_list) - 1):
            if words_list[i] == words_list[i + 1] and len(words_list[i]) > 2:
                word_reps += 1
        
        repetition_count = char_reps + word_reps
        
        lower = text.lower()
        filler_patterns = [
            r'\bum\b', r'\buh\b', r'\buhm\b', r'\ber\b', r'\bah\b', 
            r'\blike\b', r'\byou know\b', r'\bi mean\b',
            r'\bactually\b', r'\bbasically\b', r'\bliterally\b',
            r'\bso\b', r'\bwell\b', r'\bokay\b',
            r'\bhmm+\b', r'\bmm+\b', r'\boh\b'
        ]
        filler_count = 0
        for pattern in filler_patterns:
            filler_count += len(re.findall(pattern, lower))
        filler_rate = filler_count / num_words
        
        return {
            'alpha_ratio': float(alpha_ratio),
            'chars_per_word': float(chars_per_word),
            'words_per_sec': float(words_per_sec),
            'nonalpha_per_sec': float(nonalpha_per_sec),
            'repetition_count': int(repetition_count),
            'filler_rate': float(filler_rate)
        }
    
    def _logistic(self, x: float, a: float, b: float) -> float:
        return 1.0 / (1.0 + np.exp(-(x - a) / b))
    
    def _calculate_kopparapu_score(self, features: Dict) -> float:
        f1 = features['chars_per_word']
        L1 = self._logistic(f1, a=5.0, b=1.5)
        
        f2 = features['words_per_sec']
        L2 = self._logistic(f2, a=2.0, b=0.7)
        
        f3_raw = features['nonalpha_per_sec'] + 10.0 * features['filler_rate']
        L3 = self._logistic(-f3_raw, a=0.0, b=1.0)
        
        score = 0.4 * L1 + 0.4 * L2 + 0.2 * L3
        
        return float(score)
    
    def _interpret_speech_patterns(self, analysis: Dict, kopparapu_features: Dict = None, kopparapu_score: float = None) -> str:
        filler_ratio = analysis['filler_words']['ratio']
        pause_patterns = analysis['pause_patterns']
        speech_rate = analysis['speech_rate']
        
        interpretation = "**Overall Assessment:**\n\n"
        
        spontaneity_score = 0
        indicators = []
        
        if filler_ratio > 0.03:
            spontaneity_score += 1
            indicators.append(f"Filler words present ({filler_ratio*100:.1f}%)")
        
        if pause_patterns['pause_variability'] > 0.5:
            spontaneity_score += 1
            indicators.append(f"Irregular pause patterns (variability: {pause_patterns['pause_variability']:.2f})")
        
        if 120 <= speech_rate <= 180:
            spontaneity_score += 1
            indicators.append(f"Natural speech rate ({speech_rate:.1f} words/min)")
            
        if spontaneity_score >= 2:
            interpretation += "✓ **Speech patterns suggest spontaneous, natural speaking.**\n\n"
            if indicators:
                interpretation += "Key indicators:\n"
                for indicator in indicators:
                    interpretation += f"- {indicator}\n"
        else:
            interpretation += "⚠ **Speech patterns suggest potentially scripted or read speech.**\n\n"
            if filler_ratio < 0.02:
                interpretation += "- Very low filler word usage\n"
            if pause_patterns['pause_variability'] < 0.3:
                interpretation += "- Regular, consistent pause patterns\n"
            if speech_rate > 180:
                interpretation += "- Fast, steady speaking rate\n"
        
        return interpretation
    
    def get_detailed_segments(self, audio_path: str) -> List[Dict]:
        result = self.model.transcribe(audio_path, word_timestamps=True, verbose=False)
        return result.get('segments', [])


if __name__ == "__main__":
    recognizer = SpeechRecognizer(model_size="base")
    print(f"Speech recognizer initialized with {recognizer.model_size} model")
    print(f"Device: {recognizer.device}")

