import whisper
import torch
import numpy as np
import re
import warnings
import librosa
warnings.filterwarnings("ignore")


# main class for speech recognition and analysis
class SpeechRecognizer:
    def __init__(self, model_size="base", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # load whisper model
        print(f"loading whisper {model_size} on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print(f"whisper loaded")
        
        self.model_size = model_size
    
    # check if audio file is valid before processing
    def validate_audio(self, audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr
            
            if duration < 0.1:
                return False, "Audio too short", duration
            
            if np.max(np.abs(audio)) < 0.001:
                return False, "Audio is silent", duration
            
            return True, "Valid", duration
            
        except Exception as e:
            return False, f"Could not load audio file", 0.0
        
    # main transcription function
    def transcribe(self, audio_path, language=None, task="transcribe"):
        is_valid, message, audio_duration = self.validate_audio(audio_path)
        if not is_valid:
            print(f"audio check failed: {message}")
            return self.get_empty_response(message, audio_duration)
        
        # try to transcribe with word timestamps
        try:
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                verbose=False,
                word_timestamps=True,
                fp16=False  
            )
        except (KeyError, RuntimeError) as e:
            error_msg = str(e)
            # handle specific errors
            if "reshape tensor of 0 elements" in error_msg or "ambiguous" in error_msg:
                print(f"audio too short or corrupted maybe")
                return self.get_empty_response("Audio too short or corrupted", audio_duration)
            
            print(f"first try failed, trying again...")
            try:
                result = self.model.transcribe(
                    audio_path,
                    language=language,
                    task=task,
                    verbose=False,
                    word_timestamps=False,
                    fp16=False
                )
            except Exception as e2:
                print(f"couldnt transcribe: {e2}")
                return self.get_empty_response("Transcription failed", audio_duration)
        
        # extract transcription results
        transcription = result['text'].strip()
        detected_language = result.get('language', 'unknown')
        segments = result.get('segments', [])
        
        if not transcription or len(transcription.strip()) == 0:
            print("warning: transcription empty")
            return self.get_empty_response("No speech detected in audio", audio_duration)
        
        analysis = self.analyze_transcription(transcription, segments)
        
        # extract features for read/spontaneous detection
        duration = analysis['duration'] if analysis['duration'] > 0 else 1.0
        kopparapu_features = self.extract_kopparapu_features(
            transcription, duration, segments, analysis['pause_patterns']
        )
        kopparapu_score = self.calculate_kopparapu_score(kopparapu_features)
        
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
            'interpretation': self.interpret_speech_patterns(analysis, kopparapu_features, kopparapu_score)
        }
    
    # return empty response when transcription fails
    def get_empty_response(self, reason, duration=0.0):
        return {
            'transcription': f"[Error: {reason}]",
            'language': 'unknown',
            'segments': [],
            'word_count': 0,
            'duration': duration,
            'speech_rate': 0.0,
            'pause_patterns': {
                'avg_pause': 0.0,
                'max_pause': 0.0,
                'num_pauses': 0,
                'pause_variability': 0.0
            },
            'filler_words': {
                'count': 0,
                'ratio': 0.0,
                'details': {}
            },
            'kopparapu_features': {
                'chars_per_word': 0.0,
                'words_per_sec': 0.0,
                'nonalpha_per_sec': 0.0,
                'filler_rate': 0.0,
                'repetition_count': 0,
                'alpha_ratio': 0.0
            },
            'kopparapu_score': 0.5,
            'kopparapu_classification': 'unknown',
            'interpretation': f"Could not process audio: {reason}\n\nTips:\n- Make sure audio is at least 1 second\n- Check that there is actual speech\n- Try a different audio file"
        }
    
    # analyze transcription for various speech metrics
    def analyze_transcription(self, text, segments):
        words = text.split()
        word_count = len(words)
        
        # calculate duration from segments
        duration = 0
        if segments:
            duration = segments[-1]['end'] - segments[0]['start']
        
        # calculate speaking rate (words per minute)
        speech_rate = (word_count / duration * 60) if duration > 0 else 0

        # list of filler words to detect
        filler_words_list = [
            ('um', r'\bum\b'), ('uh', r'\buh\b'), ('er', r'\ber\b'), 
            ('ah', r'\bah\b'), ('like', r'\blike\b'), ('you know', r'\byou know\b'),
            ('i mean', r'\bi mean\b'), ('actually', r'\bactually\b'),
            ('basically', r'\bbasically\b'), ('literally', r'\bliterally\b'),
            ('so', r'\bso\b'), ('well', r'\bwell\b'), ('okay', r'\bokay\b'),
            ('hmm', r'\bhmm+\b'), ('mm', r'\bmm+\b')
        ]
        
        # count filler words
        text_lower = text.lower()
        filler_count = {}
        total_fillers = 0
        
        for filler_name, filler_pattern in filler_words_list:
            matches = re.findall(filler_pattern, text_lower, re.IGNORECASE)
            count = len(matches)
            if count > 0:
                filler_count[filler_name] = count
                total_fillers += count
        
        # calculate filler ratio
        filler_ratio = total_fillers / word_count if word_count > 0 else 0
        
        # analyze pause patterns
        pause_patterns = self.analyze_pauses(segments)
        
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
    
    # extract pause timing information from segments
    def analyze_pauses(self, segments):
        pauses = []
        
        # find pauses between segments
        if len(segments) >= 2:
            for i in range(len(segments) - 1):
                pause = segments[i + 1]['start'] - segments[i]['end']
                if pause > 0.05:  # pauses > 50ms
                    pauses.append(pause)
        
        # find pauses between words within segments
        for segment in segments:
            if 'words' in segment and len(segment['words']) > 1:
                words = segment['words']
                for i in range(len(words) - 1):
                    if 'start' in words[i] and 'end' in words[i] and 'start' in words[i+1]:
                        pause = words[i + 1]['start'] - words[i]['end']
                        if pause > 0.15:  # word-level pauses > 150ms
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
    
    def extract_kopparapu_features(self, text, duration_sec, segments=None, pause_patterns=None):
        text = text.strip()
        # handle empty text
        if len(text) == 0:
            return {
                'alpha_ratio': 0.0,
                'chars_per_word': 0.0,
                'words_per_sec': 0.0,
                'nonalpha_per_sec': 0.0,
                'repetition_count': 0,
                'filler_rate': 0.0,
                'pause_regularity': 0.5,
                'speech_rate_variability': 0.0,
                'sentence_length_variance': 0.0,
                'self_correction_count': 0
            }
        
        # count character types
        total_chars = len(text)
        alpha_chars = sum(c.isalpha() for c in text)
        nonalpha_chars = total_chars - alpha_chars
        
        # ratio of alphabetic characters
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        
        # average word length
        words = text.split()
        num_words = max(len(words), 1)
        chars_per_word = alpha_chars / num_words
        
        # speaking rate features
        duration_sec = max(duration_sec, 1e-3)
        words_per_sec = num_words / duration_sec
        nonalpha_per_sec = nonalpha_chars / duration_sec
        
        # detect character repetitions like "sooo" or "ummmm"
        char_reps = len(re.findall(r'(.)\1{2,}', text))
        
        # detect word repetitions like "I I think"
        words_list = text.lower().split()
        word_reps = 0
        for i in range(len(words_list) - 1):
            if words_list[i] == words_list[i + 1] and len(words_list[i]) > 2:
                word_reps += 1
        
        repetition_count = char_reps + word_reps
        
        # count filler words
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
        
        # pause regularity, read speech has regular pauses at punctuation
        pause_regularity = 0.5
        if pause_patterns and pause_patterns.get('num_pauses', 0) > 2:
            pause_var = pause_patterns.get('pause_variability', 0.5)
            # low variability = regular pauses = likely read
            pause_regularity = max(0.0, min(1.0, 1.0 - (pause_var / 0.6)))
        
        # speech rate variability across segments
        speech_rate_variability = self.compute_rate_variability(segments) if segments else 0.0
        
        # sentence length variance - uniform = likely read
        sentence_length_variance = self.compute_sentence_variance(text)
        
        # count self-corrections and false starts
        self_correction_patterns = [
            r'\bwait\b', r'\bsorry\b', r'\bno\s*,?\s*I\b',
            r'\bactually\s*,?\s*no\b', r'\blet me\b', r'\bwhat I meant\b',
            r'\bI meant\b', r'\bhold on\b', r'\bwhat was I\b', r'\bor rather\b'
        ]
        self_correction_count = 0
        for pattern in self_correction_patterns:
            self_correction_count += len(re.findall(pattern, lower))
        
        return {
            'alpha_ratio': float(alpha_ratio),
            'chars_per_word': float(chars_per_word),
            'words_per_sec': float(words_per_sec),
            'nonalpha_per_sec': float(nonalpha_per_sec),
            'repetition_count': int(repetition_count),
            'filler_rate': float(filler_rate),
            'pause_regularity': float(pause_regularity),
            'speech_rate_variability': float(speech_rate_variability),
            'sentence_length_variance': float(sentence_length_variance),
            'self_correction_count': int(self_correction_count)
        }
    
    # compute variability in speaking rate across segments
    def compute_rate_variability(self, segments):
        if not segments or len(segments) < 3:
            return 0.0
        
        segment_rates = []
        for seg in segments:
            duration = seg.get('end', 0) - seg.get('start', 0)
            if duration > 0.3:  # only segments > 300ms
                words_in_seg = len(seg.get('text', '').split())
                rate = words_in_seg / duration
                if rate > 0:
                    segment_rates.append(rate)
        
        if len(segment_rates) < 3:
            return 0.0
        
        # calculate coefficient of variation
        mean_rate = np.mean(segment_rates)
        std_rate = np.std(segment_rates)
        
        cv = std_rate / mean_rate if mean_rate > 0 else 0
        return float(min(1.0, cv / 0.5))
    
    # compute variance in sentence lengths
    def compute_sentence_variance(self, text):
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        # get word counts per sentence
        lengths = [len(s.split()) for s in sentences]
        mean_len = np.mean(lengths)
        std_len = np.std(lengths)
        
        # coefficient of variation normalized
        cv = std_len / mean_len if mean_len > 0 else 0
        return float(min(1.0, cv / 0.6))
    
    # logistic function for smooth score transitions
    def logistic(self, x, a, b):
        return 1.0 / (1.0 + np.exp(-(x - a) / b))
    
    # calculate overall score for read vs spontaneous
    def calculate_kopparapu_score(self, features):
        # l1: vocabulary complexity - higher = more formal = read
        f1 = features['chars_per_word']
        L1 = self.logistic(f1, a=4.8, b=1.2)
        
        # l2: speaking rate - faster, steadier = read
        f2 = features['words_per_sec']
        L2 = self.logistic(f2, a=2.2, b=0.6)
        
        # l3: disfluency - less disfluency = more read
        disfluency = (
            features['nonalpha_per_sec'] + 
            8.0 * features['filler_rate'] + 
            0.5 * features['repetition_count']
        )
        L3 = self.logistic(-disfluency, a=0.0, b=0.8)
        
        # l4: pause regularity - regular = read
        L4 = features.get('pause_regularity', 0.5)
        
        # l5: rate variability - low = read
        rate_var = features.get('speech_rate_variability', 0.0)
        L5 = 1.0 - rate_var
        
        # l6: sentence variance - uniform = read
        sent_var = features.get('sentence_length_variance', 0.0)
        L6 = 1.0 - sent_var
        
        # l7: self-corrections - fewer = read
        corrections = features.get('self_correction_count', 0)
        L7 = self.logistic(-corrections, a=0.0, b=1.5)
        
        # weighted combination
        score = (
            0.15 * L1 +  # vocabulary complexity
            0.15 * L2 +  # speaking rate
            0.15 * L3 +  # disfluency
            0.20 * L4 +  # pause regularity
            0.15 * L5 +  # rate variability
            0.10 * L6 +  # sentence uniformity
            0.10 * L7    # self-corrections
        )
        
        return float(score)
    
    # generate human-readable interpretation of speech patterns
    def interpret_speech_patterns(self, analysis, kopparapu_features=None, kopparapu_score=None):
        filler_ratio = analysis['filler_words']['ratio']
        pause_patterns = analysis['pause_patterns']
        speech_rate = analysis['speech_rate']
        
        interpretation = "**Overall Assessment:**\n\n"
        
        # calculate spontaneity score
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
        
        # generate interpretation based on score
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
    
    # get detailed segment information
    def get_detailed_segments(self, audio_path):
        result = self.model.transcribe(audio_path, word_timestamps=True, verbose=False)
        return result.get('segments', [])


if __name__ == "__main__":
    recognizer = SpeechRecognizer(model_size="base")
    print(f"speech recognizer ready, model={recognizer.model_size}")
    print(f"using {recognizer.device}")
