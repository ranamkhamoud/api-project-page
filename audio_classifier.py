import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from typing import Dict

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        return out


class SpeechStyleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SpeechStyleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class AudioClassifier:  
    AVAILABLE_MODELS = {
        '3s_window': 'model_checkpoints/spectrogram_cnn_3s_window.pth',
        # '4s_window': 'model_checkpoints/spectrogram_cnn_4s_window.pth',
        # '4s_488x488': 'model_checkpoints/spectrogram_cnn_4s_window_488_x_488.pth'
    }
    
    @classmethod
    def get_model_path(cls, model_name: str = '3s_window') -> str:
        import os
        if model_name not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.AVAILABLE_MODELS.keys())}")
        return os.path.join(os.path.dirname(__file__), cls.AVAILABLE_MODELS[model_name])
    
    def __init__(self, model_path: str = None, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = SpeechStyleCNN().to(self.device)
        
        if model_path is None:
            import os
            model_path = os.path.join(os.path.dirname(__file__), 'spectrogram_cnn_3s_window (1).pth')
        
        try:
            print(f"Attempting to load model from: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✓ Successfully loaded trained model from: {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model file exists.")
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {e}")
        
        self.model.eval()
        
        self.sample_rate = 16000
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        
    def extract_mel_spectrogram(self, audio_path: str, window_size: float = 3.0) -> np.ndarray:
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # If audio is longer than window_size, take multiple windows and average
        window_samples = int(window_size * sr)
        
        if len(audio) > window_samples * 1.5:  # If significantly longer
            # Split into overlapping windows
            hop_samples = window_samples // 2
            windows = []
            for start in range(0, len(audio) - window_samples, hop_samples):
                window = audio[start:start + window_samples]
                windows.append(window)
            
            # Also add the last window
            if len(audio) > window_samples:
                windows.append(audio[-window_samples:])
            
            # Compute mel spectrogram for each window and average
            mel_specs = []
            for window in windows[:5]:  # Limit to 5 windows to avoid too much computation
                mel_spec = librosa.feature.melspectrogram(
                    y=window,
                    sr=sr,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                mel_specs.append(mel_spec)
            
            # Average the spectrograms
            mel_spec = np.mean(mel_specs, axis=0)
        else:
            # Pad or use as-is for short audio
            if len(audio) < window_samples:
                audio = np.pad(audio, (0, window_samples - len(audio)), mode='constant')
            else:
                audio = audio[:window_samples]
            
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        mel_spec_3ch = np.stack([mel_spec_norm, mel_spec_norm, mel_spec_norm], axis=0)
        
        return mel_spec_3ch
    
    def extract_acoustic_features(self, audio_path: str) -> Dict[str, float]:
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        features = {}
        
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        features['tempo'] = float(tempo)
        
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            features['pitch_mean'] = float(np.mean(pitch_values))
            features['pitch_std'] = float(np.std(pitch_values))
            features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
        else:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_range'] = 0.0
        
        rms = librosa.feature.rms(y=audio)[0]
        features['energy_mean'] = float(np.mean(rms))
        features['energy_std'] = float(np.std(rms))
        
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        return features
    
    def _compute_prosody_scores(self, features: Dict[str, float]) -> Dict:
        individual_scores = {}

        sc_std = features['spectral_centroid_std']
        if sc_std >= 1100:
            spectral_score = 0.9  # Strongly indicates read
        elif sc_std >= 1050:
            spectral_score = 0.7  # Likely read
        elif sc_std >= 1000:
            spectral_score = 0.5  # Borderline
        elif sc_std >= 950:
            spectral_score = 0.3  # Likely spontaneous
        else:
            spectral_score = 0.1  # Strongly spontaneous
        
        individual_scores['spectral_variability'] = {
            'score': spectral_score,
            'value': sc_std,
            'interpretation': 'high variability (read)' if spectral_score > 0.6 else 'low variability (spontaneous)' if spectral_score < 0.4 else 'moderate'
        }
        
        zcr = features['zcr_mean']
        if zcr >= 0.13:
            zcr_score = 0.9  # Strongly indicates read
        elif zcr >= 0.115:
            zcr_score = 0.7  # Likely read
        elif zcr >= 0.105:
            zcr_score = 0.5  # Borderline
        elif zcr >= 0.095:
            zcr_score = 0.3  # Likely spontaneous
        else:
            zcr_score = 0.1  # Strongly spontaneous
        
        individual_scores['zcr_mean'] = {
            'score': zcr_score,
            'value': zcr,
            'interpretation': 'high ZCR (read)' if zcr_score > 0.6 else 'low ZCR (spontaneous)' if zcr_score < 0.4 else 'moderate'
        }
        
        # 3. Energy mean (separation: 0.69)
        # Read: 0.06 avg, Spontaneous: 0.06 avg but spontaneous tends higher
        # Threshold: ~0.06, read < threshold
        energy = features['energy_mean']
        if energy < 0.055:
            energy_score = 0.8  # Low energy -> likely read
        elif energy < 0.065:
            energy_score = 0.5  # Moderate
        elif energy < 0.075:
            energy_score = 0.3  # Higher energy -> likely spontaneous
        else:
            energy_score = 0.1  # High energy -> spontaneous
        
        individual_scores['energy_level'] = {
            'score': energy_score,
            'value': energy,
            'interpretation': 'low energy (read)' if energy_score > 0.6 else 'high energy (spontaneous)' if energy_score < 0.4 else 'moderate'
        }
        
        # 4. Tempo (separation: 0.22) - less discriminative but still useful
        # Read: 122 avg, Spontaneous: 125 avg
        tempo = features['tempo']
        if tempo < 115:
            tempo_score = 0.7  # Slower -> could be read (more deliberate)
        elif tempo < 125:
            tempo_score = 0.5  # Moderate
        else:
            tempo_score = 0.3  # Faster -> could be spontaneous
        
        individual_scores['tempo'] = {
            'score': tempo_score,
            'value': tempo,
            'interpretation': 'slow (read)' if tempo_score > 0.6 else 'fast (spontaneous)' if tempo_score < 0.4 else 'moderate'
        }
        
        # Optimized weights based on feature separation scores
        weights = {
            'spectral_variability': 0.40,  
            'zcr_mean': 0.30,            
            'energy_level': 0.20,         
            'tempo': 0.10                
        }
        
        overall_score = (
            spectral_score * weights['spectral_variability'] +
            zcr_score * weights['zcr_mean'] +
            energy_score * weights['energy_level'] +
            tempo_score * weights['tempo']
        )
        
        if overall_score > 0.60:
            classification = 'read'
            confidence = 0.5 + (overall_score - 0.5) * 0.8
        elif overall_score < 0.40:
            classification = 'spontaneous'
            confidence = 0.5 + (0.5 - overall_score) * 0.8
        else:
            classification = 'read' if overall_score >= 0.5 else 'spontaneous'
            confidence = 0.5 + abs(overall_score - 0.5) * 0.6
        
        return {
            'classification': classification,
            'confidence': min(0.95, confidence),
            'overall_score': overall_score,
            'individual_scores': individual_scores
        }
    
    def classify(self, audio_path: str) -> Dict[str, any]:
        mel_spec = self.extract_mel_spectrogram(audio_path)
        
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(mel_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            cnn_confidence = probabilities[0, predicted_class].item()
            
            print(f"CNN Logits: {logits[0].cpu().numpy()}")
            print(f"CNN Probabilities: Class 0 (read)={probabilities[0, 0].item():.3f}, Class 1 (spontaneous)={probabilities[0, 1].item():.3f}")
            print(f"CNN Prediction: Class {predicted_class} ({['read', 'spontaneous'][predicted_class]}) with confidence {cnn_confidence:.3f}")
        
        acoustic_features = self.extract_acoustic_features(audio_path)
        
        prosody_scores = self._compute_prosody_scores(acoustic_features)
        prosody_classification = prosody_scores['classification']
        prosody_confidence = prosody_scores['confidence']
        
        # Model mapping: Class 0 = read, Class 1 = spontaneous
        cnn_class_name = 'read' if predicted_class == 0 else 'spontaneous'
        print(f"CNN classification: {cnn_class_name}")
        print(f"Prosody classification: {prosody_classification} (conf={prosody_confidence:.2f})")
        

        cnn_score = 1.0 if cnn_class_name == 'read' else 0.0
        prosody_score = 1.0 if prosody_classification == 'read' else 0.0
        

        weighted_score = (
            cnn_score * cnn_confidence * 0.4 +
            prosody_score * prosody_confidence * 0.6
        ) / (cnn_confidence * 0.4 + prosody_confidence * 0.6)
        
        if weighted_score > 0.5:
            final_classification = 'read'
            final_confidence = 0.5 + (weighted_score - 0.5)
        else:
            final_classification = 'spontaneous'
            final_confidence = 0.5 + (0.5 - weighted_score)
        
        final_confidence = min(0.95, final_confidence)
        
        return {
            'classification': final_classification,
            'confidence': float(final_confidence),
            'cnn_classification': cnn_class_name,
            'cnn_confidence': float(cnn_confidence),
            'prosody_classification': prosody_classification,
            'prosody_confidence': float(prosody_confidence),
            'prosody_scores': prosody_scores['individual_scores'],
            'acoustic_features': acoustic_features,
            'interpretation': self._interpret_classification(
                final_classification, final_confidence, 
                cnn_class_name, cnn_confidence,
                prosody_classification, prosody_confidence,
                prosody_scores, acoustic_features
            )
        }
    
    def _interpret_classification(
        self, 
        final_class: str, 
        final_confidence: float,
        cnn_class: str,
        cnn_confidence: float,
        prosody_class: str,
        prosody_confidence: float,
        prosody_scores: Dict,
        features: Dict
    ) -> str:   
        interpretation = f"## Classification: **{final_class.upper()}** SPEECH\n\n"
        interpretation += f"**Confidence:** {final_confidence*100:.1f}%\n\n"
        
        if final_class == 'read':
            interpretation += "**Description:** The speech exhibits characteristics of read or scripted content. "
            interpretation += "The audio shows consistent prosodic patterns typical of someone reading from prepared text, "
            interpretation += "with steady pacing, uniform intonation, and regular energy levels.\n\n"
        else:
            interpretation += "**Description:** The speech exhibits characteristics of spontaneous speaking. "
            interpretation += "The audio shows natural prosodic variation typical of extemporaneous speech, "
            interpretation += "with variable pacing, dynamic intonation, and natural energy fluctuations.\n\n"
        
        
        return interpretation


if __name__ == "__main__":
    classifier = AudioClassifier()
    print("\nAvailable pre-trained models:")
    for name, filename in AudioClassifier.AVAILABLE_MODELS.items():
        print(f"  - {name}: {filename}")
    
    print("\nModel architecture:")
    print(classifier.model)
