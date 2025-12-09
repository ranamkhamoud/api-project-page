import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np


# basic building block for our resnet-style cnn
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


# main cnn model for speech style classification
class SpeechStyleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SpeechStyleCNN, self).__init__()
        
        # initial conv layer, takes 3 channel input (rgb spectrogram)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(64, 64, 2, stride=1)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def make_layer(self, in_channels, out_channels, blocks, stride=1):
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
    
    def forward(self, x):
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


# main classifier class that combines cnn with acoustic feature analysis
class AudioClassifier:  
    AVAILABLE_MODELS = {
        '3s_window': 'model_checkpoint/spectrogram_cnn_3s_window.pth',
        '4s_window': 'model_checkpoint/spectrogram_cnn_4s_window.pth',
    }
    
    @classmethod
    def get_model_path(cls, model_name='3s_window'):
        import os
        if model_name not in cls.AVAILABLE_MODELS:
            print(f"model not found: {model_name}")
            return None
        return os.path.join(os.path.dirname(__file__), cls.AVAILABLE_MODELS[model_name])
    
    def __init__(self, model_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # initialize the cnn model
        self.model = SpeechStyleCNN().to(self.device)
        
        if model_path is None:
            import os
            model_path = os.path.join(os.path.dirname(__file__), 'spectrogram_cnn_3s_window.pth')
        
        # load pre-trained weights
        try:
            print(f"loading model from {model_path}")
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
            print(f"model loaded ok")
        except FileNotFoundError:
            print(f"cant find model at {model_path}")
            print("check if the file exists")
        except Exception as e:
            print(f"error loading model: {e}")
        
        self.model.eval()
        
        self.sample_rate = 16000
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        
    # extract mel spectrogram from audio file
    def extract_mel_spectrogram(self, audio_path, window_size=3.0):
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        window_samples = int(window_size * sr)
        
        if len(audio) > window_samples * 1.5:
            hop_samples = window_samples // 2
            windows = []
            for start in range(0, len(audio) - window_samples, hop_samples):
                window = audio[start:start + window_samples]
                windows.append(window)
            
            if len(audio) > window_samples:
                windows.append(audio[-window_samples:])
            
            # compute mel spectrogram for each window
            mel_specs = []
            for window in windows[:5]: 
                mel_spec = librosa.feature.melspectrogram(
                    y=window,
                    sr=sr,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )
                mel_specs.append(mel_spec)
            
            # average the spectrograms
            mel_spec = np.mean(mel_specs, axis=0)
        else:
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
    
    # extract acoustic features from audio
    def extract_acoustic_features(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        features = {}
        
        # tempo/rhythm estimation
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        features['tempo'] = float(tempo)
        
        # pitch tracking
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        # calculate pitch statistics
        if pitch_values:
            features['pitch_mean'] = float(np.mean(pitch_values))
            features['pitch_std'] = float(np.std(pitch_values))
            features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
        else:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_range'] = 0.0
        
        # energy/loudness features
        rms = librosa.feature.rms(y=audio)[0]
        features['energy_mean'] = float(np.mean(rms))
        features['energy_std'] = float(np.std(rms))
        
        # zero crossing rate - indicates voice quality
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # spectral centroid - brightness of sound
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        return features
    
    # compute prosody scores from acoustic features
    # uses thresholds calibrated from training data
    def compute_prosody_scores(self, features):
        individual_scores = {}

        # spectral centroid variability
        sc_std = features['spectral_centroid_std']
        if sc_std >= 1080:
            spectral_score = 0.9  # strongly indicates read
        elif sc_std >= 1040:
            spectral_score = 0.7
        elif sc_std >= 1000:
            spectral_score = 0.5
        elif sc_std >= 970:
            spectral_score = 0.3
        else:
            spectral_score = 0.1  # strongly spontaneous
        
        individual_scores['spectral_variability'] = {
            'score': spectral_score,
            'value': sc_std,
            'interpretation': 'high variability (read)' if spectral_score > 0.6 else 'low variability (spontaneous)' if spectral_score < 0.4 else 'moderate'
        }
        
        # zero crossing rate
        zcr = features['zcr_mean']
        if zcr >= 0.125:
            zcr_score = 0.9
        elif zcr >= 0.110:
            zcr_score = 0.7
        elif zcr >= 0.100:
            zcr_score = 0.5
        elif zcr >= 0.092:
            zcr_score = 0.3
        else:
            zcr_score = 0.1
        
        individual_scores['zcr_mean'] = {
            'score': zcr_score,
            'value': zcr,
            'interpretation': 'high ZCR (read)' if zcr_score > 0.6 else 'low ZCR (spontaneous)' if zcr_score < 0.4 else 'moderate'
        }
        
        # energy level
        energy = features['energy_mean']
        if energy < 0.055:
            energy_score = 0.85
        elif energy < 0.062:
            energy_score = 0.65
        elif energy < 0.070:
            energy_score = 0.4
        else:
            energy_score = 0.15
        
        individual_scores['energy_level'] = {
            'score': energy_score,
            'value': energy,
            'interpretation': 'low energy (read)' if energy_score > 0.6 else 'high energy (spontaneous)' if energy_score < 0.4 else 'moderate'
        }
        
        # pitch range feature
        pitch_range = features.get('pitch_range', 3828)
        if pitch_range < 3815:
            pitch_range_score = 0.7
        elif pitch_range < 3828:
            pitch_range_score = 0.5
        else:
            pitch_range_score = 0.3
        
        individual_scores['pitch_range'] = {
            'score': pitch_range_score,
            'value': pitch_range,
            'interpretation': 'narrow (read)' if pitch_range_score > 0.6 else 'wide (spontaneous)' if pitch_range_score < 0.4 else 'moderate'
        }
        
        # energy variability
        energy_std = features.get('energy_std', 0.047)
        if energy_std < 0.042:
            energy_std_score = 0.7
        elif energy_std < 0.048:
            energy_std_score = 0.5
        else:
            energy_std_score = 0.3
        
        individual_scores['energy_std'] = {
            'score': energy_std_score,
            'value': energy_std,
            'interpretation': 'steady (read)' if energy_std_score > 0.6 else 'variable (spontaneous)' if energy_std_score < 0.4 else 'moderate'
        }
        
        # zcr variability
        zcr_std = features.get('zcr_std', 0.111)
        if zcr_std >= 0.115:
            zcr_std_score = 0.7
        elif zcr_std >= 0.105:
            zcr_std_score = 0.5
        else:
            zcr_std_score = 0.3
        
        individual_scores['zcr_std'] = {
            'score': zcr_std_score,
            'value': zcr_std,
            'interpretation': 'variable ZCR (read)' if zcr_std_score > 0.6 else 'steady ZCR (spontaneous)' if zcr_std_score < 0.4 else 'moderate'
        }
        
        # weights based on feature importance from analysis
        weights = {
            'spectral_variability': 0.30,
            'zcr_mean': 0.25,
            'energy_level': 0.20,
            'pitch_range': 0.10,
            'energy_std': 0.08,
            'zcr_std': 0.07,
        }
        
        # calculate weighted overall score
        overall_score = (
            spectral_score * weights['spectral_variability'] +
            zcr_score * weights['zcr_mean'] +
            energy_score * weights['energy_level'] +
            pitch_range_score * weights['pitch_range'] +
            energy_std_score * weights['energy_std'] +
            zcr_std_score * weights['zcr_std']
        )
        
        # determine classification based on thresholds
        if overall_score > 0.58:
            classification = 'read'
            confidence = 0.5 + (overall_score - 0.5) * 0.9
        elif overall_score < 0.42:
            classification = 'spontaneous'
            confidence = 0.5 + (0.5 - overall_score) * 0.9
        else:
            classification = 'read' if overall_score >= 0.50 else 'spontaneous'
            confidence = 0.5 + abs(overall_score - 0.5) * 0.6
        
        return {
            'classification': classification,
            'confidence': min(0.95, confidence),
            'overall_score': overall_score,
            'individual_scores': individual_scores
        }
    
    def classify(self, audio_path):
        # extract mel spectrogram for cnn
        mel_spec = self.extract_mel_spectrogram(audio_path)
        
        # convert to tensor and add batch dimension
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).to(self.device)
        
        # get cnn predictions
        with torch.no_grad():
            logits = self.model(mel_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            cnn_confidence = probabilities[0, predicted_class].item()
            
            print(f"cnn logits: {logits[0].cpu().numpy()}")
            print(f"cnn probs: read={probabilities[0, 0].item():.3f}, spont={probabilities[0, 1].item():.3f}")
            print(f"cnn says: {['read', 'spontaneous'][predicted_class]} (conf={cnn_confidence:.3f})")
        
        # extract acoustic features for prosody analysis
        acoustic_features = self.extract_acoustic_features(audio_path)
        
        # compute prosody-based scores
        prosody_scores = self.compute_prosody_scores(acoustic_features)
        prosody_classification = prosody_scores['classification']
        prosody_confidence = prosody_scores['confidence']
        
        cnn_class_name = 'read' if predicted_class == 0 else 'spontaneous'
        read_prob = probabilities[0, 0].item()
        
        print(f"cnn: {cnn_class_name}")
        print(f"prosody: {prosody_classification} (conf={prosody_confidence:.2f})")
        
        # combine cnn and prosody
        final_classification = prosody_classification
        final_confidence = prosody_confidence
        
        # boost confidence when both methods agree
        if cnn_class_name == prosody_classification:
            final_confidence = min(0.95, prosody_confidence * 1.15)
        elif read_prob > 0.85 and cnn_class_name == 'read':
            if prosody_confidence < 0.65:
                final_classification = 'read'
                final_confidence = 0.55
        elif read_prob < 0.10 and cnn_class_name == 'spontaneous':
            if prosody_confidence < 0.65:
                final_classification = 'spontaneous'
                final_confidence = 0.55
        
        return {
            'classification': final_classification,
            'confidence': float(final_confidence),
            'cnn_classification': cnn_class_name,
            'cnn_confidence': float(cnn_confidence),
            'prosody_classification': prosody_classification,
            'prosody_confidence': float(prosody_confidence),
            'prosody_scores': prosody_scores['individual_scores'],
            'acoustic_features': acoustic_features,
            'interpretation': self.interpret_classification(
                final_classification, final_confidence, 
                cnn_class_name, cnn_confidence,
                prosody_classification, prosody_confidence,
                prosody_scores, acoustic_features
            )
        }
    
    # generate interpretation of classification
    def interpret_classification(
        self, 
        final_class, 
        final_confidence,
        cnn_class,
        cnn_confidence,
        prosody_class,
        prosody_confidence,
        prosody_scores,
        features
    ):   
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
    print("\navailable models:")
    for name, filename in AudioClassifier.AVAILABLE_MODELS.items():
        print(f"  {name}: {filename}")
    
    print("\nmodel arch:")
    print(classifier.model)
