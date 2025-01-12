import numpy as np
import librosa

class GearboxFeatureExtractor:
    """Feature extractor optimized for gearbox sounds."""
    
    def __init__(self):
        self.sr = 16000  # Fixed sample rate
        self.n_mels = 80
        self.n_fft = 1024
        self.hop_length = 512
        self.segment_length = 32
        
    def extract_features(self, audio_file: str) -> np.ndarray:
        """Extract optimized features from audio file."""
        # Load and resample audio
        y, sr = librosa.load(audio_file, sr=self.sr, mono=True)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=20,
            fmax=8000,
            power=2.0
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-6)
        
        # Frame into segments
        segments = self._create_segments(log_mel_spec)
        
        return segments
    
    def _create_segments(self, features: np.ndarray) -> np.ndarray:
        """Create overlapping segments from features."""
        segments = []
        
        for i in range(0, features.shape[1] - self.segment_length + 1, 
                      self.segment_length // 2):
            segment = features[:, i:i + self.segment_length]
            segments.append(segment.flatten())
        
        return np.array(segments)
    
    def get_feature_dim(self) -> int:
        """Get dimension of feature vector."""
        return self.n_mels * self.segment_length