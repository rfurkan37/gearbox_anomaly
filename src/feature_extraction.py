import numpy as np
import librosa

class GearboxFeatureExtractor:
    """Simplified feature extractor optimized for Pico deployment."""
    
    def __init__(self):
        self.sr = 16000  # Fixed sample rate
        self.frame_length = 256  # Reduced from 1024
        self.hop_length = 128    # Reduced from 512
        self.n_mels = 32        # Reduced from 80
        self.segment_length = 8  # Reduced from 32
        self.n_bands = 4        # Number of frequency bands
        
    def extract_features(self, audio_file: str) -> np.ndarray:
        """Extract minimal features optimized for Pico."""
        # Load and resample audio
        y, sr = librosa.load(audio_file, sr=self.sr, mono=True)
        
        # Calculate simple features
        # 1. RMS energy
        rms = librosa.feature.rms(
            y=y,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        # 2. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=y,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        
        # 3. Simplified spectral features (4 frequency bands)
        spec = np.abs(librosa.stft(y, n_fft=self.frame_length, hop_length=self.hop_length))
        
        # Calculate frequency band energies
        n_freqs = spec.shape[0]
        band_size = n_freqs // self.n_bands
        spec_bands = np.array([
            np.mean(spec[i:i + band_size], axis=0)
            for i in range(0, n_freqs - band_size + 1, band_size)
        ])
        
        # Combine features
        features = np.vstack([
            rms,
            zcr,
            spec_bands
        ])
        
        # Normalize
        features = (features - np.mean(features, axis=1, keepdims=True)) / (
            np.std(features, axis=1, keepdims=True) + 1e-6)
        
        # Frame into segments
        segments = self._create_segments(features)
        
        return segments
    
    def _create_segments(self, features: np.ndarray) -> np.ndarray:
        """Create smaller segments from features."""
        segments = []
        
        for i in range(0, features.shape[1] - self.segment_length + 1, 
                      self.segment_length // 2):
            segment = features[:, i:i + self.segment_length]
            segments.append(segment.flatten())
        
        return np.array(segments)
    
    def get_feature_dim(self) -> int:
        """Get dimension of feature vector."""
        # 6 features (RMS + ZCR + 4 spectral bands) * segment_length
        return 6 * self.segment_length