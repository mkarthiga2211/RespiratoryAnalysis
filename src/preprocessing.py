"""
Data Preprocessing Module for Respiratory Sound Analysis
Implements: Resampling, ZCR, STFT, MFCCs, Mel-Spectrogram extraction
"""

import numpy as np
import librosa
import librosa.display
from scipy import signal
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """
    Audio preprocessing class for respiratory sound analysis.
    Implements all preprocessing steps mentioned in the paper.
    """
    
    def __init__(self, target_sr: int = 16000, n_mels: int = 40, n_mfcc: int = 13):
        """
        Initialize the preprocessor with parameters from the paper.
        
        Args:
            target_sr: Target sampling rate (16kHz as per paper)
            n_mels: Number of Mel filter banks (40 as per paper)
            n_mfcc: Number of MFCC coefficients (13 as per paper)
        """
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        Resample audio to target sampling rate.
        
        Args:
            audio: Input audio signal
            orig_sr: Original sampling rate
            
        Returns:
            Resampled audio signal
        """
        if orig_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)
        return audio
    
    def compute_zcr(self, audio: np.ndarray, frame_length: int = 2048, 
                    hop_length: int = 512) -> np.ndarray:
        """
        Compute Zero-Crossing Rate (ZCR) for respiratory sound analysis.
        ZCR helps detect inhalation and exhalation phases.
        
        Args:
            audio: Input audio signal
            frame_length: Length of each frame
            hop_length: Hop length between frames
            
        Returns:
            ZCR values
        """
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=frame_length, hop_length=hop_length
        )[0]
        return zcr
    
    def compute_stft(self, audio: np.ndarray, n_fft: int = 512, 
                     hop_length: int = 256, win_length: int = 400) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform (STFT).
        Paper specifies: 25ms Hamming window, 50% overlap, FFT size 512
        
        Args:
            audio: Input audio signal
            n_fft: FFT size (512 as per paper)
            hop_length: Hop length (50% overlap)
            win_length: Window length (25ms = 400 samples at 16kHz)
            
        Returns:
            STFT matrix
        """
        # Use Hamming window as specified in paper
        window = signal.windows.hamming(win_length)
        
        stft = librosa.stft(
            audio, n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window
        )
        return np.abs(stft)
    
    def compute_mel_spectrogram(self, audio: np.ndarray, 
                                 n_fft: int = 512, hop_length: int = 256) -> np.ndarray:
        """
        Compute Mel-Spectrogram with 40 Mel filter banks.
        
        Args:
            audio: Input audio signal
            n_fft: FFT size
            hop_length: Hop length
            
        Returns:
            Mel-spectrogram (40 x T as per paper)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.target_sr, n_fft=n_fft,
            hop_length=hop_length, n_mels=self.n_mels
        )
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def compute_mfcc(self, audio: np.ndarray, n_fft: int = 512, 
                     hop_length: int = 256) -> np.ndarray:
        """
        Compute Mel-Frequency Cepstral Coefficients (MFCCs).
        
        Args:
            audio: Input audio signal
            n_fft: FFT size
            hop_length: Hop length
            
        Returns:
            MFCC coefficients
        """
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.target_sr, n_mfcc=self.n_mfcc,
            n_fft=n_fft, hop_length=hop_length
        )
        return mfccs
    
    def preprocess(self, audio_path: str) -> dict:
        """
        Complete preprocessing pipeline for a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing all preprocessed features
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Step 1: Resampling
        audio_resampled = self.resample(audio, sr)
        
        # Step 2: Compute ZCR
        zcr = self.compute_zcr(audio_resampled)
        
        # Step 3: Compute STFT
        stft = self.compute_stft(audio_resampled)
        
        # Step 4: Compute Mel-Spectrogram
        mel_spectrogram = self.compute_mel_spectrogram(audio_resampled)
        
        # Step 5: Compute MFCCs
        mfccs = self.compute_mfcc(audio_resampled)
        
        return {
            'audio': audio_resampled,
            'zcr': zcr,
            'stft': stft,
            'mel_spectrogram': mel_spectrogram,
            'mfccs': mfccs,
            'sample_rate': self.target_sr
        }


class FeatureNormalizer:
    """
    Feature normalization class implementing Z-score and Min-Max scaling.
    """
    
    @staticmethod
    def z_score_normalize(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Z-score normalization (standardization).
        Formula: x_norm = (x - μ) / σ
        
        Args:
            features: Input features
            
        Returns:
            Tuple of (normalized_features, mean, std)
        """
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        std = np.where(std == 0, 1e-8, std)  # Avoid division by zero
        normalized = (features - mean) / std
        return normalized, mean, std
    
    @staticmethod
    def min_max_normalize(features: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Min-Max scaling.
        Formula: x_norm = (x - min) / (max - min) * (max_range - min_range) + min_range
        
        Args:
            features: Input features
            feature_range: Target range for scaling
            
        Returns:
            Tuple of (normalized_features, min_vals, max_vals)
        """
        min_vals = np.min(features, axis=0, keepdims=True)
        max_vals = np.max(features, axis=0, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1e-8, range_vals)  # Avoid division by zero
        
        normalized = (features - min_vals) / range_vals
        normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        
        return normalized, min_vals, max_vals
    
    @staticmethod
    def apply_z_score(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Apply Z-score normalization using pre-computed statistics."""
        std = np.where(std == 0, 1e-8, std)
        return (features - mean) / std
    
    @staticmethod
    def apply_min_max(features: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray,
                      feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """Apply Min-Max scaling using pre-computed statistics."""
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1e-8, range_vals)
        normalized = (features - min_vals) / range_vals
        normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        return normalized


class FeatureSegmenter:
    """
    Fixed-Length Segmentation (Windowing) for audio signals.
    """
    
    def __init__(self, window_length: float = 0.2, hop_length: float = 0.1, 
                 sample_rate: int = 16000):
        """
        Initialize segmenter.
        
        Args:
            window_length: Window length in seconds (default 200ms)
            hop_length: Hop length in seconds (default 100ms for 50% overlap)
            sample_rate: Sample rate of the audio
        """
        self.window_samples = int(window_length * sample_rate)
        self.hop_samples = int(hop_length * sample_rate)
        self.sample_rate = sample_rate
    
    def segment(self, audio: np.ndarray) -> list:
        """
        Segment audio into fixed-length windows with overlap.
        
        Args:
            audio: Input audio signal
            
        Returns:
            List of segmented windows
        """
        segments = []
        for i in range(0, len(audio) - self.window_samples, self.hop_samples):
            segment = audio[i:i + self.window_samples]
            segments.append(segment)
        return segments
    
    def segment_spectrogram(self, spectrogram: np.ndarray, window_frames: int = 40) -> list:
        """
        Segment spectrogram into fixed-length windows.
        
        Args:
            spectrogram: Input spectrogram (freq_bins x time_frames)
            window_frames: Number of time frames per segment
            
        Returns:
            List of segmented spectrograms
        """
        segments = []
        hop_frames = window_frames // 2  # 50% overlap
        
        for i in range(0, spectrogram.shape[1] - window_frames, hop_frames):
            segment = spectrogram[:, i:i + window_frames]
            segments.append(segment)
        
        return segments


class DataAugmenter:
    """
    Data augmentation techniques for handling class imbalance.
    Implements: Pitch shifting, Time-stretching, Gaussian noise injection
    """
    
    @staticmethod
    def pitch_shift(audio: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
        """
        Apply pitch shifting (±2 semitones as per paper).
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            n_steps: Number of semitones to shift
            
        Returns:
            Pitch-shifted audio
        """
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def time_stretch(audio: np.ndarray, rate: float = 1.1) -> np.ndarray:
        """
        Apply time stretching (0.9 to 1.1 as per paper).
        
        Args:
            audio: Input audio signal
            rate: Stretching rate
            
        Returns:
            Time-stretched audio
        """
        return librosa.effects.time_stretch(audio, rate=rate)
    
    @staticmethod
    def add_gaussian_noise(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """
        Add Gaussian noise at specified SNR (20 dB as per paper).
        
        Args:
            audio: Input audio signal
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Noisy audio
        """
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise
    
    def augment(self, audio: np.ndarray, sr: int, 
                pitch_range: Tuple[float, float] = (-2, 2),
                stretch_range: Tuple[float, float] = (0.9, 1.1)) -> list:
        """
        Apply all augmentation techniques.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
            pitch_range: Range for pitch shifting
            stretch_range: Range for time stretching
            
        Returns:
            List of augmented audio samples
        """
        augmented = [audio]  # Original
        
        # Pitch shift (positive and negative)
        augmented.append(self.pitch_shift(audio, sr, pitch_range[1]))
        augmented.append(self.pitch_shift(audio, sr, pitch_range[0]))
        
        # Time stretch
        augmented.append(self.time_stretch(audio, stretch_range[1]))
        augmented.append(self.time_stretch(audio, stretch_range[0]))
        
        # Gaussian noise
        augmented.append(self.add_gaussian_noise(audio))
        
        return augmented


def preprocess_pipeline(audio_path: str, preprocessor: AudioPreprocessor,
                       normalizer: FeatureNormalizer, segmenter: FeatureSegmenter,
                       normalize_type: str = 'z_score') -> dict:
    """
    Complete preprocessing pipeline for a single audio file.
    
    Args:
        audio_path: Path to audio file
        preprocessor: AudioPreprocessor instance
        normalizer: FeatureNormalizer instance
        segmenter: FeatureSegmenter instance
        normalize_type: Type of normalization ('z_score' or 'min_max')
        
    Returns:
        Dictionary containing preprocessed and normalized features
    """
    # Preprocess
    features = preprocessor.preprocess(audio_path)
    
    # Normalize Mel-spectrogram
    mel_spec = features['mel_spectrogram']
    if normalize_type == 'z_score':
        mel_spec_norm, mean, std = normalizer.z_score_normalize(mel_spec)
        features['mel_spectrogram_norm'] = mel_spec_norm
        features['mel_spectrogram_mean'] = mean
        features['mel_spectrogram_std'] = std
    else:
        mel_spec_norm, min_vals, max_vals = normalizer.min_max_normalize(mel_spec)
        features['mel_spectrogram_norm'] = mel_spec_norm
        features['mel_spectrogram_min'] = min_vals
        features['mel_spectrogram_max'] = max_vals
    
    # Normalize MFCCs
    mfccs = features['mfccs']
    if normalize_type == 'z_score':
        mfccs_norm, mean, std = normalizer.z_score_normalize(mfccs)
        features['mfccs_norm'] = mfccs_norm
        features['mfccs_mean'] = mean
        features['mfccs_std'] = std
    else:
        mfccs_norm, min_vals, max_vals = normalizer.min_max_normalize(mfccs)
        features['mfccs_norm'] = mfccs_norm
        features['mfccs_min'] = min_vals
        features['mfccs_max'] = max_vals
    
    # Segment audio
    segments = segmenter.segment(features['audio'])
    features['segments'] = segments
    
    # Segment spectrogram
    mel_segments = segmenter.segment_spectrogram(features['mel_spectrogram_norm'])
    features['mel_spectrogram_segments'] = mel_segments
    
    return features


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing preprocessing module...")
    
    # Initialize components
    preprocessor = AudioPreprocessor(target_sr=16000, n_mels=40, n_mfcc=13)
    normalizer = FeatureNormalizer()
    segmenter = FeatureSegmenter(window_length=0.2, hop_length=0.1, sample_rate=16000)
    
    print("Preprocessing module initialized successfully!")
    print(f"Target sample rate: {preprocessor.target_sr} Hz")
    print(f"Mel filter banks: {preprocessor.n_mels}")
    print(f"MFCC coefficients: {preprocessor.n_mfcc}")
