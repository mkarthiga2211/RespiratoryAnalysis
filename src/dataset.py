"""
Dataset Loader for Respiratory Sound Database
Handles ICBHI Respiratory Sound Dataset with proper patient-level separation.
"""

import os
import numpy as np
import pandas as pd
import librosa
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from preprocessing import AudioPreprocessor, FeatureNormalizer, DataAugmenter


class RespiratoryDataset:
    """
    Respiratory Sound Dataset loader with patient-level separation.
    Implements 5-fold cross-validation as described in the paper.
    """
    
    def __init__(self, 
                 data_dir: str,
                 annotation_file: str = None,
                 target_sr: int = 16000,
                 n_mels: int = 40,
                 segment_length: float = 0.2):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Directory containing audio files
            annotation_file: Path to annotation file (if available)
            target_sr: Target sampling rate
            n_mels: Number of Mel filter banks
            segment_length: Segment length in seconds
        """
        self.data_dir = data_dir
        self.annotation_file = annotation_file
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.segment_length = segment_length
        
        # Initialize preprocessor
        self.preprocessor = AudioPreprocessor(target_sr=target_sr, n_mels=n_mels)
        self.augmenter = DataAugmenter()
        
        # Load file list and labels
        self.file_list = []
        self.labels = []
        self.patient_ids = []
        self._load_file_list()
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
    def _load_file_list(self):
        """
        Load list of audio files and their labels.
        Handles ICBHI dataset structure.
        """
        # Look for audio files in subdirectories
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    
                    # Extract label from filename (ICBHI format)
                    # Format: [patient_id]_[recording_index]_[chest_location]_[acquisition_mode]_[equipment]_[crackles]_[wheezes].wav
                    parts = file.split('_')
                    
                    if len(parts) >= 2:
                        patient_id = parts[0]
                        
                        # Try to determine diagnosis from filename or use annotation file
                        diagnosis = self._get_diagnosis(file_path, parts)
                        
                        self.file_list.append(file_path)
                        self.labels.append(diagnosis)
                        self.patient_ids.append(patient_id)
        
        print(f"Loaded {len(self.file_list)} audio files")
        print(f"Number of unique patients: {len(set(self.patient_ids))}")
        print(f"Class distribution: {Counter(self.labels)}")
    
    def _get_diagnosis(self, file_path: str, filename_parts: List[str]) -> str:
        """
        Get diagnosis label for a file.
        
        Args:
            file_path: Path to audio file
            filename_parts: Parts of filename
            
        Returns:
            Diagnosis label
        """
        # This is a simplified version - in practice, you'd use the annotation file
        # For ICBHI dataset, diagnosis is typically in a separate file
        
        # Try to infer from filename or use default
        # Common labels: 'Normal', 'COPD', 'Asthma', 'Pneumonia', 'Bronchiectasis', 'Bronchiolitis', 'URTI'
        
        # For now, return a placeholder - in actual implementation,
        # read from the diagnosis file provided with ICBHI dataset
        if len(filename_parts) > 0:
            # Use patient ID to lookup diagnosis
            return 'Unknown'
        return 'Unknown'
    
    def load_diagnosis_file(self, diagnosis_file: str):
        """
        Load diagnosis information from file.
        
        Args:
            diagnosis_file: Path to diagnosis file
        """
        # Load diagnosis file (format depends on dataset)
        # ICBHI provides diagnosis information in a text file
        diagnosis_df = pd.read_csv(diagnosis_file, sep='\t', header=None)
        
        # Map patient IDs to diagnoses
        diagnosis_map = {}
        for _, row in diagnosis_df.iterrows():
            patient_id = str(row[0])
            diagnosis = row[1]
            diagnosis_map[patient_id] = diagnosis
        
        # Update labels
        for i, patient_id in enumerate(self.patient_ids):
            if patient_id in diagnosis_map:
                self.labels[i] = diagnosis_map[patient_id]
        
        # Re-encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        print(f"Updated class distribution: {Counter(self.labels)}")
    
    def extract_features(self, file_path: str) -> np.ndarray:
        """
        Extract Mel-spectrogram features from audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Mel-spectrogram features
        """
        try:
            # Load and preprocess audio
            features = self.preprocessor.preprocess(file_path)
            mel_spec = features['mel_spectrogram']
            
            # Normalize
            normalizer = FeatureNormalizer()
            mel_spec_norm, _, _ = normalizer.z_score_normalize(mel_spec)
            
            return mel_spec_norm
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def create_fixed_size_segments(self, mel_spec: np.ndarray, 
                                   target_shape: Tuple[int, int] = (40, 40)) -> List[np.ndarray]:
        """
        Create fixed-size segments from mel-spectrogram.
        
        Args:
            mel_spec: Mel-spectrogram
            target_shape: Target shape (height, width)
            
        Returns:
            List of fixed-size segments
        """
        segments = []
        height, width = target_shape
        
        # Pad or truncate to target height
        if mel_spec.shape[0] < height:
            pad_height = height - mel_spec.shape[0]
            mel_spec = np.pad(mel_spec, ((0, pad_height), (0, 0)), mode='constant')
        elif mel_spec.shape[0] > height:
            mel_spec = mel_spec[:height, :]
        
        # Create segments along time axis
        hop_length = width // 2  # 50% overlap
        for i in range(0, mel_spec.shape[1] - width, hop_length):
            segment = mel_spec[:, i:i + width]
            
            # Pad if necessary
            if segment.shape[1] < width:
                pad_width = width - segment.shape[1]
                segment = np.pad(segment, ((0, 0), (0, pad_width)), mode='constant')
            
            segments.append(segment)
        
        # If no segments created, pad the entire spectrogram
        if len(segments) == 0:
            padded = np.pad(mel_spec, ((0, 0), (0, width - mel_spec.shape[1])), mode='constant')
            segments.append(padded)
        
        return segments
    
    def prepare_dataset(self, target_shape: Tuple[int, int] = (40, 40),
                       augment: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare entire dataset with features and labels.
        
        Args:
            target_shape: Target shape for spectrograms
            augment: Whether to apply data augmentation
            
        Returns:
            Tuple of (features, labels, patient_ids)
        """
        all_features = []
        all_labels = []
        all_patient_ids = []
        
        for i, file_path in enumerate(self.file_list):
            mel_spec = self.extract_features(file_path)
            
            if mel_spec is not None:
                # Create fixed-size segments
                segments = self.create_fixed_size_segments(mel_spec, target_shape)
                
                for segment in segments:
                    all_features.append(segment)
                    all_labels.append(self.encoded_labels[i])
                    all_patient_ids.append(self.patient_ids[i])
                
                # Data augmentation for minority classes
                if augment:
                    class_label = self.labels[i]
                    class_count = Counter(self.labels)[class_label]
                    
                    # Augment if class is underrepresented
                    if class_count < len(self.file_list) / len(set(self.labels)):
                        audio, sr = librosa.load(file_path, sr=self.target_sr)
                        augmented_audios = self.augmenter.augment(audio, sr)
                        
                        for aug_audio in augmented_audios[1:]:  # Skip original
                            # Extract mel-spectrogram
                            mel_spec_aug = self.preprocessor.compute_mel_spectrogram(aug_audio)
                            normalizer = FeatureNormalizer()
                            mel_spec_aug_norm, _, _ = normalizer.z_score_normalize(mel_spec_aug)
                            
                            segments_aug = self.create_fixed_size_segments(mel_spec_aug_norm, target_shape)
                            
                            for segment in segments_aug:
                                all_features.append(segment)
                                all_labels.append(self.encoded_labels[i])
                                all_patient_ids.append(self.patient_ids[i])
        
        # Convert to arrays and add channel dimension
        X = np.array(all_features)[..., np.newaxis]  # Add channel dimension
        y = np.array(all_labels)
        groups = np.array(all_patient_ids)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        return X, y, groups
    
    def get_cross_validation_splits(self, n_splits: int = 5) -> List[Tuple]:
        """
        Get patient-level cross-validation splits.
        
        Args:
            n_splits: Number of folds
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        X, y, groups = self.prepare_dataset()
        
        group_kfold = GroupKFold(n_splits=n_splits)
        splits = []
        
        for train_idx, val_idx in group_kfold.split(X, y, groups):
            splits.append((train_idx, val_idx))
        
        return splits, X, y


class BinaryClassificationDataset(RespiratoryDataset):
    """
    Binary classification dataset: Healthy vs Diseased
    """
    
    def _get_diagnosis(self, file_path: str, filename_parts: List[str]) -> str:
        """Override to return binary labels."""
        # In actual implementation, check if diagnosis is 'Healthy' or 'Normal'
        # For now, return placeholder
        return 'Unknown'
    
    def prepare_binary_labels(self):
        """Convert multi-class labels to binary (Healthy/Diseased)."""
        binary_labels = []
        for label in self.labels:
            if label.lower() in ['healthy', 'normal']:
                binary_labels.append('Healthy')
            else:
                binary_labels.append('Diseased')
        
        self.labels = binary_labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        print(f"Binary class distribution: {Counter(self.labels)}")


def load_dataset(data_dir: str, 
                 diagnosis_file: str = None,
                 task: str = 'multiclass',
                 target_shape: Tuple[int, int] = (40, 40),
                 augment: bool = False) -> Tuple:
    """
    Load and prepare dataset.
    
    Args:
        data_dir: Directory containing audio files
        diagnosis_file: Path to diagnosis file
        task: 'binary' or 'multiclass'
        target_shape: Target shape for spectrograms
        augment: Whether to apply data augmentation
        
    Returns:
        Dataset object and prepared data
    """
    if task == 'binary':
        dataset = BinaryClassificationDataset(data_dir)
    else:
        dataset = RespiratoryDataset(data_dir)
    
    if diagnosis_file:
        dataset.load_diagnosis_file(diagnosis_file)
    
    if task == 'binary':
        dataset.prepare_binary_labels()
    
    X, y, groups = dataset.prepare_dataset(target_shape=target_shape, augment=augment)
    
    return dataset, X, y, groups


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loader...")
    
    # Create dummy data directory for testing
    import tempfile
    import soundfile as sf
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy audio files
        for i in range(5):
            dummy_audio = np.random.randn(16000)  # 1 second at 16kHz
            sf.write(os.path.join(tmpdir, f'patient_{i}_0_0_0_0_0_0.wav'), 
                    dummy_audio, 16000)
        
        # Test dataset
        dataset = RespiratoryDataset(tmpdir)
        print(f"\nFound {len(dataset.file_list)} files")
        print(f"Labels: {dataset.labels}")
    
    print("\nDataset loader test completed!")
