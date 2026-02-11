# Implementation Guide

This document provides detailed information about the implementation of the paper "Enhanced Respiratory Condition Diagnosis Through Mel Spectrogram Features and Metaheuristic-Optimized CNNs".

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Descriptions](#module-descriptions)
3. [Key Implementation Details](#key-implementation-details)
4. [Paper-to-Code Mapping](#paper-to-code-mapping)
5. [Reproducibility Notes](#reproducibility-notes)

## Architecture Overview

The implementation follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Pipeline (main.py)                   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌────────────────┐    ┌─────────────────┐
│Preprocessing │    │  GWO Optimizer │    │ Training & Eval │
│(preprocessing│    │(grey_wolf_    │    │    (train.py)   │
│    .py)      │    │  optimizer.py) │    │                 │
└──────────────┘    └────────────────┘    └─────────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌────────────────┐    ┌─────────────────┐
│   Dataset    │    │     Models     │    │     Utils       │
│  (dataset.py)│    │  (models.py)   │    │   (utils.py)    │
└──────────────┘    └────────────────┘    └─────────────────┘
```

## Module Descriptions

### 1. preprocessing.py

Implements all preprocessing steps described in Section III of the paper.

**Key Classes:**
- `AudioPreprocessor`: Main preprocessing pipeline
- `FeatureNormalizer`: Z-score and Min-Max normalization
- `FeatureSegmenter`: Fixed-length windowing
- `DataAugmenter`: Data augmentation techniques

**Key Methods:**
- `resample()`: Converts audio to 16kHz (Section III.B.1)
- `compute_zcr()`: Zero-crossing rate (Section III.B.5)
- `compute_stft()`: STFT with 25ms Hamming window (Section III.B.2)
- `compute_mel_spectrogram()`: 40 Mel filter banks (Section III.B.4)
- `compute_mfcc()`: 13 MFCC coefficients (Section III.B.3)

### 2. grey_wolf_optimizer.py

Implements the Grey Wolf Optimizer for hyperparameter optimization (Section III.E).

**Key Classes:**
- `GreyWolfOptimizer`: Main GWO implementation

**Key Methods:**
- `_calculate_A_C()`: Computes coefficient vectors A and C
- `_update_position()`: Updates wolf positions
- `optimize()`: Main optimization loop

**Hyperparameter Search Space (Table 2):**
```python
search_space = {
    'learning_rate': (1e-5, 1e-2, 'float'),
    'batch_size': (32, 64, 'int'),
    'conv_filters': (16, 128, 'int'),
    'kernel_size': (3, 7, 'int'),
    'dropout_rate': (0.2, 0.5, 'float'),
    'dense_units': (64, 512, 'int')
}
```

### 3. models.py

Implements all model architectures (Section III.F).

**Key Classes:**
- `AttentionLayer`: Attention mechanism
- `OptimizedCNN`: Main optimized CNN architecture
- `TransferLearningCNN`: Transfer learning with VGG-like base
- `BaselineModels`: RNN, LSTM, Simple CNN for comparison

**Optimized CNN Architecture:**
```
Input (40x40x1)
    │
    ├── Conv2D(64, 3x3) → BN → MaxPool
    ├── Conv2D(128, 3x3) → BN → MaxPool
    ├── Conv2D(256, 3x3) → BN → MaxPool
    ├── Conv2D(256, 3x3) → BN → MaxPool
    │
    ├── Attention Layer (optional)
    │
    ├── Dense(256) → BN → Dropout(0.3)
    │
    └── Output (num_classes)
```

### 4. dataset.py

Handles dataset loading and preparation (Section III.A).

**Key Classes:**
- `RespiratoryDataset`: Main dataset loader
- `BinaryClassificationDataset`: Binary classification variant

**Key Features:**
- Patient-level separation for cross-validation
- Data augmentation for minority classes
- Feature extraction pipeline

### 5. train.py

Implements training and evaluation pipeline (Section IV).

**Key Classes:**
- `ModelTrainer`: Single model training
- `CrossValidator`: 5-fold cross-validation
- `GWOTrainer`: Training with GWO optimization

**Key Features:**
- Early stopping (patience=10)
- Learning rate reduction on plateau
- Model checkpointing
- Comprehensive metrics

### 6. utils.py

Utility functions for visualization and analysis.

**Key Functions:**
- `plot_roc_curve()`: ROC curve visualization
- `plot_confusion_matrix()`: Confusion matrix plot
- `plot_gwo_convergence()`: GWO convergence curve
- `plot_model_comparison()`: Model comparison charts

## Key Implementation Details

### 1. Data Preprocessing

**Resampling (Equation from Section III.B.1):**
```python
audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
```

**STFT (Equation from Section III.B.2):**
```python
stft = librosa.stft(
    audio, 
    n_fft=512, 
    hop_length=256,  # 50% overlap
    win_length=400,  # 25ms at 16kHz
    window=signal.windows.hamming(400)
)
```

**Mel-Spectrogram (Equation from Section III.B.4):**
```python
mel_spec = librosa.feature.melspectrogram(
    y=audio, 
    sr=16000, 
    n_fft=512,
    hop_length=256, 
    n_mels=40  # As per paper
)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
```

### 2. Grey Wolf Optimizer

**Position Update (Equations from Section III.E):**
```python
# Calculate A and C coefficients
a = 2 - iteration * (2 / max_iterations)
r1, r2 = np.random.random(), np.random.random()
A = 2 * a * r1 - a
C = 2 * r2

# Update position
D = abs(C * leader_position - current_position)
new_position = leader_position - A * D
```

**Alpha, Beta, Delta Selection:**
- Alpha: Best solution
- Beta: Second best solution
- Delta: Third best solution

### 3. Attention Mechanism

**Implementation (Section III.F):**
```python
# Calculate attention scores
score = tanh(W * x + b)
attention_weights = softmax(score * u)

# Apply attention
context_vector = sum(x * attention_weights)
```

### 4. Training Configuration

**From Table 2 and Section III.J:**
```python
optimizer = Adam(
    learning_rate=0.001,  # Optimized by GWO
    weight_decay=1e-5
)

early_stopping = EarlyStopping(
    patience=10,
    restore_best_weights=True
)

batch_size = 32  # Optimized by GWO
dropout_rate = 0.3  # Optimized by GWO
```

## Paper-to-Code Mapping

| Paper Section | Implementation File | Key Functions/Classes |
|--------------|---------------------|----------------------|
| III.A (Dataset) | dataset.py | `RespiratoryDataset` |
| III.B (Preprocessing) | preprocessing.py | `AudioPreprocessor` |
| III.C (Normalization) | preprocessing.py | `FeatureNormalizer` |
| III.D (Segmentation) | preprocessing.py | `FeatureSegmenter` |
| III.E (GWO) | grey_wolf_optimizer.py | `GreyWolfOptimizer` |
| III.F (Feature Extraction) | models.py | `OptimizedCNN` |
| III.G (Segmentation) | preprocessing.py | `FeatureSegmenter` |
| III.H (GWO) | grey_wolf_optimizer.py | `GreyWolfOptimizer.optimize()` |
| III.I (Classification) | models.py | `OptimizedCNN.build_model()` |
| III.J (Implementation) | train.py | `ModelTrainer.train()` |
| IV (Results) | train.py | `CrossValidator.run_cross_validation()` |

## Reproducibility Notes

### Random Seeds
Set random seeds for reproducibility:
```python
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
```

### Data Splits
Patient-level separation ensures no data leakage:
```python
group_kfold = GroupKFold(n_splits=5)
for train_idx, val_idx in group_kfold.split(X, y, groups):
    # Train and validate
```

### Hyperparameter Optimization
GWO parameters from paper:
- Population size: 20 wolves
- Max iterations: 50
- Search space: As defined in Table 2

### Expected Results

**Binary Classification:**
- Accuracy: 0.93 ± 0.01
- F1-Score: 0.95 ± 0.01
- Precision: 0.97 ± 0.01
- Recall: 0.95 ± 0.01

**Multi-class Classification:**
- Accuracy: 0.75 ± 0.01
- F1-Score: 0.92 ± 0.01

**Model Comparison (Table 4):**
| Model | Training Acc | Validation Acc | F1-Score |
|-------|-------------|----------------|----------|
| RNN   | 0.86        | 0.85           | 0.83     |
| LSTM  | 0.88        | 0.87           | 0.87     |
| CNN   | 0.89        | 0.88           | 0.95     |
| Optimized CNN | 0.93 | 0.90           | 0.96     |

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Reduce number of GWO wolves
   - Use smaller input shape

2. **Slow Training**
   - Enable GPU acceleration
   - Reduce GWO iterations for testing
   - Use data subset for debugging

3. **Poor Performance**
   - Check data preprocessing
   - Verify class distribution
   - Ensure proper normalization

### GPU Setup
```python
# Check GPU availability
print(tf.config.list_physical_devices('GPU'))

# Enable memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

## Contact

For questions or issues, please open an issue on GitHub.
