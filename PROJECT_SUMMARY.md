# Project Summary

## Complete Implementation of "Enhanced Respiratory Condition Diagnosis Through Mel Spectrogram Features and Metaheuristic-Optimized CNNs"

This repository contains a complete, production-ready implementation of the research paper. The code is structured for easy understanding, reproducibility, and GitHub publication.

---

## What Has Been Implemented

### 1. Data Preprocessing Pipeline (Section III.B)
✅ **Resampling**: Audio signals resampled to 16kHz  
✅ **Zero-Crossing Rate (ZCR)**: For detecting inhalation/exhalation phases  
✅ **Short-Time Fourier Transform (STFT)**: 25ms Hamming window, 50% overlap, FFT size 512  
✅ **Mel-Frequency Cepstral Coefficients (MFCCs)**: 13 coefficients  
✅ **Mel-Spectrogram**: 40 Mel filter banks as input features  

**File**: `src/preprocessing.py` (Lines: 350+)

### 2. Feature Normalization (Section III.C)
✅ **Z-Score Normalization**: Zero mean, unit variance  
✅ **Min-Max Scaling**: Range [0, 1]  

**File**: `src/preprocessing.py` - `FeatureNormalizer` class

### 3. Feature Segmentation (Section III.D & G)
✅ **Fixed-Length Windowing**: 200ms windows with 50% overlap  
✅ **Spectrogram Segmentation**: Creates 40×40 input segments  

**File**: `src/preprocessing.py` - `FeatureSegmenter` class

### 4. Grey Wolf Optimizer (Section III.E & H)
✅ **Population-based Optimization**: 20 wolves  
✅ **Exploration-Exploitation Balance**: Dynamic coefficient adjustment  
✅ **Hyperparameter Search**: Learning rate, batch size, filters, kernel size, dropout, dense units  

**File**: `src/grey_wolf_optimizer.py` (Lines: 300+)

### 5. CNN Model Architectures (Section III.F & I)
✅ **Optimized CNN**: 4 convolutional blocks with batch normalization  
✅ **Attention Mechanism**: Focus on relevant time-frequency regions  
✅ **Transfer Learning**: VGG-like base model with fine-tuning  
✅ **Baseline Models**: RNN, LSTM, Simple CNN for comparison  

**File**: `src/models.py` (Lines: 500+)

### 6. Training & Evaluation (Section III.J & IV)
✅ **5-Fold Cross-Validation**: Patient-level separation  
✅ **Early Stopping**: Patience = 10 epochs  
✅ **Data Augmentation**: Pitch shift, time stretch, Gaussian noise  
✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, AUC  
✅ **Statistical Testing**: Wilcoxon signed-rank test  

**File**: `src/train.py` (Lines: 600+)

### 7. Dataset Handling (Section III.A)
✅ **ICBHI Dataset Loader**: Handles 920 recordings from 126 patients  
✅ **Patient-Level Splitting**: Prevents data leakage  
✅ **Class Imbalance Handling**: Oversampling and weighted loss  

**File**: `src/dataset.py` (Lines: 400+)

---

## File Structure

```
respiratory-sound-analysis/
│
├── README.md                      # Main documentation
├── IMPLEMENTATION_GUIDE.md        # Detailed implementation guide
├── PROJECT_SUMMARY.md            # This file
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── config.json                    # Configuration file
├── quickstart.py                  # Quick start script
├── .gitignore                     # Git ignore rules
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── main.py                   # Main pipeline script
│   ├── preprocessing.py          # Data preprocessing
│   ├── dataset.py                # Dataset loader
│   ├── models.py                 # CNN architectures
│   ├── grey_wolf_optimizer.py   # GWO implementation
│   ├── train.py                  # Training & evaluation
│   └── utils.py                  # Utility functions
│
├── notebooks/                    # Jupyter notebooks
│   └── demo.ipynb               # Interactive demo
│
├── data/                        # Dataset directory
│   └── .gitkeep
│
├── results/                     # Results directory
│   └── .gitkeep
│
├── models/                      # Saved models
│   └── .gitkeep
│
└── logs/                        # Log files
    └── .gitkeep
```

---

## Key Features

### 1. Complete Paper Implementation
Every component mentioned in the paper has been implemented:
- All preprocessing steps (Section III.B)
- All normalization techniques (Section III.C)
- GWO optimization (Section III.E)
- CNN architecture with attention (Section III.F)
- 5-fold cross-validation (Section IV)

### 2. Reproducibility
- Random seed setting for reproducibility
- Patient-level data separation
- Documented hyperparameters from Table 2
- Expected results from Tables 4, 5, and 6

### 3. Extensibility
- Modular design for easy modification
- Configuration file for parameter tuning
- Support for both binary and multiclass classification
- Easy to add new models or preprocessing techniques

### 4. Production Ready
- Error handling and logging
- Model checkpointing
- Comprehensive visualization utilities
- Command-line interface

---

## Usage Examples

### Quick Start
```bash
# Run demo with synthetic data
python quickstart.py --mode demo

# Run full pipeline
python quickstart.py --mode full
```

### Command Line
```bash
# Full pipeline
python src/main.py \
    --data_dir data/respiratory_sound_database \
    --mode full \
    --task multiclass \
    --epochs 50 \
    --batch_size 32 \
    --gwo_wolves 20 \
    --gwo_iterations 50

# Model comparison only
python src/main.py --mode compare_models

# Cross-validation only
python src/main.py --mode cross_validation
```

### Python API
```python
from src.preprocessing import AudioPreprocessor
from src.models import OptimizedCNN
from src.train import ModelTrainer

# Preprocess audio
preprocessor = AudioPreprocessor(target_sr=16000, n_mels=40)
features = preprocessor.preprocess('audio.wav')

# Create model
model = OptimizedCNN(input_shape=(40, 40, 1), num_classes=6)
model.build_model(use_attention=True)

# Train
trainer = ModelTrainer('optimized_cnn', (40, 40, 1), 6)
trainer.train(X_train, y_train, X_val, y_val)
```

---

## Expected Results (From Paper)

### Binary Classification (Healthy vs Diseased)
| Metric | Value |
|--------|-------|
| Accuracy | 0.93 ± 0.01 |
| F1-Score | 0.95 ± 0.01 |
| Precision | 0.97 ± 0.01 |
| Recall | 0.95 ± 0.01 |

### Multi-class Classification (7 Categories)
| Metric | Value |
|--------|-------|
| Accuracy | 0.75 ± 0.01 |
| F1-Score | 0.92 ± 0.01 |

### Model Comparison
| Model | Training Acc | Validation Acc | F1-Score |
|-------|-------------|----------------|----------|
| RNN | 0.86 | 0.85 | 0.83 |
| LSTM | 0.88 | 0.87 | 0.87 |
| CNN | 0.89 | 0.88 | 0.95 |
| **Optimized CNN** | **0.93** | **0.90** | **0.96** |

---

## Dependencies

### Core
- TensorFlow >= 2.13.0
- Keras >= 2.13.0
- NumPy >= 1.23.0
- SciPy >= 1.10.0

### Audio Processing
- Librosa >= 0.10.0
- SoundFile >= 0.12.0

### Machine Learning
- Scikit-learn >= 1.3.0

### Visualization
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/respiratory-sound-analysis.git
cd respiratory-sound-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

---

## Dataset

The implementation uses the ICBHI Respiratory Sound Database:
- **Source**: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
- **Recordings**: 920 audio files
- **Patients**: 126 unique patients
- **Classes**: 7 categories (Healthy, COPD, Asthma, Pneumonia, Bronchiectasis, Bronchiolitis, URTI)

---

## Citation

If you use this code, please cite:

```bibtex
@article{rajasekar2024respiratory,
  title={Enhanced Respiratory Condition Diagnosis Through Mel Spectrogram Features and Metaheuristic-Optimized CNNs},
  author={Rajasekar, SS and Othman, Manal and Karthiga, M and Saranya, K and Balusamy, Balamurugan and Khan, Firoz and Adubango, Musa Safiki and Nebhani, Naima},
  journal={Journal of Medical Systems},
  year={2024}
}
```

---

## License

MIT License - See LICENSE file for details.

---

## Support

For questions or issues:
1. Check the IMPLEMENTATION_GUIDE.md
2. Open an issue on GitHub
3. Contact the authors

---

## Acknowledgments

- ICBHI for the Respiratory Sound Database
- TensorFlow and Keras teams
- Librosa audio processing library
- Grey Wolf Optimizer original authors (Mirjalili et al.)
