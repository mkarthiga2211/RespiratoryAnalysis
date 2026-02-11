# Enhanced Respiratory Condition Diagnosis Through Mel Spectrogram Features and Metaheuristic-Optimized CNNs

This repository contains the complete implementation of the paper "Enhanced Respiratory Condition Diagnosis Through Mel Spectrogram Features and Metaheuristic-Optimized CNNs".

## Overview

This project implements an advanced deep learning framework for respiratory sound analysis and lung disease detection. The system uses:

- **Mel Spectrogram Features** for time-frequency representation
- **Optimized CNN Architecture** with Grey Wolf Optimizer (GWO) for hyperparameter tuning
- **Transfer Learning with Attention Mechanisms** for improved classification
- **5-Fold Cross-Validation** with patient-level separation

## Features

### 1. Data Preprocessing Pipeline
- **Resampling**: Standardizes audio to 16kHz
- **Zero-Crossing Rate (ZCR)**: Detects inhalation/exhalation phases
- **Short-Time Fourier Transform (STFT)**: 25ms Hamming window, 50% overlap
- **Mel-Frequency Cepstral Coefficients (MFCCs)**: 13 coefficients
- **Mel-Spectrogram**: 40 Mel filter banks

### 2. Feature Normalization
- **Z-Score Normalization**: Zero mean, unit variance
- **Min-Max Scaling**: Range [0, 1]

### 3. Feature Segmentation
- **Fixed-Length Windowing**: 200ms windows with 50% overlap

### 4. Grey Wolf Optimizer (GWO)
Metaheuristic optimization for CNN hyperparameters:
- Learning Rate: [1e-5, 1e-2]
- Batch Size: {32, 64}
- Convolutional Filters: [16, 128]
- Kernel Size: {3×3, 5×5, 7×7}
- Dropout Rate: [0.2, 0.5]
- Dense Units: [64, 512]

### 5. Model Architectures
- **Optimized CNN**: With attention mechanism and batch normalization
- **Transfer Learning CNN**: VGG-like architecture with fine-tuning
- **Baseline Models**: RNN, LSTM, Simple CNN for comparison

### 6. Training & Evaluation
- 5-Fold Cross-Validation with patient-level separation
- Early Stopping (patience=10)
- Data Augmentation (pitch shift, time stretch, Gaussian noise)
- Comprehensive metrics (Accuracy, Precision, Recall, F1-Score, AUC)

## Project Structure

```
respiratory-sound-analysis/
├── config.json                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/
│   ├── preprocessing.py        # Data preprocessing module
│   ├── dataset.py              # Dataset loader
│   ├── models.py               # CNN model architectures
│   ├── grey_wolf_optimizer.py  # GWO implementation
│   ├── train.py                # Training and evaluation
│   └── main.py                 # Main script
├── notebooks/
│   └── demo.ipynb              # Demo notebook
├── data/                       # Dataset directory
├── results/                    # Results directory
└── tests/                      # Unit tests
```

## Installation

### Prerequisites
- Python 3.9 or higher
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/respiratory-sound-analysis.git
cd respiratory-sound-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This implementation uses the ICBHI Respiratory Sound Database:
- 920 audio recordings
- 126 patients
- 8 diagnostic categories: Healthy, COPD, Asthma, Pneumonia, Bronchiectasis, Bronchiolitis, URTI

### Download Dataset
```bash
# Download from Kaggle
kaggle datasets download -d vbookshelf/respiratory-sound-database

# Or download from PhysioNet
wget https://physionet.org/files/respiratory-heartrate-dataset/1.0.0/
```

Extract the dataset to the `data/` directory.

## Usage

### Quick Start

Run the complete pipeline:
```bash
python src/main.py --data_dir data/respiratory_sound_database --mode full
```

### Command Line Options

```bash
python src/main.py \
    --data_dir data/respiratory_sound_database \
    --diagnosis_file data/demographic_info.txt \
    --mode full \
    --task multiclass \
    --results_dir results \
    --epochs 50 \
    --batch_size 32 \
    --gwo_wolves 20 \
    --gwo_iterations 50 \
    --n_splits 5 \
    --augment \
    --seed 42
```

### Modes

- `full`: Complete pipeline (preprocessing → GWO → training → evaluation)
- `preprocessing_only`: Only preprocess data
- `optimize_only`: Only run GWO hyperparameter optimization
- `train_optimized`: Train with pre-optimized hyperparameters
- `compare_models`: Compare RNN, LSTM, CNN, and Optimized CNN
- `cross_validation`: Run 5-fold cross-validation
- `train_final`: Train final model with best hyperparameters

### Tasks

- `binary`: Binary classification (Healthy vs Diseased)
- `multiclass`: Multi-class classification (7 disease categories)

## Configuration

Edit `config.json` to customize the pipeline:

```json
{
  "data_dir": "data/respiratory_sound_database",
  "diagnosis_file": "data/demographic_info.txt",
  "task": "multiclass",
  "mode": "full",
  "results_dir": "results",
  "target_shape": [40, 40],
  "epochs": 50,
  "batch_size": 32,
  "gwo_wolves": 20,
  "gwo_iterations": 50,
  "n_splits": 5,
  "augment": true,
  "seed": 42
}
```

## Results

### Expected Performance (from paper)

**Binary Classification (Healthy vs Diseased):**
- Accuracy: 0.93
- F1-Score: 0.95
- Precision: 0.97
- Recall: 0.95

**Multi-class Classification (7 categories):**
- Accuracy: 0.75
- F1-Score: 0.92

### Model Comparison

| Model | Training Accuracy | Validation Accuracy | F1-Score |
|-------|------------------|---------------------|----------|
| RNN   | 0.86             | 0.85                | 0.83     |
| LSTM  | 0.88             | 0.87                | 0.87     |
| CNN   | 0.89             | 0.88                | 0.95     |
| Optimized CNN | 0.93       | 0.90                | 0.96     |

## Key Implementation Details

### 1. Data Preprocessing
```python
from src.preprocessing import AudioPreprocessor, FeatureNormalizer

preprocessor = AudioPreprocessor(target_sr=16000, n_mels=40, n_mfcc=13)
features = preprocessor.preprocess(audio_file)
```

### 2. Grey Wolf Optimizer
```python
from src.grey_wolf_optimizer import GreyWolfOptimizer, get_search_space

search_space = get_search_space()
gwo = GreyWolfOptimizer(search_space, num_wolves=20, max_iterations=50)
best_params = gwo.optimize(X_train, y_train, X_val, y_val, input_shape, num_classes)
```

### 3. Model Training
```python
from src.train import ModelTrainer

trainer = ModelTrainer('optimized_cnn', input_shape, num_classes, best_params)
trainer.train(X_train, y_train, X_val, y_val, epochs=50)
metrics = trainer.evaluate(X_test, y_test)
```

### 4. Cross-Validation
```python
from src.train import CrossValidator

cv = CrossValidator('optimized_cnn', input_shape, num_classes, n_splits=5)
results = cv.run_cross_validation(X, y, groups, epochs=50)
```

## Notebooks

Explore the demo notebook for interactive examples:
```bash
jupyter notebook notebooks/demo.ipynb
```

## Testing

Run unit tests:
```bash
python -m pytest tests/
```

'''## Citation

If you use this code in your research, please cite:

```bibtex
@article{rajasekar2024respiratory,
  title={Enhanced Respiratory Condition Diagnosis Through Mel Spectrogram Features and Metaheuristic-Optimized CNNs},
  author={Rajasekar, SS and Othman, Manal and Karthiga, M and Saranya, K and Balusamy, Balamurugan and Khan, Firoz and Adubango, Musa Safiki and Nebhani, Naima},
  journal={Journal of Medical Systems},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ICBHI Respiratory Sound Database
- TensorFlow and Keras teams
- Librosa audio processing library

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

## References

1. ICBHI Respiratory Sound Database: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
2. Grey Wolf Optimizer: Mirjalili et al., 2014
3. TensorFlow: https://www.tensorflow.org/
4. Librosa: https://librosa.org/
