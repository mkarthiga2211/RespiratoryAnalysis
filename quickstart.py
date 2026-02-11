#!/usr/bin/env python3
"""
Quick Start Script for Respiratory Sound Analysis
Simplified interface for running experiments.
"""

import os
import sys
import argparse
import subprocess


def print_banner():
    """Print welcome banner."""
    print("="*70)
    print("  Respiratory Sound Classification - Quick Start")
    print("  Enhanced Respiratory Condition Diagnosis Through Mel")
    print("  Spectrogram Features and Metaheuristic-Optimized CNNs")
    print("="*70)


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nChecking dependencies...")
    
    required_packages = [
        'tensorflow', 'keras', 'numpy', 'librosa', 
        'scikit-learn', 'matplotlib', 'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("Dependencies installed successfully!")
    else:
        print("All dependencies are installed.")


def download_dataset():
    """Download ICBHI Respiratory Sound Database."""
    print("\n" + "="*70)
    print("Dataset Download")
    print("="*70)
    print("""
The ICBHI Respiratory Sound Database can be downloaded from:

1. Kaggle: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
2. PhysioNet: https://physionet.org/content/respiratory-heartrate-dataset/1.0.0/

Please download the dataset and extract it to the 'data/' directory.

Expected structure:
    data/
    └── respiratory_sound_database/
        ├── audio_files/
        │   ├── 101_1b1_Al_sc_Meditron.wav
        │   └── ...
        └── demographic_info.txt
""")
    
    response = input("Have you downloaded the dataset? (y/n): ")
    if response.lower() != 'y':
        print("\nPlease download the dataset first.")
        print("Run this script again after downloading.")
        sys.exit(0)


def run_demo():
    """Run a quick demo with synthetic data."""
    print("\n" + "="*70)
    print("Running Demo with Synthetic Data")
    print("="*70)
    
    import numpy as np
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import to_categorical
    
    sys.path.append('src')
    from models import OptimizedCNN
    from train import ModelTrainer
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    np.random.seed(42)
    n_samples = 200
    input_shape = (40, 40, 1)
    num_classes = 4
    
    X = np.random.randn(n_samples, *input_shape)
    y = to_categorical(np.random.randint(0, num_classes, n_samples), num_classes=num_classes)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create and train model
    print("\nTraining Optimized CNN...")
    trainer = ModelTrainer(
        model_type='optimized_cnn',
        input_shape=input_shape,
        num_classes=num_classes,
        results_dir='results/demo'
    )
    
    trainer.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=16)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    
    print("\nDemo Results:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nDemo completed successfully!")
    print("Results saved to: results/demo/")


def run_full_pipeline():
    """Run the full pipeline with real data."""
    print("\n" + "="*70)
    print("Running Full Pipeline")
    print("="*70)
    
    # Check if data exists
    data_dir = 'data/respiratory_sound_database'
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory not found: {data_dir}")
        print("Please download the dataset first.")
        return
    
    # Run main script
    cmd = [
        sys.executable, 'src/main.py',
        '--data_dir', data_dir,
        '--mode', 'full',
        '--task', 'multiclass',
        '--epochs', '50',
        '--batch_size', '32',
        '--gwo_wolves', '20',
        '--gwo_iterations', '50',
        '--augment'
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def run_model_comparison():
    """Run model comparison."""
    print("\n" + "="*70)
    print("Running Model Comparison")
    print("="*70)
    
    data_dir = 'data/respiratory_sound_database'
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory not found: {data_dir}")
        return
    
    cmd = [
        sys.executable, 'src/main.py',
        '--data_dir', data_dir,
        '--mode', 'compare_models',
        '--epochs', '50'
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}\n")
    subprocess.run(cmd)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Quick Start for Respiratory Sound Analysis'
    )
    
    parser.add_argument('--mode', type=str,
                       choices=['demo', 'full', 'compare', 'check'],
                       default='check',
                       help='Mode to run')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip dependency check')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check dependencies
    if not args.skip_deps:
        check_dependencies()
    
    # Run selected mode
    if args.mode == 'check':
        print("\n" + "="*70)
        print("System Check Complete")
        print("="*70)
        print("""
Available modes:
  1. demo    - Run quick demo with synthetic data
  2. full    - Run full pipeline with real data
  3. compare - Compare different models

Usage:
  python quickstart.py --mode demo
  python quickstart.py --mode full
  python quickstart.py --mode compare
""")
        
        response = input("\nWould you like to run a demo? (y/n): ")
        if response.lower() == 'y':
            run_demo()
    
    elif args.mode == 'demo':
        run_demo()
    
    elif args.mode == 'full':
        download_dataset()
        run_full_pipeline()
    
    elif args.mode == 'compare':
        download_dataset()
        run_model_comparison()
    
    print("\n" + "="*70)
    print("Thank you for using Respiratory Sound Analysis!")
    print("="*70)


if __name__ == "__main__":
    main()
